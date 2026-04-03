import torch
from torch import nn
from opt import get_opts
import os
import glob
import imageio
import numpy as np
import cv2
from einops import rearrange

# data
from torch.utils.data import DataLoader
from datasets import dataset_dict
from datasets.ray_utils import axisangle_to_R, get_rays

from models.networks_siren import Finer

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from losses import NeRFLoss
from ssimloss import S3IM
from ssimloss import SSIM
# metrics
from torchmetrics import PeakSignalNoiseRatio

# pytorch-lightning
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint

from utils import slim_ckpt, load_ckpt

import warnings; warnings.filterwarnings("ignore")

class THYSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.loss = S3IM(kernel_size=4, stride=4, repeat_time=10, patch_height=90, patch_width=90)
        self.ssim_loss = SSIM(window_size=4, stride=4)
        self.train_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1)        
        self.model = Finer()

    def forward(self, batch, split):
        if split=='train':
            poses = self.poses[batch['img_idxs']]
            directions = self.directions[batch['pix_idxs']]
        else:
            poses = batch['pose']
            poses = poses.unsqueeze(0)
            directions = self.directions

        dR = axisangle_to_R(self.dR[batch['img_idxs']])
        poses[..., :3] = dR @ poses[..., :3]
        poses[..., 3] += self.dT[batch['img_idxs']]

        rays_d = get_rays(directions, poses)

        return self.model(rays_d)

    def setup(self, stage):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir,
                  'downsample': self.hparams.downsample}
        self.train_dataset = dataset(split=self.hparams.split, **kwargs)
        self.train_dataset.batch_size = self.hparams.batch_size
        self.test_dataset = dataset(split='test', **kwargs)

    def configure_optimizers(self):
        # define additional parameters
        self.register_buffer('directions', self.train_dataset.directions.to(self.device))
        self.register_buffer('poses', self.train_dataset.poses.to(self.device))

        N = len(self.train_dataset.poses)
        self.register_parameter('dR',
            nn.Parameter(torch.zeros(N, 3, device=self.device)))
        self.register_parameter('dT',
            nn.Parameter(torch.zeros(N, 3, device=self.device)))

        load_ckpt(self.model, self.hparams.weight_path)

    
        grp1 = {'params': [p for n,p in self.named_parameters() if n not in ['dR','dT']],
                'lr': self.hparams.lr, 'eps': 1e-15}
        grp2 = {'params': [self.dR, self.dT],
                'lr': 1e-4}
        optimizer = AdamW([grp1, grp2])
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.num_epochs,
            eta_min=self.hparams.lr/30
        )
        self.net_opt = optimizer
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          num_workers=2,
                          persistent_workers=False,
                          batch_size=None,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset,
                          num_workers=2,
                          batch_size=None,
                          pin_memory=True)

    def training_step(self, batch, batch_nb, *args):
        results = self(batch, split='train')
        loss_d = self.loss(results, batch['rgb'])
        loss = loss_d.mean()
        return loss

    def on_validation_start(self) -> None:
        torch.cuda.empty_cache()
        self._val_psnr_list: list[torch.Tensor] = []
        if not self.hparams.no_save_test:
            self.val_dir = f'/data/shenchengkang/sck/thyroid/results/{self.hparams.dataset_name}/{self.hparams.exp_name}'
            os.makedirs(self.val_dir, exist_ok=True)

    def validation_step(self, batch, batch_nb):
        rgb_gt = batch['rgb']
        results = self(batch, split='test')

        logs = {}
        # compute each metric per image
        self.val_psnr(results, rgb_gt)
        psnr_val = self.val_psnr.compute()
        self.val_psnr.reset()
        self._val_psnr_list.append(psnr_val)

        w, h = self.train_dataset.img_wh
        if not self.hparams.no_save_test: # save test image to disk
            idx = batch['img_idxs']
            rgb_pred = rearrange(results.cpu().numpy(), '(h w) c -> h w c', h=h)
            rgb_pred = (rgb_pred*255).astype(np.uint8)
            imageio.imsave(os.path.join(self.val_dir, f'{idx:03d}.png'), rgb_pred[:, :, 0])
        return logs

    def on_validation_epoch_end(self) -> None:
        # stack all per-step PSNRs
        psnrs = torch.stack(self._val_psnr_list)
        # gather across GPUs if in DDP
        psnrs_all = self.all_gather(psnrs)
        mean_psnr = psnrs_all.mean()
        print(mean_psnr)
        self.log('test/psnr', mean_psnr, prog_bar=True)



if __name__ == '__main__':
    hparams = get_opts()
    if hparams.val_only and (not hparams.ckpt_path):
        raise ValueError('You need to provide a @ckpt_path for validation!')
    system = THYSystem(hparams)

    ckpt_cb = ModelCheckpoint(dirpath=f'/data/shenchengkang/sck/thyroid/ckpts/{hparams.dataset_name}/{hparams.exp_name}', #/home/test/thyroid/ckpts
                              filename='{epoch:d}',
                              save_weights_only=True,
                              every_n_epochs=hparams.num_epochs,
                              save_on_train_epoch_end=True,
                              save_top_k=-1)
    callbacks = [ckpt_cb, TQDMProgressBar(refresh_rate=0)]

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      check_val_every_n_epoch=hparams.num_epochs,
                      callbacks=callbacks,
                      #logger=logger,
                      logger=False,
                      enable_model_summary=False,
                      accelerator='gpu',
                      devices=hparams.num_gpus,
                      #strategy="auto",
                      num_sanity_val_steps=-1 if hparams.val_only else 0,
                      precision=16,
                      log_every_n_steps=1)

    trainer.fit(system, ckpt_path=hparams.ckpt_path)

    if not hparams.val_only: # save slimmed ckpt for the last epoch
        ckpt_ = \
            slim_ckpt(f'/data/shenchengkang/sck/thyroid/ckpts/{hparams.dataset_name}/{hparams.exp_name}/epoch={hparams.num_epochs-1}.ckpt',
                      save_poses=True)
        torch.save(ckpt_, f'/data/shenchengkang/sck/thyroid/ckpts/{hparams.dataset_name}/{hparams.exp_name}/epoch={hparams.num_epochs-1}_slim.ckpt')