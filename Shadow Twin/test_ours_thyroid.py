import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import cv2
import glob
import re
import time

from networks.dual_fusionNet import RegistNetwork
from lib.utils import setup_logger
import tools


def natural_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', os.path.basename(s))]


def ensure_fd(fd):
    if not os.path.exists(fd):
        os.makedirs(fd, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='THY', help='')
    parser.add_argument('--dataset_root', type=str, default='', help='dataset root dir')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--workers', type=int, default=10, help='number of data loading workers')
    parser.add_argument('--results', type=str, default='', help='results dir')
    parser.add_argument('--model', type=str, default='trained_model.pth', help='resume model')
    opt = parser.parse_args()

    if opt.dataset == 'THY':
        opt.outf = './experiments/trained_models/' + opt.dataset
        opt.log_dir = './experiments/logs/' + opt.dataset + '/logtxt'
        opt.results = './experiments/results/' + opt.dataset
        opt.dataset_root = '/data/shenchengkang/sck/Reg/test_throid/new'
    else:
        print('Unknown dataset')
        return

    ensure_fd(opt.outf)
    ensure_fd(opt.log_dir)
    ensure_fd(opt.results)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    estimator = RegistNetwork(layers=[3, 8, 36, 3]).to(device)

    estimator.load_state_dict(torch.load(opt.model, map_location=device))
    print("Checkpoint loaded.")
    estimator.eval()

    test_dir = opt.dataset_root
    test_files = sorted(glob.glob(os.path.join(test_dir, 'images', '*_slice_*.png')), key=natural_key)

    print('>>>>>>>>----------Dataset loaded!---------<<<<<<<<')
    print('length of the testing set: {0}\n'.format(len(test_files)))

    st_time = time.time()
    logger = setup_logger('test', os.path.join(opt.log_dir, 'epoch_test_log.txt'))
    logger.info('Test time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Testing started'))

    with torch.no_grad():
        for j, slice_path in enumerate(test_files):
            vol_idx = int(os.path.basename(slice_path).split('_')[0])

            volume_path = os.path.join(test_dir, f'{vol_idx}_volume.npy')
            volume_np = np.load(volume_path)

            vol_tensor = torch.from_numpy(volume_np.astype(np.float32)).to(device)
            vol_tensor = vol_tensor.unsqueeze(0).unsqueeze(0) / 255.0

            frame_np = cv2.imread(slice_path)
            slice_np = frame_np[..., 0]
            frame_tensor = torch.from_numpy(slice_np.astype(np.float32)).to(device)
            frame_tensor = frame_tensor.unsqueeze(0).unsqueeze(0) / 255.0

            pred_dof, _ = estimator(vol_tensor, frame_tensor, device=device)

            Bs = pred_dof.shape[0]
            mat = torch.eye(4).unsqueeze(0).repeat(Bs, 1, 1).to(device)
            rot_mat = tools.axisangle_to_R(pred_dof[:, :3])
            mat[:, :3, :3] = rot_mat
            mat[:, :3, 3] = pred_dof[:, 3:]

            volume_n = torch.from_numpy(volume_np.astype(np.float32)).to(device)
            volume_n = volume_n.unsqueeze(0).unsqueeze(0)

            input_vol_t = volume_n.permute(0, 1, 4, 3, 2)
            directions = tools.get_ray_directions(160, 160) - 80
            directions = directions.to(device)

            point = tools.get_rays(directions, mat)
            out_frame_tensor = tools.slice_operator(point, input_vol_t)

            out_np = out_frame_tensor.cpu().squeeze().numpy()
            if out_np.ndim > 2:
                out_np = out_np[0, :, :]

            cat_np = np.concatenate((frame_np[..., 0], out_np), axis=0)
            cv2.imwrite(os.path.join(opt.results, 'frame_{}.jpg'.format(j)), cat_np)


if __name__ == '__main__':
    main()
