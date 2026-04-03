import argparse
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel

import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from networks.dual_fusionNet import RegistNetwork
from lib.loss import regularization_loss
from lib.utils import setup_logger
import tools
import torch.nn.functional as F
import cv2
import glob
import re

def natural_key(s):

    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', os.path.basename(s))]


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default = 'CAMUS', help='')
parser.add_argument('--dataset_root', type=str, default = '', help='dataset root dir ()')
parser.add_argument('--batch_size', type=int, default = 1, help='batch size')
parser.add_argument('--workers', type=int, default = 10, help='number of data loading workers')
parser.add_argument('--results', type=str, default = '', help='dataset root dir')
parser.add_argument('--model', type=str, default = 'trained_model.pth',  help='resume model')
parser.add_argument('--use_img_similarity', type=bool, default =True, help='')
opt = parser.parse_args()

def ensure_fd(fd):
    if not os.path.exists(fd):
        os.system('mkdir -p {}'.format(fd))



def main():
    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    if opt.dataset == 'CAMUS':
        opt.outf = './experiments/trained_models/' + opt.dataset #folder to save trained models
        opt.log_dir = './experiments/logs/' + opt.dataset + '/logtxt'#folder to save logs
        opt.results = './experiments/results/' + opt.dataset  #folder to save logs
        opt.train_info_dir = './experiments/logs/' + opt.dataset + '/train_info' #folder to save logs
        opt.repeat_epoch = 1 #number of repeat times for one epoch training
        opt.dataset_root = '/data/shenchengkang/sck/CU-Reg/test_throid/new2'
    else:
        print('Unknown dataset')
        return
    
    ensure_fd(opt.outf)
    ensure_fd(opt.log_dir)
    ensure_fd(opt.results)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    estimator = RegistNetwork(layers=[3, 8, 36, 3])

    estimator.load_state_dict(torch.load('/experiments/trained_models/throid_full_correct11/pose_model_thyroid.pth'))
    print("Checkpoint loaded.")
    estimator = estimator.cuda()
    
    
    
    test_dir = opt.dataset_root
    #print(test_dir)
    test_files = sorted(glob.glob(os.path.join(test_dir, 'images','*_slice_*.png')), key=natural_key)

    print('>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the testing set: {0}\n'.format(len(test_files)))

    criterion = regularization_loss().to(device)

    
    
    st_time = time.time()

    #保存每次测试的log文件
    logger = setup_logger('test', os.path.join(opt.log_dir, 'epoch_test_log.txt'))
    logger.info('Test time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Testing started'))
    test_count = 0
    estimator.eval()
    path = '/data/shenchengkang/sck/CU-Reg/datasets/throid/data'
    
    with torch.no_grad():
        for j, slice_path in enumerate(test_files):
            # Parse file path to get vol_idx and target_sliceidx
            vol_idx = int(os.path.basename(slice_path).split('_')[0])
           
            para_dir = os.path.join(path, '{0}_s2v_{1}.txt'.format(0,j))
            para = np.loadtxt(para_dir)
            para = np.expand_dims(para, axis=0)
            para = torch.from_numpy(para).to(device)
     
            # Construct paths for other necessary files
            volume_path = os.path.join(test_dir, f'{vol_idx}_volume.npy')


            # Load volume
            
            volume_n = np.load(volume_path)
  
            volume_np = volume_n

            volume_np = np.expand_dims(volume_np, 0)
            volume_np = np.expand_dims(volume_np, 0)
            vol_tensor = torch.from_numpy(volume_np.astype(np.float32))
            vol_tensor = vol_tensor.float()/255
     
            frame_np = cv2.imread(slice_path)

            slice_np = np.expand_dims(frame_np[..., 0], axis=0)
            frame_tensor = torch.from_numpy(slice_np.astype(np.float32))
            frame_tensor = frame_tensor.unsqueeze(0)
            frame_tensor = frame_tensor.unsqueeze(0)/255

            
            # Convert to CUDA variables
            vol_tensor = Variable(vol_tensor).cuda()
            frame_tensor = Variable(frame_tensor).cuda()

            pred_dof, out_frame_tensor1 = estimator(vol_tensor, frame_tensor, device=device)

            Bs = pred_dof.shape[0]
            mat = torch.eye(4).unsqueeze(0).repeat(Bs, 1, 1).to(device)  # (B,4,4)

            rot_mat = tools.axisangle_to_R(pred_dof[:, :3]) #pred_dof
            
            mat[:,:3, :3] = rot_mat

            mat[:,:3, 3] =pred_dof[:,3:]
            
            volume_n = torch.from_numpy(volume_n.astype(np.float32)).cuda()
            volume_n = volume_n.unsqueeze(0).unsqueeze(0)
         
            input_vol_t = volume_n.permute(0, 1, 4, 3, 2)
            directions = tools.get_ray_directions(160,160)
            directions = directions-80
            directions = directions.to(device)
            #print(mat.shape)
            point = tools.get_rays(directions,mat)
            #print(input_vol_t.shape)
            out_frame_tensor = tools.slice_operator(point, input_vol_t)


            test_count += 1

            frame_tensor = frame_tensor.squeeze()
            out_np = out_frame_tensor.cpu().detach().squeeze().numpy()
            out_np1 = out_frame_tensor1.cpu().detach().squeeze().numpy()
            out_np = out_np.reshape(160,160)

            out_np1 = out_np1*255
          
            if out_np.ndim > 2:
                out_np = out_np[0, :, :]

            cat_np = np.concatenate((frame_np[...,0], out_np), axis=0)
            
            cv2.imwrite(os.path.join(opt.results, 'frame_{}.jpg'.format(j)), cat_np)
            
if __name__ == '__main__':
    main()
