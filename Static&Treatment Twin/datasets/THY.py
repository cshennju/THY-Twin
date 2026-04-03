import torch
import glob
import numpy as np
import os
import json
from .ray_utils import get_ray_directions
from einops import rearrange
import collections
from .base import BaseDataset
import imageio

class THYDataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)
        self.root_dir = root_dir

        self.read_intrinsics()

        self.read_meta()

    def read_intrinsics(self):
        w = 160
        h = 160
        self.directions = get_ray_directions(h, w)
        self.img_wh = (w, h)

    def read_meta(self):

        self.image_paths = \
            sorted(glob.glob(os.path.join(self.root_dir, 'images/*')))

        with open(os.path.join(self.root_dir,"transforms_train.json"), 'r') as f:
            meta = json.load(f,object_pairs_hook=collections.OrderedDict)
        self.poses = []
        for frame in meta['frames']:
            pose = np.array(frame['transform_matrix'])[:3, :4]            
            self.poses += [pose]

        self.poses = torch.FloatTensor(self.poses)
        self.N_frames = len(self.image_paths)
        #print(self.N_frames)
        self.rays = []
        for t in range(self.N_frames):
            img = imageio.imread(self.image_paths[t]).astype(np.float32)/255.0 # (H, W, 3) np.uint8
            
            img = rearrange(img, 'h w -> (h w) 1')
            self.rays.append(img)
        self.rays = torch.FloatTensor(np.stack(self.rays)) # (N_images, hw, ?)
