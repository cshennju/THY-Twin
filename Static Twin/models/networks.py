import torch
from torch import nn
import tinycudann as tcnn
import numpy as np

class NGP(nn.Module):
    def __init__(self, rgb_act='Sigmoid'):
        super().__init__()

        self.rgb_act = rgb_act
        scale = 0.5
        # constants
        # L = 5; F = 2; log2_T = 10; N_min = 5
        # b = np.exp(np.log(160*scale/N_min)/(L-1))#heart

        # L = 16; F = 2; log2_T = 19; N_min = 16
        # b = np.exp(np.log(2048*scale/N_min)/(L-1)) #ori

        # L = 20; F = 2; log2_T = 19; N_min = 1
        # b = np.exp(np.log(80*scale/N_min)/(L-1))

        L = 8; F = 2; log2_T = 10; N_min = 2
        b = np.exp(np.log(20*scale/N_min)/(L-1))

        
        # L = 10; F = 2; log2_T = 6; N_min = 2 ##l=10,log2_t=10 nmin=2
        # b = np.exp(np.log(20*scale/N_min)/(L-1))  # 512 23,90,65
        # L = 8; F = 4; log2_T = 19; N_min = 2
        # b = np.exp(np.log(80*scale/N_min)/(L-1))
        # L = 8; F = 4; log2_T = 19; N_min = 2  ##160 24,48,67 
        # b = np.exp(np.log(160*scale/N_min)/(L-1))
        print(f'GridEncoding: Nmin={N_min} b={b:.5f} F={F} T=2^{log2_T} L={L}')

        self.xyz_encoder = \
            tcnn.NetworkWithInputEncoding(
                n_input_dims=3, n_output_dims=16,
                encoding_config={
                    "otype": "Grid",
	                "type": "Hash",
                    "n_levels": L,
                    "n_features_per_level": F,
                    "log2_hashmap_size": log2_T,
                    "base_resolution": N_min,
                    "per_level_scale": b,
                    "interpolation": "Linear"
                },
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 1,
                }
            )

        self.rgb_net = \
            tcnn.Network(
                n_input_dims=16, n_output_dims=1,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": self.rgb_act, #
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
                }
            )
        # hidden_layers=2
        # d=32
        # layers = []
        # for i in range(hidden_layers):

        #     if i == 0:
        #         l = nn.Linear(3, d)
        #     elif 0 < i < hidden_layers-1:
        #         l = nn.Linear(d, d)

        #     act_ = nn.ReLU(True)
            

        #     if i < hidden_layers-1:
        #         layers += [l, act_]
        #     else:
        #         layers += [nn.Linear(d,d), nn.ReLU()]

        # self.net = nn.Sequential(*layers)
        # self.rgb = nn.Sequential(nn.Linear(d, 1), nn.Sigmoid())
        # self.weight = nn.Sequential(nn.Linear(d, 1), nn.Sigmoid())

    def forward(self, x, **kwargs):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            d: (N, 3) directions

        Outputs:
            sigmas: (N)
            rgbs: (N, 3)
        """
        h = self.xyz_encoder(x)
        rgbs = self.rgb_net(h)
        # h1 = self.net(x)
        # h2 =self.rgb(h1)
        # w = self.weight(h1)
        # rgbs = rgb + w*h2


        return rgbs