import torch
from torch import nn
import numpy as np
import math
import torch.nn.parallel
import numpy as np

class FinerLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30, first_bias_scale=None, scale_req_grad=False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
        self.scale_req_grad = scale_req_grad
        self.first_bias_scale = first_bias_scale
        if self.first_bias_scale != None:
            self.init_first_bias()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def init_first_bias(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.bias.uniform_(-self.first_bias_scale, self.first_bias_scale)
                # print('init fbs', self.first_bias_scale)

    def generate_scale(self, x):
        if self.scale_req_grad: 
            scale = torch.abs(x) + 1
        else:
            with torch.no_grad():
                scale = torch.abs(x) + 1
        return scale
        
    def forward(self, input):
        x = self.linear(input)
        scale = self.generate_scale(x)
        out = torch.sin(self.omega_0 * scale * x)
        return out
    

class Finer(nn.Module):
    def __init__(self, in_features=3, hidden_features=64, hidden_layers=4, out_features=1, first_omega_0=30, hidden_omega_0=30.0, bias=True, 
                 first_bias_scale=None, scale_req_grad=False):
        super().__init__()
        self.net = []
        self.net.append(FinerLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0, first_bias_scale=first_bias_scale, scale_req_grad=scale_req_grad))

        for i in range(hidden_layers):
            self.net.append(FinerLayer(hidden_features, hidden_features, omega_0=hidden_omega_0, scale_req_grad=scale_req_grad))

        final_linear = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                          np.sqrt(6 / hidden_features) / hidden_omega_0)
        self.net.append(final_linear)
        self.net.append(nn.Sigmoid())
        self.net = nn.Sequential(*self.net)


    def forward(self, coords):
        output = self.net(coords)
        
        return output   

# class SineLayer(nn.Module):
#     def __init__(self, in_features, out_features, bias=True,
#                  is_first=False, omega_0=30):
#         super().__init__()
#         self.omega_0 = omega_0
#         self.is_first = is_first
        
#         self.in_features = in_features
#         self.linear = nn.Linear(in_features, out_features, bias=bias)
        
#         self.init_weights()
    
#     def init_weights(self):
#         with torch.no_grad():
#             if self.is_first:
#                 self.linear.weight.uniform_(-1 / self.in_features, 
#                                              1 / self.in_features)      
#             else:
#                 self.linear.weight.uniform_(-np.sqrt(6/self.in_features) / self.omega_0, 
#                                              np.sqrt(6/self.in_features) / self.omega_0)
        
#     def forward(self, x):
#         return torch.sin(self.omega_0 * self.linear(x))
    
# class Siren(nn.Module):
#     def __init__(self, pos_in_dims=3,D=64,hidden_layers=4, outermost_linear=True, 
#                  first_omega_0=30., hidden_omega_0=30.):
#         super().__init__()
        
#         self.net = []
#         self.net.append(SineLayer(pos_in_dims, D, 
#                                   is_first=True, omega_0=first_omega_0))

#         for i in range(hidden_layers):
#             self.net.append(SineLayer(D, D, 
#                                       is_first=False, omega_0=hidden_omega_0))
        

#         if outermost_linear:
#             final_linear = nn.Linear(D, 32)
            
#             with torch.no_grad():
#                 final_linear.weight.uniform_(-np.sqrt(6 / D) / hidden_omega_0, 
#                                               np.sqrt(6 / D) / hidden_omega_0)
#             self.net.append(final_linear)
                
#         else:
#             self.net.append(SineLayer(D, 32, 
#                                       is_first=False, omega_0=hidden_omega_0))
        
#         self.net = nn.Sequential(*self.net)


#         self.rgb_net = SineLayer(32, 32, is_first=False, omega_0=hidden_omega_0)

#         self.transient_rgb = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())

#     def forward(self, x):
#         h1 = self.net(x)
#         h3 = self.rgb_net(h1)
#         density = self.transient_rgb(h3)
#         return density
