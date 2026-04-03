import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

#FILTER_SIZE = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
FILTER_SIZE = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        nn.init.normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        nn.init.normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.normal_(m.weight.data, mean=1, std=0.02)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, mean=1, std=0.02)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        nn.init.normal_(m.weight.data, mean=1, std=0.02)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)
                
def make_layers_instance_norm(norm=True):
    layers = []
    in_channels = 1
    for v in FILTER_SIZE:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if norm:
                layers += [conv2d, nn.InstanceNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
# def make_layers_instance_norm():
#     #u_net = unet()
#     #mobile_net = models.mobilenet_v2(pretrained=False)
#     #mobile_net.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#     res_net = models.resnet18(pretrained=False)
#     res_net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#     res_net.avgpool = nn.AdaptiveAvgPool2d(output_size=(5, 5))
#     # res_net.fc = nn.Sequential(nn.Linear(in_features=512, out_features=512, bias=True),
#     #                            nn.ReLU(True),
#     # )
#     res_net.fc = nn.Sequential()                        

#     return res_net

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )   
class unet(nn.Module):
    def __init__(self):
        super().__init__()
            
        self.dconv_down1 = double_conv(1, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)
        self.dconv_down5 = double_conv(512, 1024)        

        self.maxpool = nn.MaxPool2d(2)
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        # self.dconv_up3 = double_conv(256 + 512, 256)
        # self.dconv_up2 = double_conv(128 + 256, 128)
        # self.dconv_up1 = double_conv(128 + 64, 64)
        # #self.conv_down = nn.Conv2d(64, 80, 1,stride=2)
        # self.conv_last = nn.Conv2d(64, 1, 1)
        # self.sigmoid = torch.nn.Sigmoid()
        
        #self.conv_last = nn.Conv2d(64, n_class, 1)
        
        
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)

        conv5 = self.dconv_down5(x)
        out = self.maxpool(conv5)
        
        # x = self.upsample(x)        
        # x = torch.cat([x, conv3], dim=1)
        
        # x = self.dconv_up3(x)
        # x = self.upsample(x)        
        # x = torch.cat([x, conv2], dim=1)       

        # x = self.dconv_up2(x)
        # x = self.upsample(x)        
        # x = torch.cat([x, conv1], dim=1)   
        
        # out = self.dconv_up1(x)
        # x = self.maxpool(conv2)
        # out = self.conv_last(out)
        # #out = self.conv_down(out)
        # #out = self.maxpool(out)   
        # out = self.sigmoid(out)

        return out

class Baseline_vgg(nn.Module):

    def __init__(self, features, num_classes=9, fc_size = 512, init_weights=True, device=''):
        super(Baseline_vgg, self).__init__()
        
        self.fc_size = fc_size
        self.device = device
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((5, 5))

        self.feature_fc = nn.Sequential(
            nn.Linear(FILTER_SIZE[-2] * 5 * 5, fc_size),
            nn.ReLU(True),
            nn.Linear(fc_size, fc_size),
            nn.ReLU(True),
        )

        self.pt1 = nn.Linear(fc_size, int(num_classes[0]))
        self.pt2 = nn.Linear(fc_size, int(num_classes[1]))
        self.pt3 = nn.Linear(fc_size, int(num_classes[2]))
        
                
    def forward(self, x):
        x = torch.squeeze(x, dim=0)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.feature_fc(x)

        pt1 = self.pt1(x)
        pt2 = self.pt2(x)
        pt3 = self.pt3(x)

        return pt1, pt2, pt3, 1
    
    
class Proposed_vgg(nn.Module):
    def __init__(self, features, num_classes=9, fc_size = 512, init_weights=True, device=''):
        super(Proposed_vgg, self).__init__()

        self.fc_size = fc_size
        self.device = device
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((5, 5))

        self.feature_fc = nn.Sequential(
            nn.Linear(FILTER_SIZE[-2] * 5 * 5, fc_size),
            nn.ReLU(True),
            nn.Linear(fc_size, fc_size),
            nn.ReLU(True),
        )
        # self.feature_fc = nn.Sequential(
        #     nn.Linear(1000, fc_size),
        #     nn.ReLU(True),
        #     nn.Linear(fc_size, fc_size),
        #     nn.ReLU(True),
        # )
        # self.feature_fc = nn.Sequential(
        #      nn.Linear(fc_size, fc_size),
        #      nn.ReLU(True),
        # )


        self.alpha = nn.Sequential(
                nn.Linear(fc_size, fc_size//2),
                nn.ReLU(True),
                nn.Linear(fc_size//2, fc_size//2),
                nn.Sigmoid()
                )
        
        self.beta = nn.Sequential(
                nn.Linear(fc_size, fc_size//2),
                nn.ReLU(True),
                nn.Linear(fc_size//2, fc_size//2),
                nn.Sigmoid()
                )
        
        self.prediction = nn.Sequential(
            nn.Conv2d(fc_size*2, fc_size, kernel_size=1, padding=0),
            nn.ReLU(True),
            nn.Conv2d(fc_size, fc_size, kernel_size=1, padding=0),
            nn.ReLU(True),
        )

        self.pt1 = nn.Linear(fc_size, int(num_classes[0]))
        self.pt2 = nn.Linear(fc_size, int(num_classes[1]))
        self.pt3 = nn.Linear(fc_size, int(num_classes[2]))
        
    def forward(self, x):
        #print('sdsdss',x.size())
        #x = torch.squeeze(x, dim=0)
        
        B,C,H,W = x.size()
        
        x = self.features(x) ##(200,512,5,5)
        #print(x.shape)
        x = self.avgpool(x) 
        #print(x.shape)
        x = x.view(x.size(0), -1)##(200,512*5*5)
        x = self.feature_fc(x) ##(200,512)
        
        alpha = self.alpha(x)##(200,256)
        beta = self.beta(x)##(200,256)
        attention = torch.matmul(alpha, beta.permute(1,0))##(200,200)
        attention_repeat = torch.unsqueeze(attention,-1).repeat(1, 1, self.fc_size)##(200,200,512)
        attention_sum = attention.sum(1, keepdim=True).repeat(1, self.fc_size)##(200,512)

        x_i = torch.unsqueeze(x,0).repeat(B, 1, 1)##(200,200,512)
        x_j = x_i.permute(1,0,2)##(200,200,512)
        x_ij = torch.cat((x_i, x_j), dim=-1)##(200,200,1024)
        x_ij = torch.unsqueeze(x_ij.permute(2,0,1),0)##(1,1024,200,200)
        x_ij = self.prediction(x_ij)##(1,512,200,200)
        x_ij = torch.squeeze(x_ij, dim=0).permute(1,2,0)##(200,200,512)
        x_ij = x_ij*attention_repeat##(200,200,512)
        pred = x_ij.sum(1)/attention_sum##(200,512)
        
        pt1 = self.pt1(pred)
        pt2 = self.pt2(pred)
        pt3 = self.pt3(pred)##(200,3)

        return pt1, pt2, pt3, attention