import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .unet_blocks import UnetBlock, upconv
from .hyp_func import Unet_blocks_hyp, Unet_upconv_hyp



class upconv_Embedding(nn.Module):

    def __init__(self, z_num, z_dim,unet_1d_weights, unet_1d_bias):
        super(upconv_Embedding, self).__init__()

        self.z_list = nn.ParameterList()
        self.z_num = z_num
        self.z_dim = z_dim
        self.unet_1d_weights = unet_1d_weights
        self.unet_1d_bias = unet_1d_bias
        self.unet_1d_bias_dim = unet_1d_bias.shape[0]

        h,k = self.z_num

        for i in range(h):
            for j in range(k):
                self.z_list.append(Parameter(torch.fmod(torch.randn(self.z_dim).cuda(), 2)))
        
       
        self.z_bias = nn.Parameter(torch.fmod(torch.randn(self.unet_1d_bias_dim).cuda(), 2))

    def forward(self, hyper_net, pde_inputs):
        ww = []
        bb = self.z_bias + self.unet_1d_bias
        h, k = self.z_num
        for i in range(h):
            w = []
            for j in range(k):
                w.append(hyper_net(self.z_list[i*k + j], pde_inputs))
            ww.append(torch.cat(w, dim=1))    
        exe_unet_weight = torch.cat(ww,dim=0)
        w_2d = torch.einsum('oik,oil->oikl', exe_unet_weight, self.unet_1d_weights)
        
        return w_2d, bb
    
class Embedding(nn.Module):

    def __init__(self, z_num, z_dim,unet_1d_weights, device):
        super(Embedding, self).__init__()

        self.z_list = nn.ParameterList()
        self.z_num = z_num
        self.z_dim = z_dim
        self.unet_1d_weights = unet_1d_weights

        h,k = self.z_num

        for i in range(h):
            for j in range(k):
                self.z_list.append(Parameter(torch.fmod(torch.randn(self.z_dim, device=device), 2)))


    def forward(self, hyper_net, pde_inputs):
        ww = []
        h, k = self.z_num
        for i in range(h):
            w = []
            for j in range(k):
                w.append(hyper_net(self.z_list[i*k + j], pde_inputs))
            ww.append(torch.cat(w, dim=1))
        exe_unet_weight = torch.cat(ww,dim=0)

        w_2d = torch.einsum('oik,oil->oikl', exe_unet_weight, self.unet_1d_weights)
        
        return w_2d


class PrimaryNetwork(nn.Module):

    def __init__(self, z_dim=64, unet_1d_weights_path = None, device = None):
        super(PrimaryNetwork, self).__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if unet_1d_weights_path is not None:
            self.load_unet_1d_weights(unet_1d_weights_path)

        self.z_dim = z_dim
        self.unet_block_hope = Unet_blocks_hyp(z_dim=self.z_dim)
        self.unet_upconv_hope = Unet_upconv_hyp(z_dim=self.z_dim)
        
        self.zs_size = [[0,0], [1, 1], [2, 1], [2, 2], [4, 2], [4, 4], [8, 4], [8, 8], [16, 8], [16, 16], [16, 8],[0, 0],
                        [8, 16], [8, 8], [8, 4], [0, 0], [4, 8], [4, 4], [4, 2], [0, 0], [2, 4], [2, 2], [2, 1], [0, 0],
                        [1, 2], [1, 1]]
        
        self.filter_size = [32,64,128,256,512,256,128,64,32]
        #  self.filter_size = [[32,64],[64,128],[128,256],[256,512],[512,256],[256,128],[128,64],[64,32],[32,32]]

        self.unet_blocks = nn.ModuleList()
        self.upconv = nn.ModuleList()

        for i in range(9):
                self.unet_blocks.append(UnetBlock(self.filter_size[i]))
        
        for i in range(4):
            self.upconv.append(upconv())
        
        self.zs_unetblocks = nn.ModuleList()
        self.zs_upconv = nn.ModuleList()
        i = 1
        while i <= 25:
            if i in (10,14,18,22):
                upconv_module = upconv_Embedding(self.zs_size[i], self.z_dim, self.unet_1d_weights_list[i], self.unet_1d_weights_list[i+1])
                self.zs_upconv.append(upconv_module)
                i+=2 # skiping to next i
            else: 
                self.zs_unetblocks.append(Embedding(self.zs_size[i], self.z_dim, self.unet_1d_weights_list[i], device=self.device))
                i+=1

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.final_layer = nn.Conv2d(32,1,kernel_size=1)
    
    def load_unet_1d_weights(self, path):
        state_dict_1d = torch.load(path, map_location=self.device)
        self.unet_1d_weights_list = [v.to(self.device) for v in state_dict_1d.values()]



    def forward(self, x, pde_inputs):
        # encoder_part
        enc1 = self.unet_blocks[0](x, self.zs_unetblocks[0](self.unet_block_hope, pde_inputs))
        enc2 = self.unet_blocks[1](self.pool1(enc1), self.zs_unetblocks[2](self.unet_block_hope, pde_inputs), self.zs_unetblocks[1](self.unet_block_hope, pde_inputs))
        enc3 = self.unet_blocks[2](self.pool2(enc2),self.zs_unetblocks[4](self.unet_block_hope, pde_inputs),  self.zs_unetblocks[3](self.unet_block_hope, pde_inputs))
        enc4 = self.unet_blocks[3](self.pool3(enc3), self.zs_unetblocks[6](self.unet_block_hope, pde_inputs), self.zs_unetblocks[5](self.unet_block_hope, pde_inputs))

        bottleneck = self.unet_blocks[4](self.pool4(enc4),self.zs_unetblocks[8](self.unet_block_hope, pde_inputs) ,self.zs_unetblocks[7](self.unet_block_hope, pde_inputs))

        # decoder_part
        w, b = self.zs_upconv[0](self.unet_upconv_hope, pde_inputs)
    
        dec4 = self.upconv[0](bottleneck, w, b)

        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.unet_blocks[5](dec4, self.zs_unetblocks[10](self.unet_block_hope, pde_inputs),self.zs_unetblocks[9](self.unet_block_hope, pde_inputs))
    
        w, b = self.zs_upconv[1](self.unet_upconv_hope, pde_inputs)
        dec3 = self.upconv[1](dec4, w, b) 
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.unet_blocks[6](dec3, self.zs_unetblocks[12](self.unet_block_hope, pde_inputs),self.zs_unetblocks[11](self.unet_block_hope, pde_inputs))
        w, b = self.zs_upconv[2](self.unet_upconv_hope, pde_inputs)
        dec2 = self.upconv[2](dec3, w, b)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.unet_blocks[7](dec2, self.zs_unetblocks[14](self.unet_block_hope, pde_inputs),self.zs_unetblocks[13](self.unet_block_hope, pde_inputs))
        w, b = self.zs_upconv[3](self.unet_upconv_hope, pde_inputs)
        dec1 = self.upconv[3](dec2, w, b)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.unet_blocks[8](dec1, self.zs_unetblocks[16](self.unet_block_hope, pde_inputs),self.zs_unetblocks[15](self.unet_block_hope, pde_inputs))
        # final layer
        x = self.final_layer(dec1)


        return x