import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

    def forward(self, x, mode:str, weight, bias):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x, weight, bias)
        elif mode == 'denorm':
            x = self._denormalize(x, weight, bias)
        else: raise NotImplementedError
        return x


    def _get_statistics(self, x):
        reduce_dims = tuple(d for d in range(x.ndim) if d not in (0,1))
        self.mean = torch.mean(x, dim=reduce_dims, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=reduce_dims, keepdim=True, unbiased=False) + self.eps).detach()
    
    def _normalize(self, x, weight, bias):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * (weight)
            x = x + bias
        return x

    def _denormalize(self, x, weight, bias):
        if self.affine:
            x = x - bias
            x = x / (weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x



class resnet_hope_blocks(nn.Module):

    def __init__(self, device, f_size = 3, z_dim = 64, out_size=16, in_size=16):
        super(resnet_hope_blocks, self).__init__()
        self.z_dim = z_dim
        self.f_size = f_size
        self.out_size = out_size
        self.in_size = in_size
        self.device = device

        self.w1 = Parameter(torch.fmod(torch.randn((self.z_dim, self.out_size*self.f_size), device=device),2)) # 64x48 = 3072
        self.b1 = Parameter(torch.fmod(torch.randn((self.out_size*self.f_size), device=device),2)) # 48

        self.w2 = Parameter(torch.fmod(torch.randn(((17), self.in_size*self.z_dim), device=device),2)) # 17x1024 = 17408
        self.b2 = Parameter(torch.fmod(torch.randn((self.in_size*self.z_dim), device=device),2)) # 1024

        self.w3 = Parameter(torch.fmod(torch.randn((self.in_size*self.z_dim, self.in_size*self.z_dim//8), device=device),2)) # 1024x128 = 131072
        self.b3 = Parameter(torch.fmod(torch.randn((self.in_size*self.z_dim//8), device=device),2)) # 128

        self.w4 = Parameter(torch.fmod(torch.randn((self.in_size*self.z_dim//8, self.in_size*self.z_dim), device=device),2)) # 128x1024 = 131072
        self.b4 = Parameter(torch.fmod(torch.randn((self.in_size*self.z_dim), device=device),2)) # 1024


        

    def forward(self, z):

        h_in = torch.matmul(z, self.w2) + self.b2
        h_in = torch.tanh(h_in)  # Activation here
        h_in = torch.matmul(h_in, self.w3) + self.b3
        h_in = torch.tanh(h_in)
        h_in = torch.matmul(h_in, self.w4) + self.b4
        h_in = torch.tanh(h_in)
        #introduce w2, w3
        h_in = h_in.view(self.in_size, self.z_dim)

        h_final = torch.matmul(h_in, self.w1) + self.b1
        h_final = torch.tanh(h_final) # added tanh activation 

        kernel = h_final.view(self.out_size, self.in_size, self.f_size)

        return kernel

class ResNetBlock(nn.Module):

    def __init__(self, dilation = 1):
        super(ResNetBlock,self).__init__()
        self.group_norm = nn.GroupNorm(num_groups=1,num_channels=48)
        self.dilation = dilation

    def forward(self, x, conv1_w, bias):
        x = self.group_norm(x)
        out = F.relu(F.conv2d(x, conv1_w, bias, padding=self.dilation, dilation=self.dilation))
        return out 


class Embedding(nn.Module):
    def __init__(self,ab_tensor,c_tensor,unet_1d_weights, unet_1d_bias, device):
        super(Embedding, self).__init__()

        self.unet_1d_weights = unet_1d_weights
        self.unet_1d_bias = unet_1d_bias
        self.unet_1d_bias_dim = unet_1d_bias.shape[0]

        self.ab_tensor = ab_tensor.to(device)
        self.c_tensor = c_tensor.to(device)
       
        self.z_bias = nn.Parameter(torch.fmod(torch.randn(self.unet_1d_bias_dim, device=device), 2))


    def forward(self, hyper_net, pde_ts):
        bb = self.z_bias + self.unet_1d_bias
        www = []
        for i in range(3):
            ww = []
            for j in range(3):
                z_idx = i * 3 + j
                z = torch.cat((self.ab_tensor, self.c_tensor[z_idx], pde_ts), dim=0)
                ww.append(hyper_net(z))  
            www.append(torch.cat(ww, dim=1))
        exe_unet_weight = torch.cat(www, dim=0)
        w_2d = torch.einsum('oik,oil->oikl', exe_unet_weight, self.unet_1d_weights)
        
        return w_2d, bb

import math
def positional_encoding(pos: int, max_len: int = 4, eps : float  = 1e-4) -> torch.Tensor:
    base_freq = 2 * math.pi / max_len  
    return torch.tensor([
        math.sin(base_freq * (pos+eps)),
        math.cos(base_freq * (pos+eps)),
        math.sin(2 * base_freq * (pos+eps)),
        math.cos(2 * base_freq * (pos+eps))
    ], dtype=torch.float32)

class PrimaryNetwork(nn.Module):

    def __init__(self, unet_1d_weights_path = None, device = None):
        super(PrimaryNetwork, self).__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if unet_1d_weights_path is not None:
            self.load_unet_1d_weights(unet_1d_weights_path)

        self.resnet_blocks_hope = resnet_hope_blocks(device = device)
        self.dilation_list = [1, 2, 4, 8, 4, 2, 1] * 4

        self.dil_resnet_blocks = nn.ModuleList()


        for i in self.dilation_list:
                self.dil_resnet_blocks.append(ResNetBlock(dilation = i))
        
        self.zs_resnet_blocks = nn.ModuleList()

        ab_list = [(a, b) for a in range(4) for b in range(7)]  # 28 pairs

        ab_encodings = []
        for a, b in ab_list:
            enc_a = positional_encoding(a,4)   # (4,)
            enc_b = positional_encoding(b,7)   # (4,)
            ab_encodings.append(torch.cat([enc_a, enc_b]))  # (8,)

        ab_tensor = torch.stack(ab_encodings)  # (28, 8)

        # Encode c → shape (9, 4)
        c_encodings = [positional_encoding(c,9) for c in range(9)]
        c_tensor = torch.stack(c_encodings)  # (9, 4)

        i,ab = 0,0
        self.number = len(self.unet_1d_weights_list)
        
        while i < self.number:
            self.zs_resnet_blocks.append(Embedding(ab_tensor[ab], c_tensor, self.unet_1d_weights_list[i], self.unet_1d_weights_list[i+1], device=self.device))
            i +=2
            ab +=1
        self.enc_conv = nn.Conv2d(1,48, kernel_size=3,  padding = 1, bias= False)
        self.decoder_conv = nn.Conv2d(48,1, kernel_size=3,  padding = 1, bias= False)
        self.revin = RevIN(1)

        self.hyp_revin = nn.Sequential(
            nn.Linear(3,32),
            nn.Tanh(),
            nn.Linear(32,2),
            nn.Tanh()
        )

    
    def load_unet_1d_weights(self, path):
        state_dict_1d = torch.load(path, map_location=self.device)
        self.unet_1d_weights_list = [v.to(self.device) for v in state_dict_1d.values()]



    def forward(self, x, pde_params_ts):

        # print(f"input: {x.shape}")
        mask = torch.ones(pde_params_ts.shape[1], dtype=torch.bool)
        mask[1:3] = False
        pde_ts = pde_params_ts[:, mask].squeeze(0)
        pde_params = pde_params_ts[:, :3].squeeze(0)
        
        revin_wb = self.hyp_revin(pde_params).squeeze(0)
        revin_weight, revin_bias = revin_wb[0].view(1), revin_wb[1].view(1) 

        x = self.revin(x, 'norm', revin_weight, revin_bias)
        enc_out = self.enc_conv(x)
        skip_start = enc_out  

        for i in range(28):
            weight, bias = self.zs_resnet_blocks[i](self.resnet_blocks_hope, pde_ts)
            enc_out = self.dil_resnet_blocks[i](enc_out, weight, bias)

            if (i + 1) % 7 == 0:
                enc_out = enc_out + skip_start
                skip_start = enc_out  # Update skip_start to new point after skip connection

        dec_out = self.decoder_conv(enc_out)
        dec_out = self.revin( dec_out ,'denorm', revin_weight, revin_bias)

        return dec_out
