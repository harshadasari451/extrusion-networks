import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
class Unet_blocks_hyp(nn.Module):

    def __init__(self, f_size = 3, z_dim = 64, out_size=32, in_size=32):
        super(Unet_blocks_hyp, self).__init__()
        self.z_dim = z_dim
        self.f_size = f_size
        self.out_size = out_size
        self.in_size = in_size

        self.w1 = Parameter(torch.fmod(torch.randn((self.z_dim, self.out_size*self.f_size)).cuda(),2))
        self.b1 = Parameter(torch.fmod(torch.randn((self.out_size*self.f_size)).cuda(),2))

        self.hyp_linear = nn.Linear(3,8).cuda()

        self.w2 = Parameter(torch.fmod(torch.randn(((self.z_dim + 8), self.in_size*self.z_dim)).cuda(),2))
        self.b2 = Parameter(torch.fmod(torch.randn((self.in_size*self.z_dim)).cuda(),2))

    def forward(self, z, pde_inputs):

        pde_features = self.hyp_linear(pde_inputs)

        z_pde_features = torch.cat((z, pde_features), dim=0)

        h_in = torch.matmul(z_pde_features, self.w2) + self.b2
        h_in = h_in.view(self.in_size, self.z_dim)

        h_final = torch.matmul(h_in, self.w1) + self.b1
        kernel = h_final.view(self.out_size, self.in_size, self.f_size)

        return kernel

class Unet_upconv_hyp(nn.Module):

    def __init__(self, f_size = 2, z_dim = 64, out_size=32, in_size=32):
        super(Unet_upconv_hyp, self).__init__()
        self.z_dim = z_dim
        self.f_size = f_size
        self.out_size = out_size
        self.in_size = in_size

        self.w1 = Parameter(torch.fmod(torch.randn((self.z_dim, self.out_size*self.f_size)).cuda(),2))
        self.b1 = Parameter(torch.fmod(torch.randn((self.out_size*self.f_size)).cuda(),2))

        self.hyp_linear = nn.Linear(3,8).cuda()

        self.w2 = Parameter(torch.fmod(torch.randn(((self.z_dim+8), self.in_size*self.z_dim)).cuda(),2))
        self.b2 = Parameter(torch.fmod(torch.randn((self.in_size*self.z_dim)).cuda(),2))


    def forward(self, z, pde_inputs):

        pde_features = self.hyp_linear(pde_inputs)
        z_pde_features = torch.cat((z, pde_features), dim=0)

        h_in = torch.matmul(z_pde_features, self.w2) + self.b2
        h_in = h_in.view(self.in_size, self.z_dim)

        h_final = torch.matmul(h_in, self.w1) + self.b1
        kernel = h_final.view(self.out_size, self.in_size, self.f_size)

        return kernel