import torch
from torch import nn

class DilatedBasicBlock(nn.Module):
    """Basic block for Dilated ResNet

    Args:
        in_planes (int): number of input channels
        planes (int): number of output channels
        stride (int, optional): stride of the convolution. Defaults to 1.
        activation (str, optional): activation function. Defaults to "relu".
        norm (bool, optional): whether to use group normalization. Defaults to True.
        num_groups (int, optional): number of groups for group normalization. Defaults to 1.
    """

    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        norm: bool = True,
        num_groups: int = 1,
    ):
        super().__init__()

        self.dilation = [1, 2, 4, 8, 4, 2, 1]
        dilation_layers = []
        for dil in self.dilation:
            dilation_layers.append(
                nn.Conv2d(
                    in_planes,
                    planes,
                    kernel_size=3,
                    stride=stride,
                    dilation=dil,
                    padding=dil,
                    bias=True,
                )
            )
        self.dilation_layers = nn.ModuleList(dilation_layers)
        self.norm_layers = nn.ModuleList(
            nn.GroupNorm(num_groups, num_channels=planes) if norm else nn.Identity() for dil in self.dilation
        )
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer, norm in zip(self.dilation_layers, self.norm_layers):
            out = self.activation(layer(norm(out)))
        return out + x

class ResNet2d(nn.Module):
    def __init__(self, hidden_channels=48, in_classes=1, out_channels=1, pde_dim=4, mlp_hidden_dim=128):
        super().__init__()
        self.env_conv = nn.Conv2d(in_channels=in_classes, out_channels=hidden_channels, kernel_size=3, padding=1, bias=False)

        # MLP for pde_params
        self.pde_mlp = nn.Sequential(
            nn.Linear(pde_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, hidden_channels),
        )

        self.blocks = nn.Sequential(
            DilatedBasicBlock(hidden_channels, hidden_channels, stride=1, norm=True, num_groups=1),
            DilatedBasicBlock(hidden_channels, hidden_channels, stride=1, norm=True, num_groups=1),
            DilatedBasicBlock(hidden_channels, hidden_channels, stride=1, norm=True, num_groups=1),
            DilatedBasicBlock(hidden_channels, hidden_channels, stride=1, norm=True, num_groups=1),
        )

        self.dec_env = nn.Conv2d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x, pde_params):

        enc = self.env_conv(x) 
        pde_feat = self.pde_mlp(pde_params)  
        pde_feat = pde_feat.view(-1, 1, 1)    

        enc = enc + pde_feat
        blocks_out = self.blocks(enc)
        out = self.dec_env(blocks_out)
        return out
