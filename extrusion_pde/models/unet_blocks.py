import torch.nn as nn
import torch.nn.functional as F

class UnetBlock(nn.Module):

    def __init__(self, out_size=32):
        super(UnetBlock,self).__init__()
        self.out_size = out_size
        self.bn1 = nn.BatchNorm2d(self.out_size)
        self.bn2 = nn.BatchNorm2d(self.out_size)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)

    def forward(self, x, conv2_w, conv1_w = None):
        if conv1_w is None:
            out = F.tanh(self.conv1(x))
        else:
            out = F.tanh(self.bn1(F.conv2d(x, conv1_w, stride=1, padding=1)))

        out = self.bn2(F.conv2d(out, conv2_w, padding=1))
        out = F.tanh(out)

        return out

class upconv(nn.Module):
    def __init__(self):
        super(upconv, self).__init__()
    
    def forward(self, x, upconv_w, upconv_b):
        out = F.conv_transpose2d(x, upconv_w, bias=upconv_b, stride=2)
        return out