import torch.nn as nn
import torch

x = torch.rand([4,1,7,7])
n = nn.ConvTranspose2d(in_channels=1,out_channels=1,kernel_size=8,stride=36)
y = n(x)
print(y)