import torch
from myDDPM.modules.UNet import SelfAttentionBlock

block = SelfAttentionBlock(in_channels=64, out_channels=64)

X = torch.randn((1, 64, 32, 32))
X = block(X)

print(X.shape)