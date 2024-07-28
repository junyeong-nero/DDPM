import torch
from myDDPM.UNet import SelfAttentionBlock

block = SelfAttentionBlock(in_channels=64, out_channels=64)

X = torch.randn((8, 2, 32, 32))
X = block(X, X, X)

print(X.shape)