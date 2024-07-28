import torch
from myDDPM.UNet import SelfAttentionBlock

block = SelfAttentionBlock(in_channels=1, out_channels=1)

X = torch.randn((8, 1, 32, 32))
X = block(query=X, key=X, value=X)

print(X.shape)