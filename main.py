import torch
import torch.nn as nn
import torch.optim as optim

BATCH = 8
torch.manual_seed(0)

A = torch.randn(BATCH, 1, 32, 32)
B = torch.randn(BATCH, 1, 32, 32)

norm = torch.linalg.matrix_norm(A - B)
print(norm.shape)