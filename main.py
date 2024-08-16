import torch

A = torch.randn((32, 3, 32, 32))
B = torch.randn((32, 3, 32, 32))
C = torch.linalg.matrix_norm(A - B)
print(C.shape)

# TIME_STEPS = 1000
# extract_time_step = 100

# diff_norm = reversed(torch.load("DDIM_origin.pt", weights_only=False))
# gradient = []
# for i in range(1, len(diff_norm)):
#     gradient.append(diff_norm[i] - diff_norm[i - 1])
    
# gradient = torch.tensor(gradient)
# print(gradient)

# THRESHOLD_INFLECTION = int(torch.argmin(gradient).item() * (TIME_STEPS / extract_time_step))
# print(THRESHOLD_INFLECTION)