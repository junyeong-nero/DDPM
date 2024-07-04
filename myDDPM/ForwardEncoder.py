import torch

class ForwardEncoder:

    def __init__(self, noise_schedule) -> None:
        self.noise_schedule = noise_schedule
        
    def noise(self, data, time_step):
        # data : [B, 1, 28, 28]

        # alpha : [B, 1, 1, 1]
        alpha = self.noise_schedule._alphas[time_step]
        alpha = alpha.reshape(-1, 1, 1, 1)
        
        # torch.randn ~ N(0, 1)
        epsilon = torch.randn(data.shape)
        
        return torch.sqrt(alpha) * data + torch.sqrt(1 - alpha) * epsilon, epsilon
