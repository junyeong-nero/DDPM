import torch

class ForwardEncoder:

    def __init__(self, noise_schedule) -> None:
        self.noise_schedule = noise_schedule
        
    def noise(self, data, time_step):
        alpha = self.noise_schedule.get_alpha(time_step)
        
        # torch.randn ~ N(0, 1)
        epsilon = torch.randn(data.shape)
        return torch.sqrt(alpha) * data + torch.sqrt(1 - alpha) * epsilon, epsilon
