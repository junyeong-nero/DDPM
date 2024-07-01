import torch

class ReverseDecoder:
        
    def __init__(self, noise_schedule, g) -> None:
        self.noise_schedule = noise_schedule
        self.g = g
        
    def denoise(self, noise_data, time_step):
        self.g.eval()
        
        # t : [T - 1, T - 2, .. 2, 1, 0]
        for t in range(time_step - 1, -1, -1):
            print(t)
            
            mu = 1 / torch.sqrt(1 - self.noise_schedule.get_beta(t)) * (noise_data - (self.noise_schedule.get_beta(t) / (1 - self.noise_schedule.get_alpha(t))) * self.g(noise_data, t))
            
            if t == 0:
                # if t == 0, no add noise
                break
            
            epsilon = torch.randn(noise_data.shape)
            noise_data = mu + torch.sqrt(self.noise_schedule.get_beta[t]) * epsilon
            
        return noise_data