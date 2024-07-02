import torch

class ReverseDecoder:
        
    def __init__(self, noise_schedule, g) -> None:
        self.noise_schedule = noise_schedule
        self.g = g
        
    def denoise(self, noise_data, time_step):
        with torch.no_grad():
            # t : [T - 1, T - 2, .. 2, 1, 0]
            for t in range(time_step - 1, -1, -1):
                
                print(t)
                t = torch.tensor(t)
        
                mu = 1 / torch.sqrt(1 - self.noise_schedule._betas[t]) * (noise_data - (self.noise_schedule._betas[t] / (1 - self.noise_schedule._alphas[t])) * self.g(noise_data, t))
                
                if t == 0:
                    # if t == 0, no add noise
                    break
                
                epsilon = torch.randn(noise_data.shape)
                noise_data = mu + torch.sqrt(self.noise_schedule._betas[t]) * epsilon
            
        return noise_data
    
    