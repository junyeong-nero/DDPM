import torch

class ReverseDecoder:
        
    def __init__(self, noise_schedule, g) -> None:
        self.noise_schedule = noise_schedule
        self.g = g
        
    def denoise(self, noise_data, time_step):
        # noise_data : [B, 1, 32, 32]
        # time_step : INT
        
        batch_size = noise_data.shape[0]
        # batch_size : B
            
        with torch.no_grad():

            # step : [T - 1, T - 2, .. 2, 1, 0]
            for step in range(time_step - 1, -1, -1):
                
                t = torch.full((batch_size, ), step)
                t = t.reshape(-1, 1, 1, 1)
                # t : [B, 1, 1, 1]
        
                mu = 1 / torch.sqrt(1 - self.noise_schedule._betas[t]) * (noise_data - (self.noise_schedule._betas[t] / (1 - self.noise_schedule._alphas[t])) * self.g(noise_data, t))
                
                if step == 0:
                    # if t == 0, no add noise
                    break
                
                epsilon = torch.randn(noise_data.shape)
                noise_data = mu + torch.sqrt(self.noise_schedule._betas[t]) * epsilon
            
        return noise_data
    
    