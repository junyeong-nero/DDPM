import torch
import numpy as np

class NoiseSchedule:

    def __init__(self, n_timesteps, beta_start=0.0001, beta_end=0.02, init_type='linear') -> None:
        self._size = n_timesteps
        if init_type == 'linear':
            self._betas = torch.linspace(beta_start, beta_end, n_timesteps)
        if init_type == 'exponential':
            self._betas = torch.from_numpy(np.geomspace(beta_start, beta_end, n_timesteps))
        self._alphas = self._calculate_alphas()

    def _calculate_alphas(self):
        self._alphas = torch.cumprod(1 - self._betas, axis=0)
        return self._alphas
    
    
if __name__ == '__main__':
    schedule = NoiseSchedule(
        n_timesteps = 1000,
        beta_start = 0.001,
        beta_end = 0.002,
        init_type = 'exponential'
    )
    
    print(schedule._betas)
    print(schedule._alphas)