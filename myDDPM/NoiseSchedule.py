import torch

class NoiseSchedule:
    
    def __init__(self, n_timesteps, beta_start=0.0001, beta_end=0.02) -> None:
        self._size = n_timesteps
        self._betas = torch.linspace(beta_start, beta_end, n_timesteps) #.to(device)
        self._alphas = self._calculate_alphas()
        
        # print(self._betas)
        # print(self._alphas)
        
    def _calculate_alphas(self):
        self._alphas = torch.cumprod(1 - self._betas, axis=0)
        return self._alphas
        
    def get_beta(self, index):
        if index >= self._size:
            raise IndexError("[get] out of index :", index, " / size :", self._size)
        return self._betas[index]
    
    def get_alpha(self, index):
        if index >= self._size:
            raise IndexError("[get] out of index :", index, " / size :", self._size)
        return self._alphas[index]