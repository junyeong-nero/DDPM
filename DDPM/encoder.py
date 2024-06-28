import torch
import numpy as np

class Gaussian

class NoiseSchedule:
    
    def __init__(self, size) -> None:
        self._size = size
        
        # beta
        self._variances = [i / (size + 1) for i in range(1, size + 1)]
    
        # alpha
        self._total_variances = self._calculate_total_variances()
        
        # print(self._variances)
        # print(self._total_variances)
        
    def _calculate_total_variances(self):
        alpha, curr = [], 1
        for i in range(self._size):
            curr *= 1 - self._variances[i]
            alpha.append(curr)
            
        return alpha
        
    def get_beta(self, index):
        if index >= self._size:
            raise IndexError("[get] out of index :", index, " / size :", self._size)
        return self._variances[index]
    
    def get_alpha(self, index):
        if index >= self._size:
            raise IndexError("[get] out of index :", index, " / size :", self._size)
        return self._total_variances[index]


class ForwardEncoder:

    def __init__(self, noise_schedule) -> None:
        self.noise_schedule = noise_schedule
        
    def noise(self, data, time_step):
        alpha = self.noise_schedule.get_alpha(time_step)
        epsilon = 0
        return np.sqrt(1 - alpha) * data + np.sqrt(alpha) * epsilon
        
        
    def sample_data_point(self):
        pass
    
    def sample_timestep(self):
        pass
    
    def sample_noise(self):
        pass
    
    def evaluate_noisy_latent(self):
        pass
    
    def compute_loss(self):
        pass