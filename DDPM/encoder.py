import torch
import numpy as np

class NoiseSchedule:
    
    def __init__(self, size) -> None:
        self._size = size
        
        # beta
        self._variances = [i / (size + 1) for i in range(1, size + 1)]
    
        # alpha
        self._total_variances = self._calculate_total_variances()
        
        print(self._variances)
        print(self._total_variances)
        
    def _calculate_total_variances(self):
        alpha, curr = [], 1
        for i in range(self._size):
            curr *= 1 - self._variances[i]
            alpha.append(curr)
            
        return alpha
        
    def get(self, index):
        return self._variances[index]

class Encoder:

    def __init__(self, noise_schedule) -> None:
        self.noise_schedule = noise_schedule