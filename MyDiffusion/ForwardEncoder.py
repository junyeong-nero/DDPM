import torch

class ForwardEncoder:

    def __init__(
        self,
        noise_schedule,
        device = None
    ):
        self.noise_schedule = noise_schedule
        self.device = device

    def noise(self, data, time_step):
        # time_step : [B]
        # data : [B, 1, 32, 32]

        alpha = self.noise_schedule._alphas[time_step]
        alpha = alpha.reshape(-1, 1, 1, 1)
        # alpha : [B, 1, 1, 1]

        epsilon = torch.randn(data.shape).to(self.device)
        # torch.randn ~ N(0, 1)

        return torch.sqrt(alpha) * data + torch.sqrt(1 - alpha) * epsilon, epsilon