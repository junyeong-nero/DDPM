import torch

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
                # mu : [B, 1, 32, 32]

                if step == 0:
                    # if t == 0, no add noise
                    break

                epsilon = torch.randn(noise_data.shape)
                # epsilon : [B, 1, 32, 32]

                noise_data = mu + torch.sqrt(self.noise_schedule._betas[t]) * epsilon
                # noise_data : [B, 1, 32, 32]

        return noise_data

    def implicit_denoise(self, noise_data, time_step, n_jumps=10, sigma=0):
        # noise_data : [B, 1, 32, 32]
        # time_step : INT

        batch_size = noise_data.shape[0]
        gap = (time_step - 1) // n_jumps
        tau = [time_step - 1 - t * gap for t in range(n_jumps)]
        S = len(tau)
        # print(tau)

        # batch_size : B
        with torch.no_grad():

            # step : [T - 1, T - 2, .. 2, 1, 0]
            for i in range(S):

                t = torch.full((batch_size, ), tau[i])
                t = t.reshape(-1, 1, 1, 1)
                alpha_t = self.noise_schedule._alphas[t]

                alpha_t_1 = torch.full((batch_size, 1, 1, 1,), 1)
                if i < S - 1:
                    t_1 = torch.full((batch_size, ), tau[i + 1])
                    t_1 = t_1.reshape(-1, 1, 1, 1)
                    alpha_t_1 = self.noise_schedule._alphas[t_1]

                predict_noise = self.g(noise_data, t)
                first = torch.sqrt(alpha_t / alpha_t_1) * (noise_data - torch.sqrt(1 - alpha_t) * predict_noise)
                second = torch.sqrt(1 - alpha_t_1) * predict_noise

                noise_data = first + second

        return noise_data