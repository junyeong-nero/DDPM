import torch
import numpy as np

class ReverseDecoder:

    def __init__(
        self,
        noise_schedule,
        g,
        device = None
    ):
        self.noise_schedule = noise_schedule
        self.device = device
        self.g = g

    def DDPM_sampling(
        self,
        noise_data,
        timestep,
        c = None,
        w = 0,
        handler = None,
        history = False
    ):
        # noise_data : [B, 1, 32, 32]
        # c : [B]
        # timestep : INT

        origin_data = noise_data.clone()
        B = noise_data.shape[0]

        # history
        history_with_origin, history_with_prev = [], []

        with torch.no_grad():

            # step : [T - 1, T - 2, .. 2, 1, 0]
            for step in range(timestep - 1, -1, -1):

                t = torch.full((B, ), step).to(self.device)
                t = t.reshape(-1, 1, 1, 1)
                # t : [B, 1, 1, 1]

                predict_noise = (1 + w) * self.g(noise_data, t, c) - w * self.g(noise_data, t)
                mu = 1 / torch.sqrt(1 - self.noise_schedule._betas[t]) * (noise_data - (self.noise_schedule._betas[t] / (1 - self.noise_schedule._alphas[t])) * predict_noise)
                # mu : [B, 1, 32, 32]

                if step == 0:
                    # if t == 0, no add noise
                    break

                epsilon = torch.randn(noise_data.shape).to(self.device)
                new_data = mu + torch.sqrt(self.noise_schedule._betas[t]) * epsilon

                if history:
                    history_with_origin.append(torch.norm(origin_data - noise_data))
                    history_with_prev.append(torch.norm(new_data - noise_data))
                    
                if handler is not None:
                    handler(new_data, noise_data)

                noise_data = new_data

        if history:
            torch.save(torch.tensor(history_with_origin), "DDPM_origin.pt")
            torch.save(torch.tensor(history_with_prev), "DDPM_prev.pt")

        return noise_data

    def DDIM_sampling(
        self,
        noise_data,
        timestep,
        c = None,
        w = 0,
        sampling_steps = 10,
        sampling_types = "linear",
        custom_sampling_steps = None,
        handler = None,
        history = False
    ):
        # noise_data : [B, 1, 32, 32]
        # c : [B]
        # timestep : INT

        B = noise_data.shape[0]
        tau = None

        if sampling_types == "linear":
            tau = list(range(0, timestep, timestep // sampling_steps))
        if sampling_types == "exponential":
            tau = list(np.geomspace(1, timestep, timestep // sampling_steps))
        if custom_sampling_steps is not None:
            tau = custom_sampling_steps

        S = len(tau)
        origin_data = noise_data.clone()
        
        if history:
            history_with_origin, history_with_prev = [], []

        with torch.no_grad():
            # step : [T - 1, T - 2, .. 2, 1, 0]
            for i in range(S - 1, -1, -1):

                t = torch.full((B, ), tau[i]).to(self.device)
                t = t.reshape(-1, 1, 1, 1)
                # t : [B, 1, 1, 1]
                
                alpha_t = self.noise_schedule._alphas[t]
                alpha_t_1 = torch.full((B, 1, 1, 1,), 1).to(self.device)
                if i - 1 >= 0:
                    t_1 = torch.full((B, ), tau[i - 1]).to(self.device)
                    t_1 = t_1.reshape(-1, 1, 1, 1)
                    alpha_t_1 = self.noise_schedule._alphas[t_1]

                predict_noise = (1 + w) * self.g(noise_data, t, c) - w * self.g(noise_data, t)
                new_data = torch.sqrt(alpha_t_1) * ((noise_data - torch.sqrt(1 - alpha_t) * predict_noise) / torch.sqrt(alpha_t)) + torch.sqrt(1 - alpha_t_1) * predict_noise

                if history:
                    history_with_origin.append(torch.norm(origin_data - noise_data))
                    history_with_prev.append(torch.norm(new_data - noise_data))
                if handler is not None:
                    handler(new_data, noise_data)

                noise_data = new_data

        if history:
            torch.save(torch.tensor(history_with_origin), "DDIM_origin.pt")
            torch.save(torch.tensor(history_with_prev), "DDIM_prev.pt")
        return noise_data


    def DDIM_sampling_step(
        self,
        noise_data,
        t,
        predict_noise = None,
        c = None,
        w = 1,
        t_1 = None
    ):

        t = t.reshape(-1, 1, 1, 1)
        if t_1 is None:
            t_1 = torch.clamp(t - 1, min=0)

        alpha_t = self.noise_schedule._alphas[t]
        alpha_t_1 = self.noise_schedule._alphas[t_1]

        if predict_noise is None:
            predict_noise = (1 + w) * self.g(noise_data, t, c) - w * self.g(noise_data, t)
        V1 = torch.sqrt(alpha_t_1) * ((noise_data - torch.sqrt(1 - alpha_t) * predict_noise) / torch.sqrt(alpha_t))
        V2 = torch.sqrt(1 - alpha_t_1) * predict_noise

        return V1 + V2