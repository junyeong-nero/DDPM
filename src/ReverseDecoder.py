import torch

class ReverseDecoder:

    def __init__(
        self,
        noise_schedule,
        g
    ) -> None:
        self.noise_schedule = noise_schedule
        self.g = g

    def DDPM_sampling(
        self,
        noise_data,
        time_step,
        c = None,
        w = 0
    ):
        # noise_data : [B, 1, 32, 32]
        # c : [B]
        # time_step : INT

        batch_size = noise_data.shape[0]
        # batch_size : B
        
        history_with_origin = []
        history_with_prev = []

        with torch.no_grad():

            # step : [T - 1, T - 2, .. 2, 1, 0]
            for step in range(time_step - 1, -1, -1):

                t = torch.full((batch_size, ), step)
                t = t.reshape(-1, 1, 1, 1)
                # t : [B, 1, 1, 1]

                predict_noise = (1 + w) * self.g(noise_data, t, c) - w * self.g(noise_data, t)
                mu = 1 / torch.sqrt(1 - self.noise_schedule._betas[t]) * (noise_data - (self.noise_schedule._betas[t] / (1 - self.noise_schedule._alphas[t])) * predict_noise)
                # mu : [B, 1, 32, 32]

                if step == 0:
                    # if t == 0, no add noise
                    break

                epsilon = torch.randn(noise_data.shape)
                new_data = mu + torch.sqrt(self.noise_schedule._betas[t]) * epsilon
                
                history_with_origin.append(torch.norm(origin_data - noise_data))
                history_with_prev.append(torch.norm(new_data - noise_data))
                noise_data = new_data
                
        torch.save(torch.tensor(history_with_origin), "DDPM_origin.pt")
        torch.save(torch.tensor(history_with_prev), "DDPM_prev.pt")

        return noise_data

    def DDIM_sampling(
        self,
        noise_data,
        time_step,
        c = None,
        w = 0,
        sampling_time_step = 10,
        custom_sampling_steps = None
    ):
        # noise_data : [B, 1, 32, 32]
        # c : [B]
        # time_step : INT

        batch_size = noise_data.shape[0]
        tau = list(range(0, time_step, time_step // sampling_time_step))
        if custom_sampling_steps is not None:
            tau = custom_sampling_steps

        S = len(tau)

        origin_data = noise_data.clone()
        history_with_origin = []
        history_with_prev = []

        # batch_size : B
        with torch.no_grad():

            # step : [T - 1, T - 2, .. 2, 1, 0]
            for i in range(S - 1, -1, -1):
            
                t = torch.full((batch_size, ), tau[i])
                t = t.reshape(-1, 1, 1, 1)
                alpha_t = self.noise_schedule._alphas[t]

                alpha_t_1 = torch.full((batch_size, 1, 1, 1,), 1)
                if i - 1 >= 0:
                    t_1 = torch.full((batch_size, ), tau[i - 1])
                    t_1 = t_1.reshape(-1, 1, 1, 1)
                    alpha_t_1 = self.noise_schedule._alphas[t_1]

                predict_noise = (1 + w) * self.g(noise_data, t, c) - w * self.g(noise_data, t)
                new_data = torch.sqrt(alpha_t_1) * ((noise_data - torch.sqrt(1 - alpha_t) * predict_noise) / torch.sqrt(alpha_t)) + torch.sqrt(1 - alpha_t_1) * predict_noise
                
                history_with_origin.append(torch.norm(origin_data - noise_data))
                history_with_prev.append(torch.norm(new_data - noise_data))
                noise_data = new_data

        torch.save(torch.tensor(history_with_origin), "diff_norm_origin.pt")
        torch.save(torch.tensor(history_with_prev), "diff_norm_prev.pt")
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