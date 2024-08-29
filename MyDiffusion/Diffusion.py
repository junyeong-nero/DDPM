import tqdm
import random
import torch
from torch.utils.data import DataLoader

from modules.UNet import UNet
from ForwardEncoder import ForwardEncoder
from ReverseDecoder import ReverseDecoder
from NoiseSchedule import NoiseSchedule
from SamplingWeights import SamplingWeights

class Diffusion:

    def __init__(
        self,
        n_timesteps,
        train_set = None,
        test_set = None,
        in_channels = 1,
        out_channels = 1,
        channel_scale = 64,
        num_channle_scale = 5,
        train_batch_size = 8,
        test_batch_size = 8,
        custom_channel_scale = None,
        learning_rate = 0.0001,
        device = None
    ):

        self.n_timesteps = n_timesteps
        self.channel_scale = channel_scale
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.device = device

        # UNet 
        self.g = UNet(in_channels=in_channels,
                      out_channels=out_channels,
                      n_steps=n_timesteps,
                      channel_scale=channel_scale,
                      num_channel_scale=num_channle_scale,
                      custom_channel_scale=custom_channel_scale)

        self.g = self.g.to(device)

        # Sampling Weights
        self.sampling_weights = SamplingWeights(
            n_timesteps = n_timesteps
        )

        # Noise Scheduler
        self.noise_schedule = NoiseSchedule(
            n_timesteps=n_timesteps,
            device=device
            # init_type="exponential",
        )

        # Forward Encoder
        self.encoder = ForwardEncoder(
            noise_schedule=self.noise_schedule,
            device=device
        )

        # Reverse Decoder
        self.decoder = ReverseDecoder(
            noise_schedule=self.noise_schedule,
            g=self.g,
            device=device
        )

        # Optimizer
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.g.parameters(), lr=learning_rate)

        # DataLoader
        if train_set:
            self.training_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True)
        if test_set:
            self.testing_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=True)


    def save(self, path='./model.pt'):
        torch.save(self.g.state_dict(), path)


    def load(self, path='./model.pt'):
        self.g.load_state_dict(torch.load(path))
        self.g.eval()


    def train_one_epoch(
        self,
        p_uncond = 0.1,
        w = 1
    ):

        running_loss = 0

        for i, data in enumerate(tqdm(self.training_loader)):

            # inputs = [B, 1, 32, 32]
            inputs, label = data
            inputs = inputs.to(self.device)
            # print(inputs.shape)

            batch_size = inputs.shape[0]

            # sampled timestep and conditional variables
            t = torch.randint(0, self.n_timesteps, (batch_size, )).to(self.device)
            c = label.to(self.device)

            # Forward Encoder (image -> noise)
            noised_image, epsilon = self.encoder.noise(inputs, t)

            # Classifier Free Guidance
            if random.random() < p_uncond:
                c = None
            outputs = self.g(noised_image, t, c)

            loss = self.criterion(outputs, epsilon)


            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

            running_loss += loss.item()
            
            ############ Training Sampling Weights ###########
            
            # outputs = outputs.detach().clone()
            # sampling_loss = self.decoder.DDIM_sampling_step(
            #     noise_data=noised_image,
            #     predict_noise=outputs,
            #     t=t,
            #     c=c,
            #     w=w
            # )

            # sampling_loss = torch.linalg.matrix_norm(sampling_loss - noised_image)
            # self.sampling_weights.train_one_epoch(t, sampling_loss)


        return running_loss / len(self.training_loader)


    def train(
        self,
        n_epoch = 5,
        n_iter_limit = None,
        p_uncond = 0.1
    ):

        history = []

        for epoch in range(n_epoch):
            self.g.train(True)
            print('EPOCH {}:'.format(epoch + 1))
            avg_loss = self.train_one_epoch(p_uncond=p_uncond)
            history.append(avg_loss)
            print('# epoch {} avg_loss: {}'.format(epoch + 1, avg_loss))


            model_path = 'U{}_T{}_E{}.pt'.format(
                self.channel_scale,
                self.n_timesteps,
                epoch + 1
            )

            torch.save(self.g.state_dict(), model_path)
            torch.save(torch.tensor(history), 'history.pt')

        return history


    def evaluate(
        self,
        epochs = None,
        sampling_type = "DDPM",
        sampling_timestep = 10,
        custom_sampling_steps = None,
        w = 0,
        handler = None
    ):
        self.decoder.g = self.g
        result = []
        for i, data in enumerate(tqdm(self.testing_loader)):

            # inputs = [B, 1, 32, 32]
            inputs, label = data 
            inputs = inputs.to(self.device)

            B = inputs.shape[0]

            # timestep
            t = torch.full((B, ), self.n_timesteps - 1).to(self.device)
            c = label.to(self.device)

            # outputs = [B, 1, 28, 28]
            noised_image, epsilon = self.encoder.noise(inputs, t)

            # denoised image
            denoised_image = None
            if sampling_type == "DDPM":
                denoised_image = self.decoder.DDPM_sampling(
                    noised_image,
                    self.n_timesteps,
                    c=c,
                    w=w
                )
            if sampling_type == "DDIM":
                denoised_image = self.decoder.DDIM_sampling(
                    noised_image,
                    self.n_timesteps,
                    c=c,
                    w=w,
                    sampling_steps=sampling_timestep,
                    custom_sampling_steps=custom_sampling_steps
                )

            result.append((inputs, noised_image, denoised_image))
            if handler is not None:
                handler(inputs, noised_image, denoised_image)

            if epochs is not None and i == epochs - 1:
                break

        return result

    def evaluate_with_noise(
        self,
        epochs = 10,
        sampling_type = "DDPM",
        sampling_timestep = 10,
        custom_sampling_steps = None,
        w = 0,
        handler = None
    ):
        self.decoder.g = self.g
        B = self.test_batch_size
        
        result = []
        
        for _ in range(epochs):

            # inputs = [B, 1, 32, 32]
            inputs = torch.randn((B, 1, 32, 32)).to(self.device)

            # Timestep and Conditional variables
            t = torch.full((B, ), self.n_timesteps - 1).to(self.device)
            c = torch.randint(0, 10, (B, )).to(self.device)
            noised_image = inputs

            # Denoised Images
            denoised_image = None
            if sampling_type == "DDPM":
                denoised_image = self.decoder.DDPM_sampling(
                    noised_image,
                    self.n_timesteps,
                    c=c,
                    w=w
                )
            if sampling_type == "DDIM":
                denoised_image = self.decoder.DDIM_sampling(
                    noised_image,
                    self.n_timesteps,
                    c=c,
                    w=w,
                    sampling_steps=sampling_timestep,
                    custom_sampling_steps=custom_sampling_steps
                )

            if handler is not None:
                handler(inputs, noised_image, denoised_image)
            result.append((inputs, noised_image, denoised_image))

        return result
    
    
if __name__ == '__main__':
    model = Diffusion(100)
    
    