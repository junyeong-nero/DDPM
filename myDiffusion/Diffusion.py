import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from myDiffusion.modules import UNet
from myDiffusion.encoder.ForwardEncoder import ForwardEncoder
from myDiffusion.decoder.ReverseDecoder import ReverseDecoder
from myDiffusion.scheduler.NoiseSchedule import NoiseSchedule
from myDiffusion.SamplingWeights import SamplingWeights

class MyDiffusion:

    def __init__(
        self,
        n_timesteps,
        n_classes = 10,
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
        self.device = device

        # UNet for predicting total noise
        self.g = UNet(in_channels=in_channels,
                      out_channels=out_channels,
                      n_steps=n_timesteps,
                      channel_scale=channel_scale,
                      num_channel_scale=num_channle_scale,
                      custom_channel_scale=custom_channel_scale)
        self.g = self.g.to(device)

        # alpha, betas
        self.noise_schedule = NoiseSchedule(n_timesteps=n_timesteps)

        # forward encoder
        self.encoder = ForwardEncoder(noise_schedule=self.noise_schedule)
        self.decoder = ReverseDecoder(noise_schedule=self.noise_schedule, g=self.g)

        # optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.g.parameters(), lr=learning_rate)

        # Sampling Weights
        self.sampling_weights = SamplingWeights(n_timesteps=n_timesteps)
        
        # datasets
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
        n_iter_limit = None,
        p_uncond = 0.1,
        w = 0
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

            # outputs = [B, 1, 28, 28]
            noised_image, epsilon = self.encoder.noise(inputs, t)

            outputs = None
            if torch.rand((1, )).item() < p_uncond:
                outputs = self.g(noised_image, t)
            else:
                outputs = self.g(noised_image, t, c)

            loss = self.criterion(outputs, epsilon)
            
            # [B, 1, 32, 32]
            sampling_loss = self.decoder.DDIM_sampling_step(
                noise_data=noised_image,
                t=t,
                c=c,
                w=w
            )
            
            
            self.sampling_weights.train_one_epoch(t, )

            # Adjust learning weights
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            if i == n_iter_limit:
                break

        return running_loss / len(self.training_loader)


    def train(
        self,
        n_epoch = 5,
        n_iter_limit = None,
        p_uncond = 0.1
    ):

        history = []

        for epoch in range(n_epoch):
            print('EPOCH {}:'.format(epoch + 1))

            # Make sure gradient tracking is on, and do a pass over the data
            self.g.train(True)
            avg_loss = self.train_one_epoch(n_iter_limit=n_iter_limit,
                                            p_uncond=p_uncond)
            history.append(avg_loss)
            print('# epoch {} avg_loss: {}'.format(epoch + 1, avg_loss))

            model_path = 'U{}_T{}_E{}.pt'.format(self.channel_scale,
                                                             self.n_timesteps,
                                                             epoch + 1)
            torch.save(self.g.state_dict(), model_path)
            torch.save(torch.tensor(history), 'history.pt')

        return history


    def evaluate(
        self,
        epochs = None,
        sampling_type = "DDPM",
        sampling_time_step = 10,
        w = 0
    ):
        self.decoder.g = self.g
        result = []
        for i, data in enumerate(tqdm(self.testing_loader)):

            # inputs = [B, 1, 32, 32]
            inputs, label = data # data['image']
            inputs = inputs.to(self.device)

            batch_size = inputs.shape[0]

            # timestep
            t = torch.full((batch_size, ), self.n_timesteps - 1).to(self.device)
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
                    sampling_time_step=sampling_time_step
                )

            result.append((inputs, noised_image, denoised_image))

            if i == epochs - 1:
                break

        return result