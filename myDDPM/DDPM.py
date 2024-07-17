import tqdm

import torch
from torch.utils.data import DataLoader

from myDDPM.UNet import UNet
from myDDPM.ForwardEncoder import ForwardEncoder
from myDDPM.ReverseDecoder import ReverseDecoder
from myDDPM.NoiseSchedule import NoiseSchedule

class DDPM:

    def __init__(self, n_timesteps,
                 train_set=None,
                 test_set=None,
                 in_channels=1,
                 out_channels=1,
                 channel_scale=64,
                 train_batch_size=8,
                 test_batch_size=8) -> None:

        self.n_timesteps = n_timesteps
        self.channel_scale = channel_scale

        # UNet for predicting total noise
        self.g = UNet(in_channels=in_channels,
                      out_channels=out_channels,
                      n_steps=n_timesteps,
                      channel_scale=channel_scale)
        self.g = self.g

        # alpha, betas
        self.noise_schedule = NoiseSchedule(n_timesteps=n_timesteps)

        # forward encoder
        self.encoder = ForwardEncoder(noise_schedule=self.noise_schedule)
        self.decoder = ReverseDecoder(noise_schedule=self.noise_schedule, g=self.g)

        # optimizer
        self.lossFunction = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.g.parameters(), lr=0.0001)

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


    def train_one_epoch(self, n_iter_limit=None):
        running_loss = 0

        for i, data in enumerate(tqdm(self.training_loader)):

            # inputs = [B, 1, 32, 32]
            inputs = data[0] # data['image']
            inputs = inputs
            # print(inputs.shape)

            batch_size = inputs.shape[0]

            # sampled timestep
            t = torch.randint(0, self.n_timesteps, (batch_size, ))

            # outputs = [B, 1, 28, 28]
            noised_image, epsilon = self.encoder.noise(inputs, t)
            outputs = self.g(noised_image, t)
            # Utils.print_image(noised_image[0][0])

            # Compute the loss and its gradients
            loss = self.lossFunction(outputs, epsilon)

            # Adjust learning weights
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Gather data and report
            running_loss += loss.item()

            if i == n_iter_limit:
                break

        return running_loss / len(self.training_loader)

    def train(self, n_epoch=5, n_iter_limit=None):
        best_vloss = 1_000_000
        history = []

        for epoch in range(n_epoch):
            print('EPOCH {}:'.format(epoch + 1))

            # Make sure gradient tracking is on, and do a pass over the data
            self.g.train(True)
            avg_loss = self.train_one_epoch(n_iter_limit=n_iter_limit)
            history.append(avg_loss)
            print('# epoch {} avg_loss: {}'.format(epoch + 1, avg_loss))

            if avg_loss < best_vloss:
                best_vloss = avg_loss
                model_path = 'U{}_T{}_E{}.pt'.format(self.channel_scale,
                                                     self.n_timesteps,
                                                     epoch)
                torch.save(self.g.state_dict(), model_path)

            torch.save(torch.tensor(history), 'history.pt')

        return history


    def evaluate(self, num=None, sampling_type='DDPM', n_jumps=10):
        self.decoder.g = self.g
        result = []
        for i, data in enumerate(tqdm(self.testing_loader)):

            # inputs = [B, 1, 32, 32]
            inputs = data[0] # data['image']
            inputs = inputs

            batch_size = inputs.shape[0]

            # timestep
            t = torch.full((batch_size, ), self.n_timesteps - 1)

            # outputs = [B, 1, 28, 28]
            noised_image, epsilon = self.encoder.noise(inputs, t)

            # denoised image
            denoised_image = None
            if sampling_type == 'DDPM':
                denoised_image = self.decoder.denoise(noised_image, self.n_timesteps)
            if sampling_type == 'DDIM':
                denoised_image = self.decoder.implicit_denoise(noised_image, self.n_timesteps, n_jumps=n_jumps)

            result.append((inputs, noised_image, denoised_image))

            if i == num - 1:
                break

        return result