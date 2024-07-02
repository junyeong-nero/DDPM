from UNet import UNet 
import Encoder
import Utils

import torch, torchvision
from torch.utils.data import DataLoader

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

class DDPM:
    
    def __init__(self, n_timesteps, train_set) -> None:
        
        self.n_timesteps = n_timesteps
        
        # alpha, betas
        self.noise_schedule = NoiseSchedule(n_timesteps=n_timesteps)
        
        # forward encoder
        self.encoder = Encoder.ForwardEncoder(noise_schedule=self.noise_schedule)
        
        # UNet for predicting total noise
        self.g = UNet(in_channels=1, out_channels=1, n_steps=n_timesteps)
        
        # optimizer
        self.lossFunction = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.g.parameters(), lr=0.0001)

        # datasets
        self.training_loader = DataLoader(train_set, batch_size=8, shuffle=True)
        
        
        
    def save(self, path='./model.pt'):
        torch.save(self.g.state_dict(), path)
    
    def load(self, path='./model.pt'):
        self.g.load_state_dict(torch.load(path))
        self.g.eval()
        
    def train_one_epoch(self):
        running_loss = 0
        last_loss = 0

        for i, data in enumerate(self.training_loader):
            
            print(data['image'])
            
            # inputs = [bs, 1, 28, 28]
            inputs = torch.FloatTensor(data['image'])
            inputs = inputs.unsqueeze(1)
            print(inputs.shape)
            
            batch_size = inputs.shape[0]
            
            # sampled timestep
            t = torch.randint(0, self.n_timesteps, (batch_size, ))
            
            # outputs = [bs, 1, 28, 28]
            self.optimizer.zero_grad()
            outputs = self.g(inputs, t) 
            
            noised_image, epsilon = self.encoder.noise(inputs, t)
            # Utils.print_image(noised_image[0][0])

            # Compute the loss and its gradients
            loss = self.lossFunction(outputs, epsilon)
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            
            print(i, loss.item())
            
            if i % 1000 == 999:
                last_loss = running_loss / 1000 # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                running_loss = 0

        return last_loss

    def train(self):
        epoch_number = 0
        EPOCHS = 5
        
        best_vloss = 1_000_000.

        for epoch in range(EPOCHS):
            print('EPOCH {}:'.format(epoch_number + 1))

            # Make sure gradient tracking is on, and do a pass over the data
            self.g.train(True)
            avg_loss = self.train_one_epoch()
            
            if avg_loss < best_vloss:
                best_vloss = avg_loss
                model_path = 'model_'.format(epoch_number)
                torch.save(self.g.state_dict(), model_path)