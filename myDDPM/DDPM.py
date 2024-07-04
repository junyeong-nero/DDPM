import torch
from torch.utils.data import DataLoader

from myDDPM.UNet import UNet
from myDDPM.ForwardEncoder import ForwardEncoder
from myDDPM.NoiseSchedule import NoiseSchedule

class DDPM:
    
    def __init__(self, n_timesteps, train_set=None, test_set=None) -> None:
        
        self.n_timesteps = n_timesteps
        
        # alpha, betas
        self.noise_schedule = NoiseSchedule(n_timesteps=n_timesteps)
        
        # forward encoder
        self.encoder = ForwardEncoder(noise_schedule=self.noise_schedule)
        
        # UNet for predicting total noise
        self.g = UNet(in_channels=1, out_channels=1, n_steps=n_timesteps)
        
        # optimizer
        self.lossFunction = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.g.parameters(), lr=0.0001)

        # datasets
        if train_set:
            self.training_loader = DataLoader(train_set, batch_size=8, shuffle=True)
        if test_set:
            self.testing_loader = DataLoader(test_set, batch_size=8, shuffle=True)
        
    def save(self, path='./model.pt'):
        torch.save(self.g.state_dict(), path)
    
    def load(self, path='./model.pt'):
        self.g.load_state_dict(torch.load(path))
        self.g.eval()
        
    def train_one_epoch(self, n_iter_limit=None):
        running_loss = 0
        last_loss = 0

        for i, data in enumerate(self.training_loader):
            
            # print(data)
            
            # inputs = [B, 1, 28, 28]
            inputs = data['image']
            inputs = inputs.unsqueeze(1).type(torch.float32)
            # print(inputs.shape)
            
            batch_size = inputs.shape[0]
            
            # sampled timestep
            t = torch.randint(0, self.n_timesteps, (batch_size, ))
            
            # outputs = [B, 1, 28, 28]
            self.optimizer.zero_grad()
            outputs = self.g(inputs, t) 
            
            noised_image, epsilon = self.encoder.noise(inputs, t)
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
            
            if i % 100 == 0:
                last_loss = running_loss / 1000 # loss per batch
                print('  batch {} loss: {}'.format(i, last_loss))
                running_loss = 0

        return last_loss

    def train(self, n_epoch=5, n_iter_limit=None):
        best_vloss = 1_000_000

        for epoch in range(n_epoch):
            print('EPOCH {}:'.format(epoch + 1))

            # Make sure gradient tracking is on, and do a pass over the data
            self.g.train(True)
            avg_loss = self.train_one_epoch(n_iter_limit=n_iter_limit)
            
            if avg_loss < best_vloss:
                best_vloss = avg_loss
                model_path = 'model_{}.pt'.format(epoch)
                torch.save(self.g.state_dict(), model_path)