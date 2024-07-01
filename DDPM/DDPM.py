from UNet import UNet 
import Encoder
import Utils

import torch, torchvision
from torch.utils.data import DataLoader

class DDPM:
    
    def __init__(self, num_timesteps, train_set) -> None:
        
        self.num_timesteps = num_timesteps
        
        # alpha, betas
        self.noise_schedule = Encoder.NoiseSchedule(num_timesteps=num_timesteps)
        
        # forward encoder
        self.encoder = Encoder.ForwardEncoder(noise_schedule=self.noise_schedule)
        
        # DNN for predicting total noise
        self.g = UNet(in_channels=1, out_channels=1)
        
        self.optimizer = torch.optim.SGD(self.g.parameters(), lr=0.004, momentum=0.9)

        # Create data loaders for our datasets; shuffle for training, not for validation
        self.training_loader = DataLoader(train_set, batch_size=4, shuffle=True)
        
        self.lossFunction = torch.nn.MSELoss()
        
        
    def save(self, path='./model.pt'):
        torch.save(self.g.state_dict(), path)
    
    def load(self, path='./model.pt'):
        self.g.load_state_dict(torch.load(path))
        self.g.eval()
        
    def train_one_epoch(self):
        running_loss = 0
        last_loss = 0

        for i, data in enumerate(self.training_loader):
            
            inputs = data['image'].type(torch.float32)
            inputs = inputs.unsqueeze(1)
            # inputs = [bs, 1, 28, 28]
            
            self.optimizer.zero_grad()
            outputs = self.g(inputs)            
            # outputs = [bs, 1, 28, 28]
            
            noised_image, epsilon = self.encoder.noise(inputs, torch.randint(0, self.num_timesteps, [1, 1]).item())
            
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