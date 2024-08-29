import torch

from MyDiffusion.Diffusion import Diffusion
from MyDiffusion.Utils import print_seq

from torchvision.datasets import MNIST
import torchvision.transforms as transforms

def data_prepare():
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    train = MNIST(root='./data', train=True, download=True, transform=transform)
    test = MNIST(root='./data', train=False, download=True, transform=transform)
    
    return train, test


TIME_STEPS = 1000
BATCH_SIZE = 16
EPOCHS = 30
P_UNCOND = 0.1

if __name__ == "__main__":
    
    train, test = data_prepare()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Diffusion(
        n_timesteps=TIME_STEPS,
        in_channels=1,
        out_channels=1,
        custom_channel_scale=[128, 128, 256, 256, 512, 512],
        train_set=train,
        test_set=test,
        train_batch_size=BATCH_SIZE,
        test_batch_size=8,
        device=device
    )

    # model.load('/content/drive/My Drive/models/DDPM_MNIST/MNIST_T1000_E30_S.pt')
    # model.sampling_weights.load('/content/drive/My Drive/models/DDPM_MNIST/MNIST_T1000_E30_W.pt')
    
    history = model.train(
        n_epoch=EPOCHS,
        p_uncond=P_UNCOND
    )
    
    print_seq(history)
    
    