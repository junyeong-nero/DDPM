import torch 
from datasets import load_dataset

from myDDPM import Utils
from myDDPM.DDPM import DDPM
from myDDPM.ForwardEncoder import ForwardEncoder
from myDDPM.NoiseSchedule import NoiseSchedule


def TEST_encoder():
    TIME_STEPS = 1000
    noise_schedule = NoiseSchedule(n_timesteps=TIME_STEPS)
    encoder = ForwardEncoder(noise_schedule=noise_schedule)
    image = torch.load('sample_image.pt')
    # print_image(image)
    for i in range(0, TIME_STEPS, 100):
        noised_image, epsilon = encoder.noise(image, i)
        print(i)
        Utils.print_image(noised_image[0][0])
        # print_image(epsilon)
        
def train():
    
    dataset = load_dataset("junyeong-nero/mnist_32by32").with_format("torch")
    train, test = dataset['train'], dataset['test']
        
    D = DDPM(n_timesteps=1000, train_set=train)
    D.train()
    
    
if __name__ == '__main__':
    # T_noise()
    train()
    