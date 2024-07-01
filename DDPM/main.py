import torch 
import Encoder
from datasets import load_dataset
from DDPM import DDPM 
import Utils


# CONTSANTS
TIME_STEPS = 1000

# Encoder and Decoder
noise_schedule = Encoder.NoiseSchedule(num_timesteps=TIME_STEPS)
encoder = Encoder.ForwardEncoder(noise_schedule=noise_schedule)

def extract_sample_image(index=0):
    data = load_dataset("ylecun/mnist")
    data = data.with_format("torch")
    image = data['train'][index]['image']
    Utils.print_image(image)
    return image

def T_noise():
    image = torch.load('sample_image.pt')
    # print_image(image)
    for i in range(0, TIME_STEPS, 100):
        noised_image, epsilon = encoder.noise(image, i)
        print(i)
        Utils.print_image(noised_image)
        # print_image(epsilon)
        
def T_train():
    train_data = load_dataset("ylecun/mnist", split="train").with_format("torch")
    valid_data = load_dataset("ylecun/mnist", split="test").with_format("torch")
    D = DDPM(num_timesteps=1000, train_set=train_data)
    D.train()
    

if __name__ == '__main__':
    # extract_sample_image()
    # T_noise()
    T_train()