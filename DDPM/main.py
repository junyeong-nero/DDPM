import torch 
import Encoder
import numpy as np
from datasets import load_dataset
from PIL import Image
from matplotlib import pyplot as plt


# CONTSANTS
TIME_STEPS = 1000

# Encoder and Decoder
noise_schedule = Encoder.NoiseSchedule(num_timesteps=TIME_STEPS)
encoder = Encoder.ForwardEncoder(noise_schedule=noise_schedule)

def extract_sample_image(index=0):
    data = load_dataset("ylecun/mnist")
    image = torch.from_numpy(np.array(data['train'][index]['image']))
    return image
    
def print_image(image):
    plt.figure(figsize=(5,5))
    plt.imshow(image)
    plt.show()

def T_noise(image):
    # print_image(image)
    for i in range(0, TIME_STEPS, 10):
        noised_image = encoder.noise(image, i)
        print(i)
        print_image(noised_image)

if __name__ == '__main__':
    image = torch.load('sample_image.pt')
    T_noise(image)