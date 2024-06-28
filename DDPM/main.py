import Encoder
import numpy as np
from datasets import load_dataset
from PIL import Image

noise_schedule = Encoder.NoiseSchedule(9)
encoder = Encoder.ForwardEncoder(noise_schedule=noise_schedule)

def MNIST_test():
    data = load_dataset("ylecun/mnist")
    image = data['train'][0]['image']
    
    # summarize some details about the image
    print(image.format)
    print(image.size)
    print(image.mode)

    # image as vector
    print(np.array(image))
    
MNIST_test()