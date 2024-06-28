import Encoder
import numpy as np
from datasets import load_dataset
from PIL import Image
from matplotlib import pyplot as plt

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
    array = np.array(image)
    # print(np.array(image))
    
    plt.figure(figsize=(5,5))
    plt.imshow(array)
    plt.show()
    
def noise_test():
    
    
MNIST_test()