import torch 
from datasets import load_dataset

from myDDPM import Utils
from myDDPM.DDPM import DDPM
from myDDPM.ForwardEncoder import ForwardEncoder
from myDDPM.ReverseDecoder import ReverseDecoder
from myDDPM.NoiseSchedule import NoiseSchedule


TIME_STEPS = 1000
noise_schedule = NoiseSchedule(n_timesteps=TIME_STEPS)

def TEST_encoder():
    encoder = ForwardEncoder(noise_schedule=noise_schedule)
    
    # batch size = 1, channel = 1
    image = torch.load('sample_image.pt').unsqueeze(0).unsqueeze(0)
    print(image.shape)
    
    for i in range(0, TIME_STEPS, 100):
        noised_image, epsilon = encoder.noise(image, i)
        print(i)
        Utils.print_image(noised_image[0][0]) 
        # print_image(epsilon)
    
        
def TEST_train():
    dataset = load_dataset("junyeong-nero/mnist_32by32").with_format("torch")
    train, test = dataset['train'], dataset['test']
    
    D = DDPM(n_timesteps=TIME_STEPS, train_set=train)
    D.train(n_epoch=1, n_iter_limit=500) # limited training for testing
    D.save(path="./model_test.pt")

        
def TEST_decoder():
    D = DDPM(n_timesteps=1000)
    D.load(path='./model_test.pt')

    decoder = ReverseDecoder(noise_schedule=noise_schedule, g=D.g)
    
    # batch size = 1, channel = 1
    test_noise = torch.randn((1, 1, 32, 32))
    generated_image = decoder.denoise(test_noise, torch.tensor(TIME_STEPS))
    print(generated_image.shape)
    
    Utils.print_image(test_noise[0][0])
    Utils.print_image(generated_image[0][0])
    
    
if __name__ == '__main__':
    TEST_encoder()
    # TEST_train()
    # TEST_decoder()
    