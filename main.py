import torch 
from datasets import load_dataset

from myDDPM import Utils
from myDDPM.DDPM import DDPM
from myDDPM.ForwardEncoder import ForwardEncoder
from myDDPM.ReverseDecoder import ReverseDecoder
from myDDPM.NoiseSchedule import NoiseSchedule


TIME_STEPS = 1000
BATCH_SIZE = 4096
noise_schedule = NoiseSchedule(n_timesteps=TIME_STEPS)

dataset = load_dataset("junyeong-nero/mnist_32by32").with_format("torch")
train, test = dataset['train'], dataset['test']

def TEST_encoder():
    D = DDPM(n_timesteps=TIME_STEPS, train_set=train)

    sample_image = None
    for i, data in enumerate(D.training_loader):
        sample_image = data['image']
        break

    Utils.print_image(sample_image[-1][0])
    for step in range(0, TIME_STEPS, 10):
        t = torch.full((8, ), step)
        noised_image, epsilon = D.encoder.noise(sample_image, t)
        Utils.print_image(noised_image[-1][0])


def TEST_train():
    D = DDPM(n_timesteps=TIME_STEPS, train_set=train)
    D.train(n_epoch=50) # limited training for testing
    D.save(path="./model_test.pt")


def TEST_decoder(path, n_test=1):
    D = DDPM(n_timesteps=TIME_STEPS)
    D.load(path=path)

    decoder = ReverseDecoder(noise_schedule=noise_schedule, g=D.g)

    # batch size = 1, channel = 1
    test_noise = torch.randn((n_test, 1, 32, 32))
    generated_image = decoder.denoise(test_noise, torch.tensor(TIME_STEPS))

    print(generated_image.shape)
    for i in range(n_test):
        Utils.print_image(test_noise[i][0])
        Utils.print_image(generated_image[i][0])
