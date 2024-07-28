import time

import torch 
from datasets import load_dataset

from myDDPM import Utils
from myDDPM.DDPM import DDPM
from myDDPM.encoder.ForwardEncoder import ForwardEncoder
from myDDPM.decoder.ReverseDecoder import ReverseDecoder
from myDDPM.scheduler.NoiseSchedule import NoiseSchedule


TIME_STEPS = 1000
BATCH_SIZE = 4096
noise_schedule = NoiseSchedule(n_timesteps=TIME_STEPS)


def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 시작 시간
        result = func(*args, **kwargs)
        end_time = time.time()    # 종료 시간
        execution_time = end_time - start_time
        print(f"실행 시간: {execution_time:.6f} 초")
        return result
    return wrapper

dataset = load_dataset("junyeong-nero/mnist_32by32").with_format("torch")
train, test = dataset['train'], dataset['test']

def TEST_encoder():
    D = DDPM(n_timesteps=TIME_STEPS, train_set=train)

    sample_image = None
    for i, data in enumerate(D.training_loader):
        sample_image = data['image']
        break

    Utils.print_image(sample_image[-1][0])
    sample_image = sample_image
    for step in range(0, TIME_STEPS, 100):
        t = torch.full((8, ), step)
        noised_image, epsilon = D.encoder.noise(sample_image, t)
        Utils.print_image(noised_image.cpu()[-1][0])

@measure_time
def TEST_decoder(n_test=1):
    D = DDPM(n_timesteps=TIME_STEPS)
    # D.load(path=path)

    decoder = ReverseDecoder(noise_schedule=noise_schedule, g=D.g)

    # batch size = 1, channel = 1
    test_noise = torch.randn((n_test, 1, 32, 32))
    generated_image = decoder.denoise(test_noise, torch.tensor(TIME_STEPS))

    print(generated_image.shape)
    # for i in range(n_test):
    #     Utils.print_image(test_noise[i][0])
    #     Utils.print_image(generated_image[i][0])


if __name__ == '__main__':
    
    TEST_decoder()
    # model = DDPM(
    #     n_timesteps=TIME_STEPS,
    #     train_set=train,
    #     test_set=test,
    #     train_batch_size=BATCH_SIZE,
    #     test_batch_size=2
    # )
    
    # history = model.train(n_epoch=5)
    # Utils.print_loss(history)