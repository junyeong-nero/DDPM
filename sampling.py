import torch

from MyDiffusion.diffusion import Diffusion
from MyDiffusion.utils import print_digits

TIME_STEPS = 1000
BATCH_SIZE = 16
EPOCHS = 30
P_UNCOND = 0.1

MODEL_PATH = "PATH"

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Diffusion(
        n_timesteps = TIME_STEPS,
        in_channels = 1,
        out_channels = 1,
        custom_channel_scale = [128, 128, 256, 256, 512, 512],
        train_batch_size = BATCH_SIZE,
        test_batch_size = 8,
        device = device
    )
    model.load(MODEL_PATH)
    
    test_noise = torch.randn((1, 1, 32, 32)).to(device)
    test_noise = test_noise.repeat(10, 1, 1, 1)
    condition = torch.tensor(list(range(10))).to(device)

    # DDIM Sampling Example
    for steps in [10, 20, 50, 100, 200]:
        test_denoised_image = model.decoder.DDIM_sampling(
            test_noise,
            TIME_STEPS,
            c = condition,
            w = 1,
            sampling_steps=steps
        )

        print_digits(test_denoised_image)
    
    