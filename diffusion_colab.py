
# !pip install --upgrade pip
# !pip install -U -q torch torchvision scipy tdqm matplotlib scipy transformers

# %% [markdown]
# # Import Modules

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image

from scipy.linalg import sqrtm
import numpy as np
import os

from tqdm import tqdm
from matplotlib import pyplot as plt

# %%
def image_normalize(image):
    image = image.cpu()
    n_channels = image.shape[0]
    for channel in range(n_channels):
        max_value = torch.max(image[channel])
        min_value = torch.min(image[channel])
        image[channel] = (image[channel] - min_value) / (max_value - min_value)

    image = image.permute(1, 2, 0)

    return image

def print_image(image):
    image = image_normalize(image)
    plt.figure(figsize=(5,5))
    plt.imshow(image)
    plt.show()

def print_2images(image1, image2):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image_normalize(image1))
    axes[0].set_title('Image 1')

    axes[1].imshow(image_normalize(image2))
    axes[1].set_title('Image 2')

    plt.tight_layout()
    plt.show()

def print_digits(result):
    fig, axes = plt.subplots(1, 10, figsize=(10, 5))

    B = result.shape[0]
    for i in range(B):
        axes[i].imshow(image_normalize(result[i]))
        axes[i].set_title(i)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

def print_result(result):
    for image, noised_image, denoised_image in result:
        batch_size = image.shape[0]
        for idx in range(batch_size):
            print_2images(image[idx], denoised_image[idx])
            # print_image(image[idx])
            # print_image(noised_image[idx])
            # print_image(denoised_image[idx])


def print_loss(loss_values):
    epochs = list(range(1, len(loss_values) + 1))
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss_values, 'b-o', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Epoch vs Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# # NoiseSchedule
# - betas, alphas

# %%
class NoiseSchedule:

    def __init__(self, n_timesteps, beta_start=0.0001, beta_end=0.02, device=device, init_type='linear') -> None:
        self._size = n_timesteps
        if init_type == 'linear':
            self._betas = torch.linspace(beta_start, beta_end, n_timesteps).to(device)
        if init_type == 'exponential':
            self._betas = torch.from_numpy(np.geomspace(beta_start, beta_end, n_timesteps)).to(device)
        self._alphas = self._calculate_alphas()

        # print(self._betas)
        # print(self._alphas)

    def _calculate_alphas(self):
        self._alphas = torch.cumprod(1 - self._betas, axis=0)
        return self._alphas

    def get_beta(self, index):
        if index >= self._size:
            raise IndexError("[get] out of index :", index, " / size :", self._size)
        return self._betas[index]

    def get_alpha(self, index):
        if index >= self._size:
            raise IndexError("[get] out of index :", index, " / size :", self._size)
        return self._alphas[index]

# %% [markdown]
# # ForwardEncoder

# %%
class ForwardEncoder:

    def __init__(self, noise_schedule) -> None:
        self.noise_schedule = noise_schedule

    def noise(self, data, time_step):
        # time_step : [B]
        # data : [B, 1, 32, 32]

        alpha = self.noise_schedule._alphas[time_step]
        alpha = alpha.reshape(-1, 1, 1, 1)
        # alpha : [B, 1, 1, 1]

        epsilon = torch.randn(data.shape).to(device)
        # torch.randn ~ N(0, 1)

        return torch.sqrt(alpha) * data + torch.sqrt(1 - alpha) * epsilon, epsilon

# %% [markdown]
# # ReverseDecoder

# %%
import torch

class ReverseDecoder:

    def __init__(self, noise_schedule, g) -> None:
        self.noise_schedule = noise_schedule
        self.g = g

    def denoise(self, noise_data, time_step, c=None, w=0):
        # noise_data : [B, 1, 32, 32]
        # c : [B]
        # time_step : INT

        batch_size = noise_data.shape[0]
        # batch_size : B

        with torch.no_grad():

            # step : [T - 1, T - 2, .. 2, 1, 0]
            for step in range(time_step - 1, -1, -1):

                t = torch.full((batch_size, ), step).to(device)
                t = t.reshape(-1, 1, 1, 1)
                # t : [B, 1, 1, 1]

                predict_noise = (1 + w) * self.g(noise_data, t, c) - w * self.g(noise_data, t)
                mu = 1 / torch.sqrt(1 - self.noise_schedule._betas[t]) * (noise_data - (self.noise_schedule._betas[t] / (1 - self.noise_schedule._alphas[t])) * predict_noise)
                # mu : [B, 1, 32, 32]

                if step == 0:
                    # if t == 0, no add noise
                    break

                epsilon = torch.randn(noise_data.shape).to(device)
                # epsilon : [B, 1, 32, 32]

                noise_data = mu + torch.sqrt(self.noise_schedule._betas[t]) * epsilon
                # noise_data : [B, 1, 32, 32]

        return noise_data

    def implicit_denoise(self, noise_data, time_step, c=None, w=0, sampling_time_step=10):
        # noise_data : [B, 1, 32, 32]
        # c : [B]
        # time_step : INT

        batch_size = noise_data.shape[0]
        tau = list(range(0, time_step, time_step // sampling_time_step))
        S = len(tau)
        # print(tau)

        # batch_size : B
        with torch.no_grad():

            # step : [T - 1, T - 2, .. 2, 1, 0]
            for i in range(S - 1, -1, -1):

                t = torch.full((batch_size, ), tau[i]).to(device)
                t = t.reshape(-1, 1, 1, 1)
                alpha_t = self.noise_schedule._alphas[t]

                alpha_t_1 = torch.full((batch_size, 1, 1, 1,), 1).to(device)
                if i - 1 >= 0:
                    t_1 = torch.full((batch_size, ), tau[i - 1]).to(device)
                    t_1 = t_1.reshape(-1, 1, 1, 1)
                    alpha_t_1 = self.noise_schedule._alphas[t_1]

                predict_noise = (1 + w) * self.g(noise_data, t, c) - w * self.g(noise_data, t)
                first = torch.sqrt(alpha_t_1) * ((noise_data - torch.sqrt(1 - alpha_t) * predict_noise) / torch.sqrt(alpha_t))
                second = torch.sqrt(1 - alpha_t_1) * predict_noise

                noise_data = first + second

        return noise_data

# %% [markdown]
# # UNet

# %% [markdown]
# ### Backbones

# %%
class PositionalEmbedding(nn.Module):

    def __init__(self, num_steps, time_emb_dim) -> None:
        super(PositionalEmbedding, self).__init__()

        self.time_embed = nn.Embedding(num_steps, time_emb_dim)
        self.time_embed.weight.data = self.sinusoidal_embedding(num_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

    def sinusoidal_embedding(self, n, d):
        # Returns the standard positional embedding
        embedding = torch.tensor([[i / 10_000 ** (2 * j / d) for j in range(d)] for i in range(n)])
        sin_mask = torch.arange(0, n, 2)
        embedding[sin_mask] = torch.sin(embedding[sin_mask])
        embedding[1 - sin_mask] = torch.cos(embedding[sin_mask])

        return embedding

    def forward(self, input):
        return self.time_embed(input)


class WideResNetBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        is_batchnorm = True,
        n = 3,
        kernel_size = 3,
        stride = 1,
        padding = 1,
        num_groups = 32
    ):
        super(WideResNetBlock, self).__init__()
        self.n = n
        self.ks = kernel_size
        self.stride = stride
        self.padding = padding

        self.shortcut = nn.Sequential()
        if kernel_size != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
                # nn.BatchNorm2d(out_size)
            )

        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                    #  nn.BatchNorm2d(out_size),
                    nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
                    nn.SiLU(inplace=True)
                )
                setattr(self, 'conv%d' % i, conv)
                in_channels = out_channels

        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                    nn.SiLU(inplace=True)
                )
                setattr(self, 'conv%d' % i, conv)
                in_channels = out_channels


    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)
        x += self.shortcut(inputs)
        return x


class MultiHeadAttentionBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        is_batchnorm = True,
        num_heads = 2,
        num_groups = 32,
    ):
        super(MultiHeadAttentionBlock, self).__init__()

        self.is_batchnorm = is_batchnorm
        # For each of heads use d_k = d_v = d_model / num_heads
        self.num_heads = num_heads
        self.d_model = out_channels
        self.d_keys = out_channels // num_heads
        self.d_values = out_channels // num_heads

        self.W_Q = nn.Linear(in_channels, out_channels)
        self.W_K = nn.Linear(in_channels, out_channels)
        self.W_V = nn.Linear(in_channels, out_channels)

        self.final_projection = nn.Linear(out_channels, out_channels)
        self.norm = nn.GroupNorm(num_channels=out_channels, num_groups=num_groups)

    def split_features_for_heads(self, tensor):
        batch, hw, emb_dim = tensor.shape
        channels_per_head = emb_dim // self.num_heads
        heads_splitted_tensor = torch.split(tensor, split_size_or_sections=channels_per_head, dim=-1)
        heads_splitted_tensor = torch.stack(heads_splitted_tensor, 1)
        return heads_splitted_tensor

    def attention(self, q, k, v):

        B, C, H, W = q.shape
        q = q.view(B, C, q.shape[2] * q.shape[3]).transpose(1, 2)
        k = k.view(B, C, k.shape[2] * k.shape[3]).transpose(1, 2)
        v = v.view(B, C, v.shape[2] * v.shape[3]).transpose(1, 2)

        # [B, H * W, C_in]

        q = self.W_Q(q)
        k = self.W_K(k)
        v = self.W_V(v)
        # N = H * W
        # [B, N, C_out]
        # print(Q.shape)

        Q = self.split_features_for_heads(q)
        K = self.split_features_for_heads(k)
        V = self.split_features_for_heads(v)
        # [B, num_heads, N, C_out / num_heads]
        # print(Q.shape)

        scale = self.d_keys ** -0.5
        attention_scores = torch.softmax(torch.matmul(Q, K.transpose(-1, -2)) * scale, dim=-1)
        attention_scores = torch.matmul(attention_scores, V)
        # [B, num_heads, N, C_out / num_heads]
        # print(attention_scores.shape)

        attention_scores = attention_scores.permute(0, 2, 1, 3).contiguous()
        # [B, num_heads, N, C_out / num_heads] --> [B, N, num_heads, C_out / num_heads]

        concatenated_heads_attention_scores = attention_scores.view(B, H * W, self.d_model)
        # [B, N, num_heads, C_out / num_heads] --> [batch, N, C_out]

        linear_projection = self.final_projection(concatenated_heads_attention_scores)
        linear_projection = linear_projection.transpose(-1, -2).reshape(B, self.d_model, H, W)
        # [B, N, C_out] -> [B, C_out, N] -> [B, C_out, H, W]
        # print('linear_projection', linear_projection.shape)
        # print('linear_projection', v.shape)

        # Residual connection + norm
        out = linear_projection
        if self.is_batchnorm:
            v = v.transpose(-1, -2).reshape(B, self.d_model, H, W)
            out = self.norm(out + v)
        return out

    def forward(self, q, k, v):
        return self.attention(q, k, v)


class SelfAttentionBlock(MultiHeadAttentionBlock):
    def __init__(
        self,
        in_channels,
        out_channels,
        is_batchnorm = True,
        num_heads = 2,
        num_groups = 32,
    ):
        super().__init__(in_channels, out_channels, num_heads, num_groups)


    def forward(self, x):
        return super().forward(x, x, x)

# %% [markdown]
# ### UNet Body

# %%
class UNetDown(nn.Module):

    def __init__(
        self, 
        in_channels,
        out_channels,
        base_model = WideResNetBlock,
        is_deconv = True,
        is_batchnorm = True
    ):
        super(UNetDown, self).__init__()
        self.conv = base_model(in_channels, out_channels, is_batchnorm=is_batchnorm)

    def forward(self, input):
        return self.conv(input)
        

class UNetUp(nn.Module):
    def __init__(
        self, 
        in_channels,
        out_channels,
        base_model = WideResNetBlock,
        is_deconv = True,
        is_batchnorm = True
    ):
        super(UNetUp, self).__init__()
        self.conv = base_model(out_channels * 2, out_channels, is_batchnorm=is_batchnorm)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs0, *input):
        outputs0 = self.up(inputs0)
        for i in range(len(input)):
            outputs0 = torch.cat([outputs0, input[i]], 1)
        return self.conv(outputs0)


class UNetTimeEmbedding(nn.Module):

    def __init__(self, dim_in, dim_out) -> None:
        super(UNetTimeEmbedding, self).__init__()
        self.ln = nn.Linear(dim_in, dim_out)
        self.activation = nn.SiLU()
        self.ln2 = nn.Linear(dim_out, dim_out)


    def forward(self, inputs):
        B = inputs.shape[0]

        x = self.ln(inputs)
        x = self.activation(x)
        x = self.ln2(x)

        return x.reshape(B, -1, 1, 1)


class UNet(nn.Module):

    def __init__(
        self,
        in_channels = 1,
        out_channels = 1,
        n_steps = 1000,
        time_emb_dim = 256,
        n_classes = 10,
        class_emb_dim = 64,
        channel_scale = 64,
        feature_scale = 5,
        custom_scale = None,
        is_deconv = True,
        is_batchnorm = True
    ):
        super(UNet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_batchnorm = is_batchnorm

        # time embedding
        self.time_embed = PositionalEmbedding(n_steps, time_emb_dim)

        # conditional variable embedding
        self.class_embed = PositionalEmbedding(n_classes, class_emb_dim)
        
        if custom_scale is None:
            filters = [channel_scale * (2 ** i) for i in range(feature_scale + 1)]
        else:
            feature_scale = len(custom_scale) - 1
            filters = custom_scale
        self.feature_scale = feature_scale
        
        # Downsampling
        filters[0] = in_channels
        
        for i in range(1, feature_scale):
            base_model = SelfAttentionBlock if i == feature_scale - 1 else WideResNetBlock
            conv = UNetDown(in_channels=filters[i - 1], 
                            out_channels=filters[i], 
                            is_batchnorm=self.is_batchnorm,
                            base_model=base_model)
            temb = UNetTimeEmbedding(time_emb_dim, filters[i])
            cemb = UNetTimeEmbedding(class_emb_dim, filters[i])
            maxpool = nn.MaxPool2d(kernel_size=2)
            cross_attention = MultiHeadAttentionBlock(
                in_channels=filters[i],
                out_channels=filters[i],
                is_batchnorm=False
            )
            
            setattr(self, 'down_conv%d' % i, conv)
            setattr(self, 'down_temb%d' % i, temb)
            setattr(self, 'down_cemb%d' % i, cemb)
            setattr(self, 'down_maxpool%d' % i, maxpool)
            setattr(self, 'down_cross_attention%d' % i, cross_attention)
        
        # Bottleneck        

        self.center = UNetDown(filters[-2], filters[-1], is_batchnorm=self.is_batchnorm)
        self.temb_center = UNetTimeEmbedding(time_emb_dim, filters[-1])
        self.cemb_center = UNetTimeEmbedding(class_emb_dim, filters[-1])
        self.cross_attention_center = MultiHeadAttentionBlock(
            in_channels=filters[-1],
            out_channels=filters[-1],
            is_batchnorm=False
        )

        # upsampling
        filters[0] = out_channels
        
        for i in range(1, feature_scale):
            base_model = SelfAttentionBlock if i == feature_scale - 1 else WideResNetBlock
            conv = UNetUp(filters[i + 1], 
                          filters[i], 
                          is_deconv=self.is_deconv, 
                          is_batchnorm=self.is_batchnorm,
                          base_model=base_model)
            temb = UNetTimeEmbedding(time_emb_dim, filters[i])
            cemb = UNetTimeEmbedding(class_emb_dim, filters[i])
            cross_attention = MultiHeadAttentionBlock(
                in_channels=filters[i],
                out_channels=filters[i],
                is_batchnorm=False
            )
            
            setattr(self, 'up_conv%d' % i, conv)
            setattr(self, 'up_temb%d' % i, temb)
            setattr(self, 'up_cemb%d' % i, cemb)
            setattr(self, 'up_cross_attention%d' % i, cross_attention)

        # output
        self.outconv = nn.Conv2d(filters[1], self.out_channels, 3, padding=1)
        

    def forward(self, inputs, t, c=None):
        
        B, C, H, W = inputs.shape
        t = self.time_embed(t)
        if c is not None:
            c = self.class_embed(c)
        # inputs : [B, 1, 32, 32]
        
        x = inputs
        downsampling_result = [None]
        
        # DOWN-SAMPLING
        for i in range(1, self.feature_scale):
            
            conv = getattr(self, 'down_conv%d' % i)
            temb = getattr(self, 'down_temb%d' % i)
            cemb = getattr(self, 'down_cemb%d' % i)
            maxpool = getattr(self, 'down_maxpool%d' % i)
            CA = getattr(self, 'down_cross_attention%d' % i)
            
            x = conv(x)
            downsampling_result.append(x)
            
            if c is not None and i == self.feature_scale - 1:
                context_emb = cemb(c)
                x = CA(x, context_emb, context_emb)
            x += temb(t)
            x = maxpool(x)
            
        
        # BOTTLENECK
            
        x = self.center(x)
        if c is not None:
            context_emb = self.cemb_center(c)
            x = self.cross_attention_center(x, context_emb, context_emb)
        x += self.temb_center(t)
        
        # UP-SAMPLING

        for i in range(self.feature_scale - 1, 0, -1):
        
            conv = getattr(self, 'up_conv%d' % i)
            temb = getattr(self, 'up_temb%d' % i)
            cemb = getattr(self, 'up_cemb%d' % i)
            CA = getattr(self, 'up_cross_attention%d' % i)
            
            x = conv(x, downsampling_result[i])
        
            # if c is not None:
            #     context_emb = cemb(c)
            #     x = CA(x, context_emb, context_emb)
            x += temb(t)

        return self.outconv(x)

# %% [markdown]
# # Diffusion

# %%
class DDPM:

    def __init__(
        self,
        n_timesteps,
        train_set = None,
        test_set = None,
        in_channels = 1,
        out_channels = 1,
        channel_scale = 64,
        feature_scale = 5,
        train_batch_size = 8,
        test_batch_size = 8,
        custom_scale = None,
        learning_rate = 0.0001
    ):

        self.n_timesteps = n_timesteps
        self.channel_scale = channel_scale

        # UNet for predicting total noise
        self.g = UNet(in_channels=in_channels,
                      out_channels=out_channels,
                      n_steps=n_timesteps,
                      feature_scale=feature_scale,
                      channel_scale=channel_scale,
                      custom_scale=custom_scale)
        self.g = self.g.to(device)

        # alpha, betas
        self.noise_schedule = NoiseSchedule(n_timesteps=n_timesteps)

        # forward encoder
        self.encoder = ForwardEncoder(noise_schedule=self.noise_schedule)
        self.decoder = ReverseDecoder(noise_schedule=self.noise_schedule, g=self.g)

        # optimizer
        self.lossFunction = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.g.parameters(), lr=learning_rate)

        # datasets
        if train_set:
            self.training_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True)
        if test_set:
            self.testing_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=True)


    def save(self, path='./model.pt'):
        torch.save(self.g.state_dict(), path)


    def load(self, path='./model.pt'):
        self.g.load_state_dict(torch.load(path))
        self.g.eval()


    def train_one_epoch(
        self,
        n_iter_limit = None,
        p_uncond = 0.1
    ):

        running_loss = 0

        for i, data in enumerate(tqdm(self.training_loader)):

            # inputs = [B, 1, 32, 32]
            inputs, label = data
            inputs = inputs.to(device)
            # print(inputs.shape)

            batch_size = inputs.shape[0]

            # sampled timestep and conditional variables
            t = torch.randint(0, self.n_timesteps, (batch_size, )).to(device)
            c = label.to(device)

            # outputs = [B, 1, 28, 28]
            noised_image, epsilon = self.encoder.noise(inputs, t)

            outputs = None
            if torch.rand((1, )).item() < p_uncond:
                outputs = self.g(noised_image, t)
            else:
                outputs = self.g(noised_image, t, c)

            loss = self.lossFunction(outputs, epsilon)

            # Adjust learning weights
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            if i == n_iter_limit:
                break

        return running_loss / len(self.training_loader)


    def train(
        self,
        n_epoch=5,
        n_iter_limit=None,
        p_uncond=0.1
    ):

        history = []

        for epoch in range(n_epoch):
            print('EPOCH {}:'.format(epoch + 1))

            # Make sure gradient tracking is on, and do a pass over the data
            self.g.train(True)
            avg_loss = self.train_one_epoch(n_iter_limit=n_iter_limit,
                                            p_uncond=p_uncond)
            history.append(avg_loss)
            print('# epoch {} avg_loss: {}'.format(epoch + 1, avg_loss))

            model_path = 'U{}_T{}_E{}.pt'.format(self.channel_scale,
                                                             self.n_timesteps,
                                                             epoch + 1)
            torch.save(self.g.state_dict(), model_path)
            torch.save(torch.tensor(history), 'history.pt')

        return history


    def evaluate(
        self,
        epochs = None,
        sampling_type = 'DDPM',
        sampling_time_step = 10,
        w = 0
    ):
        self.decoder.g = self.g
        result = []
        for i, data in enumerate(tqdm(self.testing_loader)):

            # inputs = [B, 1, 32, 32]
            inputs, label = data # data['image']
            inputs = inputs.to(device)

            batch_size = inputs.shape[0]

            # timestep
            t = torch.full((batch_size, ), self.n_timesteps - 1).to(device)
            c = label.to(device)

            # outputs = [B, 1, 28, 28]
            noised_image, epsilon = self.encoder.noise(inputs, t)

            # denoised image
            denoised_image = None
            if sampling_type == 'DDPM':
                denoised_image = self.decoder.denoise(noised_image,
                                                      self.n_timesteps,
                                                      c=c,
                                                      w=w)
            if sampling_type == 'DDIM':
                denoised_image = self.decoder.implicit_denoise(
                    noised_image,
                    self.n_timesteps,
                    c=c,
                    w=w,
                    sampling_time_step=sampling_time_step
                )

            result.append((inputs, noised_image, denoised_image))

            if i == epochs - 1:
                break

        return result

# %% [markdown]
# # Train

# %%
TIME_STEPS = 1000
BATCH_SIZE = 768

noise_schedule = NoiseSchedule(n_timesteps=TIME_STEPS, init_type='exponential')

# %%
# dataset = load_dataset("junyeong-nero/mnist_32by32").with_format("torch")
# train, test = dataset['train'], dataset['test']

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train = MNIST(root='./data', train=True, download=True, transform=transform)
test = MNIST(root='./data', train=False, download=True, transform=transform)

# %%
model = DDPM(
    n_timesteps=TIME_STEPS,
    in_channels=1,
    out_channels=1,
    channel_scale=128,
    custom_scale=[128, 128, 256, 256, 512, 512, 1024],
    train_set=train,
    test_set=test,
    train_batch_size=BATCH_SIZE,
    test_batch_size=8
)

# MODEL_PATH = '/content/drive/My Drive/models/DDPM_MNIST/U256_T1000_E30_P05.pt'
# MODEL_PATH = './U256_T1000_E30.pt'
# model.load(MODEL_PATH)

# %%
print("model size : ", sum(p.numel() for p in model.g.parameters() if p.requires_grad))

# %%
history = model.train(
    n_epoch=30,
    p_uncond=0.1
)
