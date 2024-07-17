import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

def sinusoidal_embedding(n, d):
    # Returns the standard positional embedding
    embedding = torch.tensor([[i / 10_000 ** (2 * j / d) for j in range(d)] for i in range(n)])
    sin_mask = torch.arange(0, n, 2)

    embedding[sin_mask] = torch.sin(embedding[sin_mask])
    embedding[1 - sin_mask] = torch.cos(embedding[sin_mask])

    return embedding

class MyConv(nn.Module):
    def __init__(self, shape, in_c, out_c, kernel_size=3, stride=1, padding=1, activation=None, normalize=True):
        super(MyConv, self).__init__()
        self.ln = nn.LayerNorm(shape)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
        self.activation = nn.SiLU() if activation is None else activation
        self.normalize = normalize

    def forward(self, x):
        out = self.ln(x) if self.normalize else x
        out = self.conv1(out)
        out = self.activation(out)
        return out
    

class MyTimeEmbedding(nn.Module):
    
    def __init__(self, dim_in, dim_out):
        super(MyTimeEmbedding, self).__init__()
        self.linear1 = nn.Linear(dim_in, dim_out)
        self.activation = nn.SiLU()
        self.linear2 = nn.Linear(dim_out, dim_out)

    def forward(self, x, batch_size):
        out = self.linear1(x)
        out = self.activation(out)
        out = self.linear2(out)
        return out.reshape(batch_size, -1, 1, 1)


# [B, in_c, 32, 32] -> [B, out_c, 32, 32]
def MyTinyBlock(size, in_c, out_c):
    return nn.Sequential(MyConv((in_c, size, size), in_c, out_c),
                         MyConv((out_c, size, size), out_c, out_c),
                         MyConv((out_c, size, size), out_c, out_c))

# [B, in_c, 32, 32] -> [B, in_c // 4, 32, 32]
def MyTinyUp(size, in_c):
    return nn.Sequential(MyConv((in_c, size, size), in_c, in_c//2),
                         MyConv((in_c//2, size, size), in_c//2, in_c//4),
                         MyConv((in_c//4, size, size), in_c//4, in_c//4))

class UNet(nn.Module):
  # Here is a network with 3 down and 3 up with the tiny block
    def __init__(self, in_channels=1, out_channels=1, size=32, 
                 n_steps=1000,
                 time_emb_dim=256,
                 n_classes=10,
                 class_emb_dim=256,
                 channel_scale=64):
        super(UNet, self).__init__()

        ### Sinusoidal embedding

        # [1] -> [time_emb_dim]
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)
        
        # [1] -> [class_emb_dim]
        self.class_embed = nn.Embedding(n_classes, class_emb_dim)
        self.class_embed.weight.data = sinusoidal_embedding(n_classes, class_emb_dim)
        self.class_embed.requires_grad_(False)

        ### Downsampling
        self.te_enc1 = MyTimeEmbedding(time_emb_dim, 1)
        self.te_enc2 = MyTimeEmbedding(time_emb_dim, channel_scale)
        self.te_enc3 = MyTimeEmbedding(time_emb_dim, channel_scale * 2)

        self.enc1 = MyTinyBlock(size, in_channels, channel_scale)
        self.enc2 = MyTinyBlock(size//2, channel_scale, channel_scale * 2)
        self.end3 = MyTinyBlock(size//4, channel_scale * 2, channel_scale * 4)

        self.down1 = nn.Conv2d(channel_scale, channel_scale, kernel_size=4, stride=2, padding=1)
        self.down2 = nn.Conv2d(channel_scale * 2, channel_scale * 2, kernel_size=4, stride=2, padding=1)
        self.down3 = nn.Conv2d(channel_scale * 4, channel_scale * 4, kernel_size=4, stride=2, padding=1)

        ### Bottleneck
        self.te_bottleneck = MyTimeEmbedding(time_emb_dim, channel_scale * 4)
        self.bottleneck = nn.Sequential(
            MyConv((channel_scale * 4, size//8, size//8), channel_scale * 4, channel_scale * 2),
            MyConv((channel_scale * 2, size//8, size//8), channel_scale * 2, channel_scale * 2),
            MyConv((channel_scale * 2, size//8, size//8), channel_scale * 2, channel_scale * 4)
        )

        ### Upsampling

        self.dec1 = nn.ConvTranspose2d(channel_scale * 4, channel_scale * 4, kernel_size=4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(channel_scale * 2, channel_scale * 2, kernel_size=4, stride=2, padding=1)
        self.dec3 = nn.ConvTranspose2d(channel_scale, channel_scale, kernel_size=4, stride=2, padding=1)

        self.te_dec1 = MyTimeEmbedding(time_emb_dim, channel_scale * 8)
        self.te_dec2 = MyTimeEmbedding(time_emb_dim, channel_scale * 4)
        self.te_dec3 = MyTimeEmbedding(time_emb_dim, channel_scale * 2)

        self.up1 = MyTinyUp(size//4, channel_scale * 8)
        self.up2 = MyTinyUp(size//2, channel_scale * 4)
        self.up3 = MyTinyBlock(size, channel_scale * 2, channel_scale)

        self.conv_out = nn.Conv2d(channel_scale, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t, c=None):
        # x : [B, 1, 32, 32]
        # t : [B]

        t = self.time_embed(t)
        c = torch.full_like(t, 0) if c is None else self.class_embed(c)
        
        emb = t + c
            
        n = len(x)
        # t : [B, time_emb_dim]
        # n : B

        ### Downsampling
        enc1 = self.enc1(x + self.te_enc1(emb, n))
        # out1 : (B, 10, 32, 32)

        enc2 = self.enc2(self.down1(enc1) + self.te_enc2(emb, n))
        # out2 : (B, 20, 16, 16)

        enc3 = self.end3(self.down2(enc2) + self.te_enc3(emb, n))
        # out3 : (B, 40, 8, 8)


        ### Bottleneck
        bottleneck = self.bottleneck(self.down3(enc3) + self.te_bottleneck(emb, n))
        # out_mid : (B, 40, 4, 4)
        # 40 -> 20 -> 40

        ### Upsampling
        dec1 = torch.cat((enc3, self.dec1(bottleneck)), dim=1)
        # out4 : (B, 80, 8, 8)

        dec1 = self.up1(dec1 + self.te_dec1(emb, n))
        # out4 : (B, 20, 8, 8)

        dec2 = torch.cat((enc2, self.dec2(dec1)), dim=1)
        # out4 : (B, 40, 16, 16)

        dec2 = self.up2(dec2 + self.te_dec2(emb, n))
        # out5 : (B, 10, 16, 16)

        dec3 = torch.cat((enc1, self.dec3(dec2)), dim=1)
        # out5 : (B, 20, 32, 32)

        dec3 = self.up3(dec3 + self.te_dec3(emb, n))
        # out5 : (B, 10, 32, 32)

        out = self.conv_out(dec3)
        # out : (B, 1, 32, 32)

        return out

    
if __name__ == '__main__':
    unet = UNet(in_channels=3, out_channels=3, n_steps=1000)
    
    B = 64
    t = torch.randint(0, 1000, (B, )) # .type(torch.float32)
    c = torch.randint(0, 10, (B, )) # .type(torch.float32)
    x = torch.randn(B, 3, 32, 32)
    output = unet(x, t, c)

    print(output.shape)