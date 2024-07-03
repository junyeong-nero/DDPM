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


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, n_steps, time_emb_dim=100):
        super(UNet, self).__init__()
        
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        # Contracting path (Downsampling)
        self.te_enc1 = self._make_time_embedding(time_emb_dim, 1)
        self.enc1 = self.double_conv(in_channels, 64)
        
        self.te_enc2 = self._make_time_embedding(time_emb_dim, 64)
        self.enc2 = self.double_conv(64, 128)
        
        self.te_enc3 = self._make_time_embedding(time_emb_dim, 128)
        self.enc3 = self.double_conv(128, 256)
        
        self.te_enc4 = self._make_time_embedding(time_emb_dim, 256)
        self.enc4 = self.double_conv(256, 512)
        
        # Bottleneck
        
        self.te_bottleneck = self._make_time_embedding(time_emb_dim, 512)
        self.bottleneck = self.double_conv(512, 1024)

        # Expanding path (Upsampling)
        self.te_dec4 = self._make_time_embedding(time_emb_dim, 1024)
        self.upconv4 = self.up_conv(1024, 512)
        self.dec4 = self.double_conv(1024, 512)
        
        self.te_dec3 = self._make_time_embedding(time_emb_dim, 512)
        self.upconv3 = self.up_conv(512, 256)
        self.dec3 = self.double_conv(512, 256)
        
        self.te_dec2 = self._make_time_embedding(time_emb_dim, 256)
        self.upconv2 = self.up_conv(256, 128)
        self.dec2 = self.double_conv(256, 128)
        
        self.te_dec1 = self._make_time_embedding(time_emb_dim, 128)
        self.upconv1 = self.up_conv(128, 64)
        self.dec1 = self.double_conv(128, 64)

        # Output layer
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def double_conv(self, in_channels, out_channels):
        """Two consecutive convolutional layers followed by ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def up_conv(self, in_channels, out_channels):
        """Upsampling followed by convolution"""
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x, t=0):
        
        # batch size
        t = self.time_embed(t)
        n = len(x)
        
        # Contracting path
        enc1 = self.enc1(x + self.te_enc1(t).reshape(n, -1, 1, 1))  # (B, 64, 32, 32)
        enc2 = self.enc2(F.max_pool2d(enc1, kernel_size=2) + self.te_enc2(t).reshape(n, -1, 1, 1))  # (B, 128, 16, 16)
        enc3 = self.enc3(F.max_pool2d(enc2, kernel_size=2) + self.te_enc3(t).reshape(n, -1, 1, 1))  # (B, 256, 8, 8)
        enc4 = self.enc4(F.max_pool2d(enc3, kernel_size=2) + self.te_enc4(t).reshape(n, -1, 1, 1))  # (B, 512, 4, 4)
        
        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, kernel_size=2) + self.te_bottleneck(t).reshape(n, -1, 1, 1))  # (B, 1024, 2, 2)

        # Expanding path
        dec4 = self.upconv4(bottleneck)  # (B, 512, 2, 2)
        dec4 = torch.cat((dec4, enc4), dim=1) + self.te_dec4(t).reshape(n, -1, 1, 1)
        dec4 = self.dec4(dec4)  # (B, 512, 2, 2)
        
        dec3 = self.upconv3(dec4)  # (B, 512, 2, 2)
        dec3 = torch.cat((dec3, enc3), dim=1) + self.te_dec3(t).reshape(n, -1, 1, 1)
        dec3 = self.dec3(dec3)  # (B, 512, 2, 2)
        
        dec2 = self.upconv2(enc3)  # (B, 512, 2, 2)
        dec2= torch.cat((dec2, enc2), dim=1) + self.te_dec2(t).reshape(n, -1, 1, 1)
        dec2 = self.dec2(dec2)  # (B, 512, 2, 2)
        
        dec1 = self.upconv1(enc2)  # (B, 512, 2, 2)
        dec1 = torch.cat((dec1, enc1), dim=1) + self.te_dec1(t).reshape(n, -1, 1, 1)
        dec1 = self.dec1(dec1)  # (B, 512, 2, 2)

        output = F.interpolate(dec1, size=(32, 32), mode='bilinear', align_corners=False)
        output = self.out_conv(output)
        return output

    def center_crop(self, tensor, target_height, target_width):
        """Center crop tensor to target size"""
        _, _, height, width = tensor.size()
        diff_height = (height - target_height) // 2
        diff_width = (width - target_width) // 2
        return tensor[:, :, diff_height:diff_height + target_height, diff_width:diff_width + target_width]
    
    def _make_time_embedding(self, dim_in, dim_out):
        return nn.Sequential(nn.Linear(dim_in, dim_out), nn.SiLU(), nn.Linear(dim_out, dim_out))


if __name__ == '__main__':
    # 모델 초기화 예시
    unet = UNet(in_channels=1, out_channels=1, n_steps=1000)

    # 임의의 입력 예시
    x = torch.randn(1, 1, 32, 32)  # Batch size = 64, 채널 수 = 1, 이미지 크기 = 28x28
    output = unet(x, torch.tensor(2))

    print(output.shape)  # torch.Size([1, 1, 28, 28])
