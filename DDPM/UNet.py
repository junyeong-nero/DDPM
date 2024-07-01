import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # Contracting path (Downsampling)
        self.enc1 = self.double_conv(in_channels, 64)
        self.enc2 = self.double_conv(64, 128)
        self.enc3 = self.double_conv(128, 256)
        self.enc4 = self.double_conv(256, 512)

        # Bottleneck
        self.bottleneck = self.double_conv(512, 1024)

        # Expanding path (Upsampling)
        self.upconv4 = self.up_conv(1024, 512)
        self.dec4 = self.double_conv(1024, 512)
        self.upconv3 = self.up_conv(512, 256)
        self.dec3 = self.double_conv(512, 256)
        self.upconv2 = self.up_conv(256, 128)
        self.dec2 = self.double_conv(256, 128)
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
        
        # TODO : add timestep t into UNet.
        
        # Contracting path
        enc1 = self.enc1(x)  # (B, 64, 28, 28)
        enc2 = self.enc2(F.max_pool2d(enc1, kernel_size=2))  # (B, 128, 14, 14)
        enc3 = self.enc3(F.max_pool2d(enc2, kernel_size=2))  # (B, 256, 7, 7)
        enc4 = self.enc4(F.max_pool2d(enc3, kernel_size=2))  # (B, 512, 3, 3)

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, kernel_size=2))  # (B, 1024, 1, 1)

        # Expanding path
        dec4 = self.upconv4(bottleneck)  # (B, 512, 2, 2)
        dec4 = torch.cat((dec4, self.center_crop(enc4, dec4.size(2), dec4.size(3))), dim=1)
        dec4 = self.dec4(dec4)  # (B, 512, 2, 2)

        dec3 = self.upconv3(dec4)  # (B, 256, 4, 4)
        dec3 = torch.cat((dec3, self.center_crop(enc3, dec3.size(2), dec3.size(3))), dim=1)
        dec3 = self.dec3(dec3)  # (B, 256, 4, 4)

        dec2 = self.upconv2(dec3)  # (B, 128, 8, 8)
        dec2 = torch.cat((dec2, self.center_crop(enc2, dec2.size(2), dec2.size(3))), dim=1)
        dec2 = self.dec2(dec2)  # (B, 128, 8, 8)

        dec1 = self.upconv1(dec2)  # (B, 64, 16, 16)
        dec1 = torch.cat((dec1, self.center_crop(enc1, dec1.size(2), dec1.size(3))), dim=1)
        dec1 = self.dec1(dec1)  # (B, 64, 16, 16)

        # Final upsampling to get 28x28 output
        output = F.interpolate(dec1, size=(28, 28), mode='bilinear', align_corners=False)
        output = self.out_conv(output)
        return output

    def center_crop(self, tensor, target_height, target_width):
        """Center crop tensor to target size"""
        _, _, height, width = tensor.size()
        diff_height = (height - target_height) // 2
        diff_width = (width - target_width) // 2
        return tensor[:, :, diff_height:diff_height + target_height, diff_width:diff_width + target_width]


if __name__ == '__main__':
    # 모델 초기화 예시
    unet = UNet(in_channels=1, out_channels=1)

    # 임의의 입력 예시
    x = torch.randn(1, 1, 28, 28)  # Batch size = 1, 채널 수 = 1, 이미지 크기 = 28x28
    output = unet(x)

    print(output.shape)  # torch.Size([1, 1, 28, 28])
