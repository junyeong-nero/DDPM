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
    

class UNetConv2D(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(UNetConv2D, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)

        return x
    

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(UNetUp, self).__init__()
        # self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
        self.conv = UNetConv2D(out_size*2, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs0, *input):
        # print(self.n_concat)
        # print(input)
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

    def __init__(self, in_channels=1, out_channels=1, n_steps=1000, time_emb_dim=100, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(UNet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        
        # time embedding is not trained
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)
        
        filters = [64, 128, 256, 512, 1024]
        # filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = UNetConv2D(self.in_channels, filters[0], self.is_batchnorm)
        self.emb1 = UNetTimeEmbedding(time_emb_dim, filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = UNetConv2D(filters[0], filters[1], self.is_batchnorm)
        self.emb2 = UNetTimeEmbedding(time_emb_dim, filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = UNetConv2D(filters[1], filters[2], self.is_batchnorm)
        self.emb3 = UNetTimeEmbedding(time_emb_dim, filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = UNetConv2D(filters[2], filters[3], self.is_batchnorm)
        self.emb4 = UNetTimeEmbedding(time_emb_dim, filters[3])
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = UNetConv2D(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = UNetUp(filters[4], filters[3], self.is_deconv)
        self.up_emb4 = UNetTimeEmbedding(time_emb_dim, filters[3])
        
        self.up_concat3 = UNetUp(filters[3], filters[2], self.is_deconv)
        self.up_emb3 = UNetTimeEmbedding(time_emb_dim, filters[2])
        
        self.up_concat2 = UNetUp(filters[2], filters[1], self.is_deconv)
        self.up_emb2 = UNetTimeEmbedding(time_emb_dim, filters[1])
        
        self.up_concat1 = UNetUp(filters[1], filters[0], self.is_deconv)
        self.up_emb1 = UNetTimeEmbedding(time_emb_dim, filters[0])
        
        # output
        self.outconv1 = nn.Conv2d(filters[0], self.out_channels, 3, padding=1)

    def forward(self, inputs, t):
        B = inputs.shape[0]
        t = self.time_embed(t)
        
        conv1 = self.conv1(inputs)  # 16*512*1024
        maxpool1 = self.maxpool1(conv1 + self.emb1(t))  # 16*256*512

        conv2 = self.conv2(maxpool1)  # 32*256*512
        maxpool2 = self.maxpool2(conv2 + self.emb2(t))  # 32*128*256

        conv3 = self.conv3(maxpool2)  # 64*128*256
        maxpool3 = self.maxpool3(conv3 + self.emb3(t))  # 64*64*128

        conv4 = self.conv4(maxpool3)  # 128*64*128
        maxpool4 = self.maxpool4(conv4 + self.emb4(t))  # 128*32*64

        center = self.center(maxpool4)  # 256*32*64

        up4 = self.up_concat4(center, conv4) + self.up_emb4(t)  # 128*64*128
        up3 = self.up_concat3(up4, conv3) + self.up_emb3(t) # 64*128*256
        up2 = self.up_concat2(up3, conv2) + self.up_emb2(t)  # 32*256*512
        up1 = self.up_concat1(up2, conv1) + self.up_emb1(t)  # 16*512*1024

        d1 = self.outconv1(up1)  # 256

        return F.sigmoid(d1)


if __name__ == '__main__':
    unet = UNet(in_channels=3, out_channels=3, n_steps=1000)
    
    B = 64
    t = torch.randint(0, 1000, (B, )) # .type(torch.float32)
    x = torch.randn(B, 3, 32, 32)
    output = unet(x, t)

    print(output.shape)
