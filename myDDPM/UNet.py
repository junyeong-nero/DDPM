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


class SelfAttentionBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size=1) -> None:
        super(SelfAttentionBlock, self).__init__()
        
        self.W_Q = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)
        # [B, 1, 32, 32] -> [B, 3, 32, 32]
        
        self.W_K = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)
        
        self.W_V = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)
        
        self.activation = nn.Softmax(dim=1)
        
    def forward(self, query, key, value):
        B, C, W, H = query.shape
        d_k = torch.tensor(C)
        
        q = self.W_Q(query)
        k = self.W_K(key)
        v = self.W_V(value)
        
        # mutlicative cross attention
        out = self.activation(q @ k / torch.sqrt(d_k)) @ v
        return out
        
        

class UNetConv2D(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=3, kernel_size=3, stride=1, padding=1):
        super(UNetConv2D, self).__init__()
        self.n = n
        self.ks = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.shortcut = nn.Sequential()
        if kernel_size != 1 or in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_size)
            )
        
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size, stride, padding),
                                     nn.BatchNorm2d(out_size),
                                     nn.SiLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size, stride, padding),
                                     nn.SiLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size
                
        

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)
        x += self.shortcut(inputs)
        return x


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv=True, is_batchnorm=True):
        super(UNetUp, self).__init__()
        # self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
        self.conv = UNetConv2D(out_size*2, out_size, is_batchnorm)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
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
        is_deconv = True,
        is_batchnorm = True
    ):
        super(UNet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        # time embedding
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        # conditional variable embedding
        self.class_embed = nn.Embedding(n_classes, class_emb_dim)
        self.class_embed.weight.data = sinusoidal_embedding(n_classes, class_emb_dim)
        self.class_embed.requires_grad_(False)

        # filters = [64, 128, 256, 512, 1024]
        filters = [channel_scale * i for i in range(1, 1 + feature_scale)]
        # filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = UNetConv2D(self.in_channels, filters[0], self.is_batchnorm)
        self.temb1 = UNetTimeEmbedding(time_emb_dim, filters[0])
        self.cemb1 = UNetTimeEmbedding(class_emb_dim, filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = UNetConv2D(filters[0], filters[1], self.is_batchnorm)
        self.temb2 = UNetTimeEmbedding(time_emb_dim, filters[1])
        self.cemb2 = UNetTimeEmbedding(class_emb_dim, filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = UNetConv2D(filters[1], filters[2], self.is_batchnorm)
        self.temb3 = UNetTimeEmbedding(time_emb_dim, filters[2])
        self.cemb3 = UNetTimeEmbedding(class_emb_dim, filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = UNetConv2D(filters[2], filters[3], self.is_batchnorm)
        self.temb4 = UNetTimeEmbedding(time_emb_dim, filters[3])
        self.cemb4 = UNetTimeEmbedding(class_emb_dim, filters[3])
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = UNetConv2D(filters[3], filters[4], self.is_batchnorm)
        self.temb_center = UNetTimeEmbedding(time_emb_dim, filters[4])
        self.cemb_center = UNetTimeEmbedding(class_emb_dim, filters[4])

        # upsampling
        self.up_concat4 = UNetUp(filters[4], filters[3], self.is_deconv, self.is_batchnorm)
        self.up_temb4 = UNetTimeEmbedding(time_emb_dim, filters[3])
        self.up_cemb4 = UNetTimeEmbedding(class_emb_dim, filters[3])

        self.up_concat3 = UNetUp(filters[3], filters[2], self.is_deconv, self.is_batchnorm)
        self.up_temb3 = UNetTimeEmbedding(time_emb_dim, filters[2])
        self.up_cemb3 = UNetTimeEmbedding(class_emb_dim, filters[2])

        self.up_concat2 = UNetUp(filters[2], filters[1], self.is_deconv, self.is_batchnorm)
        self.up_temb2 = UNetTimeEmbedding(time_emb_dim, filters[1])
        self.up_cemb2 = UNetTimeEmbedding(class_emb_dim, filters[1])

        self.up_concat1 = UNetUp(filters[1], filters[0], self.is_deconv, self.is_batchnorm)
        self.up_temb1 = UNetTimeEmbedding(time_emb_dim, filters[0])
        self.up_cemb1 = UNetTimeEmbedding(class_emb_dim, filters[0])

        # output
        self.outconv1 = nn.Conv2d(filters[0], self.out_channels, 3, padding=1)


    def forward(self, inputs, t, c=None):
        t = self.time_embed(t)
        if c is not None:
            c = self.class_embed(c)
        # inputs : [B, 1, 32, 32]

        conv1 = self.conv1(inputs)  # [B, 64, 32, 32]
        emb1 = self.temb1(t) + (self.cemb1(c) if c is not None else 0)
        maxpool1 = self.maxpool1(conv1 + emb1)  # [B, 64, 16, 16]

        conv2 = self.conv2(maxpool1)  # [B, 128, 16, 16]
        emb2 = self.temb2(t) + (self.cemb2(c) if c is not None else 0)
        maxpool2 = self.maxpool2(conv2 + emb2)  # [B, 128, 8, 8]

        conv3 = self.conv3(maxpool2)  # [B, 256, 8, 8]
        emb3 = self.temb3(t) + (self.cemb3(c) if c is not None else 0)
        maxpool3 = self.maxpool3(conv3 + emb3)  # [B, 256, 4, 4]

        conv4 = self.conv4(maxpool3)  # [B, 512, 4, 4]
        emb4 = self.temb4(t) + (self.cemb4(c) if c is not None else 0)
        maxpool4 = self.maxpool4(conv4 + emb4)  # [B, 512, 2, 2]

        emb_center = self.temb_center(t) + (self.cemb_center(c) if c is not None else 0)
        center = self.center(maxpool4) + emb_center # [B, 1024, 2, 2]


        up_emb4 = self.up_temb4(t) + (self.up_cemb4(c) if c is not None else 0)
        up4 = self.up_concat4(center, conv4) + up_emb4  # [B, 512, 4, 4]

        up_emb3 = self.up_temb3(t) + (self.up_cemb3(c) if c is not None else 0)
        up3 = self.up_concat3(up4, conv3) + up_emb3 # [B, 256, 8, 8]

        up_emb2 = self.up_temb2(t) + (self.up_cemb2(c) if c is not None else 0)
        up2 = self.up_concat2(up3, conv2) + up_emb2 # [B, 128, 16, 16]

        up_emb1 = self.up_temb1(t) + (self.up_cemb1(c) if c is not None else 0)
        up1 = self.up_concat1(up2, conv1) + up_emb1 # [B, 64, 32, 32]

        out = self.outconv1(up1)  # [B, 1, 32, 32]

        return out
    
if __name__ == '__main__':
    unet = UNet(in_channels=3, out_channels=3, n_steps=1000)
    
    B = 64
    t = torch.randint(0, 1000, (B, )) # .type(torch.float32)
    x = torch.randn(B, 3, 32, 32)
    c = torch.randint(0, 10, (B, ))
    output = unet(x, t)
    
    self_atteniton = SelfAttentionBlock(in_channels=3, out_channels=3)
    
    output = self_atteniton(x, x, x)
    print(output.shape)
