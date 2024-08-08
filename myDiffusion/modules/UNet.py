import torch
import torch.nn as nn
from layer import SelfAttentionBlock, PositionalEmbedding, WideResNetBlock, MultiHeadAttentionBlock

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
    
if __name__ == '__main__':
    unet = UNet(in_channels=3, 
                out_channels=3, 
                n_steps=1000,
                custom_scale=[128, 128, 256, 256, 512, 512])
    
    B = 1
    t = torch.randint(0, 1000, (B, )) # .type(torch.float32)
    x = torch.randn(B, 3, 32, 32)
    c = torch.randint(0, 10, (B, ))
    output = unet(x, t)
    print(output.shape)