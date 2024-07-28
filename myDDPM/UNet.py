import torch
import torch.nn as nn
import torch.nn.functional as F

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

class SelfAttentionBlock(nn.Module):
    """
    Self-attention blocks are applied at the 16x16 resolution in the original DDPM paper.
    Implementation is based on "Attention Is All You Need" paper, Vaswani et al., 2015
    (https://arxiv.org/pdf/1706.03762.pdf)
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        num_heads = 2,
        num_groups = 32,
    ):
        super(SelfAttentionBlock, self).__init__()
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

        
        q = q.view(B, C, H * W).transpose(1, 2)
        k = k.view(B, C, H * W).transpose(1, 2)
        v = v.view(B, C, H * W).transpose(1, 2)
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
        v = v.transpose(-1, -2).reshape(B, self.d_model, H, W)
        x = self.norm(linear_projection + v)
        return x
    
    def forward(self, x):
        return self.attention(x, x, x)
    

class UNetConv2D(nn.Module):
    def __init__(
        self, 
        in_size, 
        out_size, 
        is_batchnorm, 
        n = 3, 
        kernel_size=3, 
        stride = 1, 
        padding = 1,
        num_groups = 32
    ):
        super(UNetConv2D, self).__init__()
        self.n = n
        self.ks = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.shortcut = nn.Sequential()
        if kernel_size != 1 or in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups=num_groups, num_channels=out_size)
                # nn.BatchNorm2d(out_size)
            )
        
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(
                    nn.Conv2d(in_size, out_size, kernel_size, stride, padding),
                    #  nn.BatchNorm2d(out_size),
                    nn.GroupNorm(num_groups=num_groups, num_channels=out_size),
                    nn.SiLU(inplace=True)
                )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(
                    nn.Conv2d(in_size, out_size, kernel_size, stride, padding),
                    nn.SiLU(inplace=True)
                )
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
        self.time_embed = PositionalEmbedding(n_steps, time_emb_dim)

        # conditional variable embedding
        self.class_embed = PositionalEmbedding(n_classes, class_emb_dim)

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

        # Self-attention Block
        # self.conv4 = UNetConv2D(filters[2], filters[3], self.is_batchnorm)
        self.conv4 = SelfAttentionBlock(filters[2], filters[3])
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
    print(output.shape)