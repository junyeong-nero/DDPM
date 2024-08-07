import torch
import torch.nn as nn

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
        
        B, C, H, W = v.shape
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
        v = v.transpose(-1, -2).reshape(B, self.d_model, H, W)
        if self.is_batchnorm:
            x = self.norm(linear_projection + v)
        return x
    
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