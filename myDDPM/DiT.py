import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import math

# Helper function to generate sinusoidal positional embeddings
def sinusoidal_embedding(n, d):
    # Returns the standard positional embedding
    embedding = torch.tensor([[i / 10_000 ** (2 * j / d) for j in range(d)] for i in range(n)])
    sin_mask = torch.arange(0, n, 2)

    embedding[sin_mask] = torch.sin(embedding[sin_mask])
    embedding[1 - sin_mask] = torch.cos(embedding[sin_mask])

    return embedding


# Define the Transformer encoder layer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask,
                                 key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

# Define the DiT model
class DiT(nn.Module):
    def __init__(self, image_size, in_channels, n_layers, d_model, nhead, dim_feedforward, dropout=0.1):
        super(DiT, self).__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.d_model = d_model
        self.positional_encoding = nn.Parameter(torch.zeros(1, image_size * image_size, d_model))
        self.input_projection = nn.Linear(in_channels, d_model)
        self.time_embedding = nn.Sequential(
            nn.Linear(d_model // 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        for _ in range(n_layers)])
        self.output_projection = nn.Linear(d_model, in_channels)

    def forward(self, x, t):
        b, c, h, w = x.size()
        x = x.view(b, c, h * w).transpose(1, 2)  # Shape: (batch_size, seq_len, in_channels)
        x = self.input_projection(x)  # Project input to model dimensions
        pos_emb = self.positional_encoding[:, :x.size(1)]
        print(pos_emb.shape)
        
        time_emb = self.time_embedding(sinusoidal_embedding(t, self.d_model))
        x = x + pos_emb + time_emb.unsqueeze(1)  # Add positional and time embeddings

        for layer in self.layers:
            x = layer(x)
        
        x = self.output_projection(x)
        x = x.transpose(1, 2).view(b, c, h, w)  # Reshape back to image format
        return x

if __name__ == '__main__':
    model = DiT(image_size=32, 
                in_channels=3, 
                n_layers=3, 
                d_model=1000, 
                nhead=4,
                dim_feedforward=3072)

    # print(model)    
    B = 64
    t = torch.randint(0, 1000, (B, )) # .type(torch.float32)
    x = torch.randn(B, 3, 32, 32)
    output = model(x, t)

    # print(output.shape)