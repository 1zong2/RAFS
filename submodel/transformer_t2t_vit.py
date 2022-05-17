import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import DropPath


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, in_dim = None, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, in_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(in_dim, in_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape # [8, 12, 512]
        # self.qkv(x) # [8, 12, 1536]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.in_dim//self.num_heads).permute(2, 0, 3, 1, 4) # [8, 12, 1536] -> [3, 8, 8, 12, 64]
        q, k, v = qkv[0], qkv[1], qkv[2] # [8, 8, 12, 64], [8, 8, 12, 64], [8, 8, 12, 64]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn) # [8, 8, 12, 12]

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.in_dim) # [8, 8, 12, 64] -> [8, 12, 8, 64] --> [8, 12, 512]
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Attention_3(nn.Module):
    def __init__(self, q_dim, v_dim, k_dim, num_heads=8, in_dim = None, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim
        head_dim = q_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.fc_q = nn.Linear(q_dim, in_dim, bias=qkv_bias)
        self.fc_k = nn.Linear(v_dim, in_dim, bias=qkv_bias)
        self.fc_v = nn.Linear(k_dim, in_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(in_dim, in_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, N, C = q.shape

        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.in_dim).permute(2, 0, 3, 1, 4)
        q = self.fc_q(q).reshape(B, 4096, 1, self.num_heads, self.in_dim//self.num_heads).permute(2, 0, 3, 1, 4).squeeze(0)
        k = self.fc_k(k).reshape(B, 12, 1, self.num_heads, self.in_dim//self.num_heads).permute(2, 0, 3, 1, 4).squeeze(0)
        v = self.fc_v(v).reshape(B, 12, 1, self.num_heads, self.in_dim//self.num_heads).permute(2, 0, 3, 1, 4).squeeze(0)

        # print(q.size()) # [8, 8, 4096, 64]
        # print(k.size()) # [8, 8, 12, 64]
        # print(v.size()) # [8, 8, 12, 64]

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn) # [8, 8, 4096, 12]

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.in_dim)  # [8, 4096, 512]
        x = self.proj(x)
        x = self.proj_drop(x) 

        return x

class Token_transformer(nn.Module):

    def __init__(self, dim, in_dim, num_heads, mlp_ratio=1., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, in_dim=in_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(in_dim)
        self.mlp = Mlp(in_features=in_dim, hidden_features=int(in_dim*mlp_ratio), out_features=in_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = self.attn(self.norm1(x))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
