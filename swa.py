import torch
import time

class SlidingWindowAttention(nn.Module):
    def __init__(self,n_heads,hidden_dim,window_size,context_len):
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.context_len = context_len
        self.to_q = nn.Linear(hidden_dim,hidden_dim)
        self.to_k = nn.Linear(hidden_dim,hidden_dim)
        self.to_v = nn.Linear(hidden_dim,hidden_dim)
        self.layernorm1 = nn.LayerNorm(hidden_dim)
        self.layernorm2 = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim,hidden_dim)
        self.dropout = nn.Dropout(p=0.5)
    def forward(self,x,apply_mask=False):
        B,L,C = x.shape
        H = self.n_heads
        D = self.head_dim
        q = self.to_q(x).reshape((B,L,H,D)).permute(0,2,1,3)
        k = self.to_k(x).reshape((B,L,H,D)).permute(0,2,1,3)
        v = self.to_v(x).reshape((B,L,H,D)).permute(0,2,1,3)
        mask = torch.full((L,L),float('-inf'))
        mask = torch.triu(mask) + torch.tril(mask,diagonal=-self.window_size) 
        #first term turns all the upper triangular elements to inf, second term turns bottom triangle for the causal SWA
        attn = q @ k.transpose(-1,-2)
        attn = attn + mask if apply_mask else attn
        scale = 1 / math.sqrt(D)
        a = (attn * scale).softmax(dim= -1)
        out = a @ v
        out = out.permute(0,2,1,3).reshape((B,L,C))
        out = self.proj(out)
        out = self.dropout(out)
        return out