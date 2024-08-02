import torch
import torch.nn as nn

# vocab_size:32000
# hidden_dim: 4096
# mlp_hidden_dim: 14336
# out_hidden_dim: 1024
# num_layers: 32



class MistralModel:
    def __init__(self,vocab_size,hidden_dim,out_hidden_dim,mlp_hidden_dim,num_layers):
        self.embed_tokens = nn.Embedding(vocab_size,hidden_dim)
        self.layers = nn.ModuleList([
            MistralTransformerBlocks(hidden_dim,mlp_hidden_dim,out_hidden_dim) for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim,vocab_size)
    def forward(self,x):
        x = self.embed_tokens(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.lm_head(x)
        return x


class MistralTransformerBlocks(nn.Module):
    def __init__(self,hidden_dim,mlp_hidden_dim,out_hidden_dim):
        self.hidden_dim = hidden_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.out_hidden_dim = out_hidden_dim
        self.input_layer_norm = nn.LayerNorm(hidden_dim)
        self.post_attention_layernorm = nn.LayerNorm(hidden_dim)
        self.self_attn = MistralSelfAttention(hidden_dim,out_hidden_dim)
        self.mlp = MistralMLP(hidden_dim=hidden_dim,mlp_hidden_dim=mlp_hidden_dim)

    def forward(self,x):
        x = self.self_attn(self.input_layer_norm(x))+x
        x = self.mlp(self.post_attention_layernorm(x)) +x
        return x

# KV Cache: https://www.omrimallis.com/posts/techniques-for-kv-cache-optimization/
class MistralSelfAttention(nn.Module):
    def __init__(self,hidden_dim,out_hidden_dim):
        self.hidden_dim = hidden_dim
        self.out_hidden_dim = out_hidden_dim
        self.q_rope = MistralRoPE(hidden_dim)
        self.k_rope = MistralRoPE(hidden_dim)
        self.q_proj =nn.Linear(hidden_dim,hidden_dim)
        self.k_proj = nn.Linear(out_hidden_dim,hidden_dim)
        self.v_proj = nn.Linear(out_hidden_dim,hidden_dim)
        self.o_proj = nn.Linear(hidden_dim,hidden_dim)
    def forward(self,x):
        B,L,C = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        




class MistralMLP(nn.Module):
    def __init__(self,hidden_dim,mlp_hidden_dim):
        self.hidden_dim = hidden_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.down_proj = nn.Linear(hidden_dim,mlp_hidden_dim)
        self.gate_proj = nn.Linear(mlp_hidden_dim,hidden_dim)
        self.up_proj = nn.Linear(mlp_hidden_dim,hidden_dim)
    def forward(self,x):
        pass

# https://medium.com/@ngiengkianyew/understanding-rotary-positional-encoding-40635a4d078e
class MistralRoPE:
    def __init__(self,d,base):
        self.base = base
        self.d = d
        self.cos_cached = None
        self.sin_cached = None    
    def build_cache(self, x: torch.Tensor):
        if self.cos_cached is not None and x.shape[0] <= self.cos_cached.shape[0]:
            return
        # **sequence length** of the feature tensor
        seq_len = x.shape[0] 
        # theta is of shape (self.d//2), 1/10000^{i/d} for i goes from 0 to d by 2
        # NOTE: the exponent of 10000 is normalized by d
        theta = 1. / (self.base ** (torch.arange(0, self.d, 2).float() / self.d)).to(x.device)
        # seq_idx is of shape (seq_len) e.g.[0,1,2,3,4...seq_len]
        seq_idx = torch.arange(seq_len, device=x.device).float().to(x.device)
        # seq_idx dot product with theta -> (seq_len,1)*(1,self.d//2) = (seq_len,self.d//2)
        #                                     seq_idx      theta
        # so each pos has a vector of size self.d//2
        idx_theta = torch.einsum('n,d->nd', seq_idx, theta)
        # repeat itself along dim=1 so now idx_theta is of shape (seq_len,self.d)
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)
        # apply cos and sin on idx_theta 
        # each cos and sin copy is of shape (seq_len,1,1,self.d)
        self.cos_cached = idx_theta2.cos()[:, None, None, :]
        self.sin_cached = idx_theta2.sin()[:, None, None, :]
 
    def _neg_half(self,x:torch.Tensor):
        d_2 = self.d//2
        return torch.cat(-x[:,:,:,d_2:],x[:,:,:,:d_2],dim=-1)
    def forward(self, x: torch.Tensor):
        self._build_cache(x)
        x_rope, x_pass = x[..., :self.d], x[..., self.d:]
        neg_half_x = self._neg_half(x_rope)
        x_rope = (x_rope * self.cos_cached[:x.shape[0]]) + (neg_half_x * self.sin_cached[:x.shape[0]])
        return torch.cat((x_rope, x_pass), dim=-1)

