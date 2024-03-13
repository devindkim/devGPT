"""Full GPT model implementation, including the GPT class and all submodules.

Dimension key:
B: batch size
L: sequence length
M: memory length (length of sequence being attended to)
D: model dimension (sometimes called d_model or embedding_dim)
V: vocabulary size
F: feed-forward subnetwork hidden size
H: number of attention heads in a layer
K: size of each attention key or value (sometimes called d_kv)

Reference: 
https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F # TODO: Change import to not overlap with dimension key

class CausalSelfAttention(nn.Module):
    """Implements Causal Self Attention used within a Transformer block.

    Attributes:
        c_attn (nn.Linear): Linear layer for combined key, query, value projections.
        c_proj (nn.Linear): Linear layer for the output projection.
        attn_dropout (nn.Dropout): Dropout layer for attention weights.
        resid_dropout (nn.Dropout): Dropout layer for the output.
        H (int): Number of attention heads.
        D (int): Model dimension.
        dropout (float): Dropout rate.
        bias (torch.Tensor): Bias tensor for masked attention.
    """

    def __init__(self, config):
        super().__init__()
        if config.D % config.H != 0:
            raise ValueError(f"Model dimension {config.D} must be divisible by number of heads {config.H}")

        self.c_attn = nn.Linear(config.D, 3*config.D)
        self.c_proj = nn.Linear(config.D, config.D)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.H = config.H
        self.D = config.D
        self.dropout = config.dropout
        
        self.register_buffer(
            "bias", 
            torch.tril(torch.ones(config.L, config.L)).view(1, 1, config.L, config.L)
        )


    def forward(self, input_BLD):
        B, L, D = input_BLD.size()
        K = D // self.H

        # Split input into query, key, and value
        query, key, value  = self.c_attn(input_BLD).split(self.D, dim=2)
        key_BHLK = key.view(B, L, self.H, K).transpose(1, 2) 
        query_BHLK = query.view(B, L, self.H, K).transpose(1, 2) 
        value_BHLK = value.view(B, L, self.H, K).transpose(1, 2)

        # Scale dot-product attention with masking
        logits_BHLL = torch.einsum('BHLK,BHKM->BHLM', query_BHLK, key_BHLK.transpose(-2, -1)*(1.0/math.sqrt(K)))
        masked_logits_BHLL = logits_BHLL.masked_fill(self.bias[:,:,:L,:L] == 0, float('-inf'))
        weights_BHLL = torch.softmax(masked_logits_BHLL, dim=-1)
        weights_BHLL = self.attn_dropout(weights_BHLL)

        # Apply attention to value
        w_value_BHLK = torch.einsum('BHLL,BHLK->BHLK', weights_BHLL, value_BHLK)

        # Final projection and dropout
        out_BLD = w_value_BHLK.transpose(1, 2).contiguous().view(B, L, D)
        out_BLD = self.resid_dropout(self.c_proj(out_BLD))

        return out_BLD

class MLP(nn.Module):
    """Implements the feedforward network (MLP) used within a Transformer block.

    Attributes:
        c_fc (nn.Linear): First linear layer expanding input dimensions.
        gelu (nn.GELU): GELU non-linearity.
        c_proj (nn.Linear): Second linear layer projecting dimensions back.
        dropout (nn.Dropout): Dropout layer applied after the second linear layer.
    """

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.D, 4*config.D)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4*config.D, config.D)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_BLD):
        hidden_BL4D = self.c_fc(input_BLD)  
        hidden_BL4D = self.gelu(hidden_BL4D)  
        output_BLD = self.c_proj(hidden_BL4D)
        output_BLD = self.dropout(output_BLD) 
        return output_BLD

class Block(nn.Module):
    """Implements a single Transformer block, combining self-attention and a feedforward network.

    Attributes:
        ln_1 (nn.LayerNorm): Layer normalization before self-attention.
        attn (CausalSelfAttention): The self-attention mechanism.
        ln_2 (nn.LayerNorm): Layer normalization before the feedforward network.
        mlp (MLP): The feedforward network (multi-layer perceptron).
    """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.D)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.D)
        self.mlp = MLP(config)

    def forward(self, input_BLD):
        attn_BLD = self.attn(self.ln_1(input_BLD))
        resid_attn_BLD = input_BLD + attn_BLD
        mlp_BLD = self.mlp(self.ln_2(resid_attn_BLD))
        output_BLD = resid_attn_BLD + mlp_BLD   
        return output_BLD

@dataclass
class GPTConfig:
    """Configuration for the GPT model."""
    V: int = 128 # Vocabulary size
    H: int = 4 # Number of attention heads
    D: int = 64 # Model dimension
    L: int = 64 # Maximum sequence length
    num_layers: int = 12 # Number of Transformer blocks
    dropout: float = 0.0 # Dropout rate

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        if not config.V:
            raise ValueError("config.V must be set to the size of the vocabulary")
        if not config.L:
            raise ValueError("config.L must be set to the maximum sequence length")

        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.V, config.D),
            wpe = nn.Embedding(config.L, config.D),
            dropout = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.num_layers)]),
            ln_f = nn.LayerNorm(config.D),
        ))
        self.lm_head = nn.Linear(config.D, config.V, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for name, param in self.named_parameters():
            if name.endswith('c_proj.weight'):
                torch.nn.init.normal_(param, mean=0.0, std=0.02/math.sqrt(2*config.num_layers))

        print("Total Params: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_BL, targets_BL=None):
        """Processes input through the GPT model and calculates loss if targets are provided.

        Args:
            input_BL (torch.Tensor): Input tensor of shape BL, where B is the batch size and L is the sequence length.
            targets_BL (torch.Tensor, optional): Target tensor of the same shape as input_BL. Used for calculating loss.

        Returns:
            tuple: A tuple containing:
                - logits (torch.Tensor): Logits tensor of shape BLV, where V is the vocabulary size.
                - loss (torch.Tensor, optional): The computed loss value, if targets are provided. None otherwise.

        Raises:
            ValueError: If the input sequence length exceeds the configured block size.
        """
        B, L = input_BL.size()
        if L > self.config.L:
            raise ValueError(f"Cannot forward sequence of length {L}, block size is only {self.L}")
        pos_L = torch.arange(0, L, dtype=torch.long, device=input_BL.device)

        tok_emb_BLD = self.transformer.wte(input_BL)
        pos_emb_LD = self.transformer.wpe(pos_L)
        input_BLD = self.transformer.dropout(tok_emb_BLD + pos_emb_LD)
        for block in self.transformer.h:
            input_BLD = block(input_BLD)
        output_BLD = self.transformer.ln_f(input_BLD)

        logits_BLV = self.lm_head(output_BLD) if targets_BL is not None else self.lm_head(output_BLD[:, [-1], :])
        loss = F.cross_entropy(logits_BLV.view(-1, logits_BLV.size(-1)), targets_BL.view(-1), ignore_index=-1) if targets_BL is not None else None

        return logits_BLV, loss


    def configure_optimizer(self, weight_decay, learning_rate, betas):
        # Collect params that require gradients, otherwise they cannot be optimized.
        params = [p for _, p in self.named_parameters() if p.requires_grad]

        # All weight tensors will be optimized together with decay, and all else will be optimized separately.
        decay_params = [p for p in params if p.dim() >= 2] # Weight tensors
        nodecay_params = [p for p in params if p.dim() < 2] # Bias, LayerNorm, etc.
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"Decaying Tensors:\t{len(decay_params)}\nDecaying Params:\t{num_decay_params:,}")
        print(f"Non-Decaying Tensors:\t{len(nodecay_params)}\nNon-Decaying Params:\t{num_nodecay_params:,}")

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ Estimate Model Flops Utilization (MFU) in units of Apple M2 Chip float16 peak FLOPS """
        N = self.get_num_params()
        num_layers, H, K, L = self.config.num_layers, self.config.H, self.config.D//self.config.H, self.config.L
        flops_per_token = 6*N + 12*num_layers*H*K*L # PaLM paper Appendix B https://arxiv.org/abs/2204.02311
        flops_per_fwdbwd = flops_per_token * L
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter

        flops_achieved = flops_per_iter * (1.0/dt) # Per second
        flops_promised = 3.6e12 # Apple M2 Chip float16 peak flops is 3.6 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.L else idx[:, -self.config.L:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx