"""
Sample from a trained model
"""
import os
import pickle

import torch
import pytorch_lightning as L

from model import GPTConfig, GPT
from train import GPTLightningModule

# -----------------------------------------------------------------------------
out_dir = 'out'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 5 # number of samples to draw
max_new_tokens = 512 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 20 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 31297
device = 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
# -----------------------------------------------------------------------------
torch.manual_seed(seed)

ckpt_path = os.path.join(out_dir, 'ckpt.ckpt')
model = GPTLightningModule.load_from_checkpoint(ckpt_path)
config = model.hparams
model = model.gpt # Directly use the GPT model inside the LightningModule
model.eval()

load_meta = False
meta_path = os.path.join('data', config['dataset'], 'meta.pkl')
load_meta = os.path.exists(meta_path)

if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
with torch.no_grad():
    for k in range(num_samples):
        y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
        print(decode(y[0].tolist()))
        print('---------------')