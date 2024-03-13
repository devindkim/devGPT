import os
import time
import math
import pickle

import numpy as np
import torch
import wandb

from model import GPTConfig, GPT

class Trainer:

    def get_default_config(self):
        """Returns a default configuration dictionary for the trainer."""

        config = {
            # Data
            'dataset': 'avatar',
            'B': 8,  # Batch Size
            'L': 64,  # Sequence Length
            'V': None,  # Vocab size, will be set by the dataset

            # Model
            'num_layers': 4,  # Number of transformer layers
            'H': 4,  # Number of attention heads
            'D': 64,  # Model dimension
            'dropout': 0.0,  # For pretraining 0 is good, for finetuning try 0.1+
            'init_from': 'scratch',  # either 'scratch' or 'resume'

            # WandB logging
            'wandb_log': False,  # disabled by default
            'wandb_project': 'devGPT',
            'wandb_run_name': 'devGPT',  # 'run' + str(time.time())

            # I/O
            'out_dir': 'out',
            'eval_interval': 2000,
            'log_interval': 1,
            'eval_iters': 1,
            'eval_only': False,
            'always_save_checkpoint': True,

            # AdamW optimizer
            'weight_decay': 1e-1,
            'learning_rate': 6e-4,
            'beta1': 0.9,
            'beta2': 0.95,
            'min_lr': 6e-5,
            'max_iters': 1000000,
            'lr_decay_iters': 1000000,
            'grad_clip': 1.0,
            'decay_lr': True,
            'warmup_iters': 2000,
            
            # System
            'device': 'mps',
            
            # Additional settings
            'seed': 31297,
        }
        config['tokens_per_iter'] = config['B'] * config['L']
        return config

    def __init__(self, config=None):
        if config is None:
            config = self.get_default_config()
        self.config = config

    def run(self):      
        config = self.config  
        torch.manual_seed(config['seed'])
        os.makedirs(config['out_dir'], exist_ok=True)
        
        iter_num, best_val_loss = 0, 1e9
        model, optimizer = self.init_model()

        if config['wandb_log']:
            wandb.init(project=wandb_project, name=wandb_run_name, config=config)

        t0 = time.time()
        raw_model = model
        running_mfu = -1.0
        while iter_num <= config['max_iters']:
            learning_rate = self.get_lr(iter_num)
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

            # Evaluate loss and write checkpoints
            if iter_num % config['eval_interval'] == 0:
                losses = self.estimate_loss(model)
                print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                if config['wandb_log']:
                    wandb.log({
                        "iter": iter_num,
                        "train/loss": losses['train'],
                        "val/loss": losses['val'],
                        "lr": lr,
                        "mfu": running_mfu*100, # convert to percentage
                    })
                if losses['val'] < best_val_loss or config['always_save_checkpoint']:
                    best_val_loss = losses['val']
                    if iter_num > 0:
                        model_args = {'V': config['V'], 'H': config['H'], 'D': config['D'], 'L': config['L'], 'num_layers': config['num_layers'], 'dropout': config['dropout']}
                        checkpoint = {
                            'model': raw_model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'iter_num': iter_num,
                            'best_val_loss': best_val_loss,
                            'model_args': model_args,
                            'config': config,
                        }
                        print(f"saving checkpoint to {config['out_dir']}")
                        torch.save(checkpoint, os.path.join(config['out_dir'], 'ckpt.pt'))
            if iter_num == 0 and config['eval_only']:
                break

            X, Y = self.get_batch('train')
            logits, loss = model(X, Y)
            loss.backward()
            if config['grad_clip'] != 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if iter_num % config['log_interval'] == 0:
                lossf = loss.item()
                mfu = raw_model.estimate_mfu(config['B'], dt)
                running_mfu = mfu if running_mfu <= 0 else 0.9*running_mfu + 0.1*mfu
                print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
            iter_num += 1

    def get_batch(self, split):
        config = self.config
        data_dir, L, B = os.path.join('data', config['dataset']), config['L'], config['B']
        if split == 'train':
            data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        else:
            data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        ix = torch.randint(len(data) - L, (B,))
        x = torch.stack([torch.from_numpy((data[i:i+L]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+L]).astype(np.int64)) for i in ix])
        x, y = x.to(config['device']), y.to(config['device'])
        return x, y

    def get_vocab_size(self):
        config = self.config
        data_dir = os.path.join('data', config['dataset'])
        meta_path = os.path.join(data_dir, 'meta.pkl')
        meta_V = None
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            meta_V = meta['V']
            print(f"Vocab_size: {meta_V} (inside {meta_path})")
        return meta_V

    @torch.no_grad()
    def estimate_loss(self, model):
        config = self.config
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(config['eval_iters'])
            for k in range(config['eval_iters']):
                X, Y = self.get_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    def get_lr(self, it):
        config = self.config

        if not config['decay_lr']:
            return config['learning_rate']

        if it < config['warmup_iters']:
            return config['learning_rate'] * it / config['warmup_iters']
        if it > config['lr_decay_iters']:
            return config['min_lr']
        decay_ratio = (it - config['warmup_iters']) / (config['lr_decay_iters'] - config['warmup_iters'])

        if decay_ratio > 1 or decay_ratio < 0:
            raise ValueError(f"decay_ratio {decay_ratio} should be in [0, 1]")

        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # Cosine decay
        return config['min_lr'] + coeff * (config['learning_rate'] - config['min_lr'])

    def init_model(self):
        config = self.config
        model_args = {'V': config['V'], 'H': config['H'], 'D': config['D'], 'L': config['L'], 'num_layers': config['num_layers'], 'dropout': config['dropout']}
        if config['init_from'] == 'scratch':
            print("Initializing a new model from scratch")
            model_args['V'] = self.get_vocab_size()
            self.config['V'] = model_args['V']
            gptconf = GPTConfig(**model_args)
            model = GPT(gptconf)
        elif config['init_from'] == 'resume':
            print(f"Resuming training from {config['out_dir']}")
            ckpt_path = os.path.join(config['out_dir'], 'ckpt.pt')
            checkpoint = torch.load(ckpt_path, map_location=config['device'])
            checkpoint_model_args = checkpoint['model_args']

            # These config attributes must be equal to resume training.
            for k in ['num_layers', 'H', 'D', 'L', 'V']:
                model_args[k] = checkpoint_model_args[k]
            gptconf = GPTConfig(**model_args)
            model = GPT(gptconf)

            state_dict = checkpoint['model']
            model.load_state_dict(state_dict)

        model.to(config['device'])
        optimizer = model.configure_optimizer(config['weight_decay'], config['learning_rate'], (config['beta1'], config['beta2']))

        if config['init_from'] == 'resume':
            optimizer.load_state_dict(checkpoint['optimizer'])
        checkpoint = None

        return model, optimizer

if __name__ == "__main__":
    trainer = Trainer()
    trainer.run()
