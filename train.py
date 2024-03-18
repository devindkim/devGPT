import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as L
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.callbacks import ModelCheckpoint

from model import GPTConfig, GPT

class GPTDataset(Dataset):
    def __init__(self, data_file, L):
        super().__init__()
        self.data_file = data_file
        self.L = L
        self.data = np.memmap(self.data_file, dtype=np.uint16, mode='r')
        self.data_length = self.data.shape[0]

    def __len__(self):
        # Return the number of possible sequences based on data length and sequence length
        return self.data_length - self.L

    def __getitem__(self, idx):
        # Generate a single input-target pair
        start_index = idx
        end_index = start_index + self.L + 1
        sequence = torch.tensor(self.data[start_index:end_index], dtype=torch.long)
        input_sequence = sequence[:-1]  # Exclude the last token from input
        target_sequence = sequence[1:]  # Exclude the first token from target
        return input_sequence, target_sequence

class GPTDataModule(L.LightningDataModule):
    def __init__(self, dataset_path, B, L):
        super().__init__()
        self.dataset_path = dataset_path
        self.B = B
        self.L = L

    def setup(self, stage=None):
        # Create datasets for training and validation
        train_data_file = os.path.join(self.dataset_path, 'train.bin')
        val_data_file = os.path.join(self.dataset_path, 'val.bin')
        self.train_dataset = GPTDataset(train_data_file, self.L)
        self.val_dataset = GPTDataset(val_data_file, self.L)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.B, shuffle=True, num_workers=4, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.B, shuffle=False, num_workers=4, pin_memory=True)


class GPTLightningModule(L.LightningModule):
    def __init__(self, **config):
        super().__init__()
        self.save_hyperparameters(config)
        model_args = ['V','H','D','L','num_layers','dropout']
        gpt_config = {k: v for k, v in config.items() if k in model_args}
        self.gpt = GPT(GPTConfig(**gpt_config))

    def training_step(self, batch, batch_idx):
        X, Y = batch
        logits, loss = self.gpt(X, Y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X, Y = batch
        logits, loss = self.gpt(X, Y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        config = self.hparams
        weight_decay, learning_rate, betas = config['weight_decay'], config['learning_rate'], (config['beta1'], config['beta2'])
        # Collect params that require gradients, otherwise they cannot be optimized.
        params = [p for _, p in self.gpt.named_parameters() if p.requires_grad]

        # All weight tensors will be optimized together with decay, and all else will be optimized separately.
        decay_params = [p for p in params if p.dim() >= 2] # Weight tensors
        nodecay_params = [p for p in params if p.dim() < 2] # Bias, LayerNorm, etc.
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer


def get_vocab_size(config):
    data_dir = os.path.join('data', config['dataset'])
    meta_path = os.path.join(data_dir, 'meta.pkl')
    meta_V = None
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        meta_V = meta['V']
        print(f"Vocab_size: {meta_V} (inside {meta_path})")
    return meta_V

def init_model(config):
        model_args = {'V': config['V'], 'H': config['H'], 'D': config['D'], 'L': config['L'], 'num_layers': config['num_layers'], 'dropout': config['dropout']}
        if config['init_from'] == 'scratch':
            print("Initializing a new model from scratch")
            config['V'] = get_vocab_size(config)
            model = GPTLightningModule(**config)
            return model
        elif config['init_from'] == 'resume':
            print(f"Resuming training from {config['out_dir']}")
            ckpt_path = os.path.join(config['out_dir'], 'ckpt.ckpt')
            model = GPTLightningModule.load_from_checkpoint(ckpt_path)
            return model

if __name__ == "__main__":
    config = {
        # Data
        'dataset': 'avatar',
        'B': 32,  # Batch Size
        'L': 128,  # Sequence Length
        'V': None,  # Vocab size, will be set by the dataset

        # Model
        'num_layers': 8,  # Number of transformer layers
        'H': 4,  # Number of attention heads
        'D': 64,  # Model dimension
        'dropout': 0.0,  # For pretraining 0 is good, for finetuning try 0.1+
        'init_from': None,

        # I/O
        'out_dir': 'out',

        # AdamW optimizer
        'weight_decay': 1e-1,
        'learning_rate': 6e-4,
        'beta1': 0.9,
        'beta2': 0.95,

        # System
        'device': 'mps',
    }
    config['tokens_per_iter'] = config['B'] * config['L']
    config['init_from'] = 'resume' # either 'scratch' or 'resume'
    model = init_model(config)
    dataset_path = os.path.join('data', config['dataset'])
    data_module = GPTDataModule(dataset_path=dataset_path, B=config['B'], L=config['L'])

    ckpt_dir = config['out_dir']
    checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dir, filename='ckpt', save_top_k=1, every_n_train_steps=100)
    profiler = PyTorchProfiler(profile_memory=True, with_stack=True)
    trainer = L.Trainer(max_epochs=1, default_root_dir=ckpt_dir, profiler=profiler, callbacks=[checkpoint_callback])
    trainer.fit(model, datamodule=data_module)
    print(profiler.summary())