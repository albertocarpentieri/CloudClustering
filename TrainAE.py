import numpy as np
import pickle as pkl
import torch
from utils import Encoder, Decoder, AutoEncoder
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import Dataset, DataLoader
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

class Dataset(Dataset):
    def __init__(self, 
                 data_path,
                 start_idx,
                 end_idx,
                 validation=False):
        self.data_path = data_path
        self.filenames = os.listdir(data_path)[start_idx:end_idx] 
        self.validation = validation
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self,
                    idx):
        filename = self.filenames[idx]
        with open(self.data_path+filename, 'rb') as o:
            patches = pkl.load(o)
        patches = np.stack(patches, axis=0).astype(np.float32)/255
        if not self.validation:
            k = np.random.choice([-1,0,1,2])
            patches = np.rot90(patches, k=k, axes=(2,3)).copy()
        return torch.tensor(patches)
    
def my_collate_fn(batch):
    return torch.concat(batch, dim=0)

def train():
    train_dataset = Dataset(data_path='/scratch/snx3000/acarpent/train_patches/',
                        start_idx=0,
                        end_idx=3000,
                        validation=False)
    val_dataset = Dataset(
        data_path='/scratch/snx3000/acarpent/train_patches/',
        start_idx=3000,
        end_idx=-1,
        validation=True)
    
    train_dataloader = DataLoader(
        train_dataset,
        num_workers=24,
        batch_size=8,
        shuffle=True,
        collate_fn=my_collate_fn)

    val_dataloader = DataLoader(
        val_dataset,
        num_workers=0,
        batch_size=8,
        shuffle=True,
        collate_fn=my_collate_fn)
    
    encoder = Encoder(dims=2, 
                      in_dim=3, 
                      levels=6, 
                      min_ch=64, 
                      max_ch=256, 
                      time_compression_levels=[],
                      extra_resblock_levels=[0, 1, 2, 3, 4, 5],
                      downsampling_mode='resblock',
                      norm='group')

    decoder = Decoder(dims=2, 
                      in_dim=32, 
                      out_dim=3, 
                      levels=6, 
                      min_ch=64, 
                      max_ch=256, 
                      time_compression_levels=[],
                      extra_resblock_levels=[0, 1, 2, 3, 4, 5],
                      upsampling_mode='nearest',
                      norm=None)
    
    autoencoder = AutoEncoder(
        encoder,
        decoder,
        lr=0.001,
        encoded_channels=256,
        hidden_width=8,
        opt_patience=3)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='/scratch/snx3000/acarpent/CloudClustering/',
        filename='CloudsAE2-{epoch}-{val_loss:.5f}',
        save_top_k=1,
        every_n_epochs=1
    )

    early_stop_callback = EarlyStopping(monitor='val_loss',
                                        patience=6)
    
    logger = CSVLogger('/scratch/snx3000/acarpent/CloudClustering/', name='CloudsAE2')
    trainer = pl.Trainer(
        default_root_dir='/scratch/snx3000/acarpent/CloudClustering/',
        accelerator='gpu',
        devices=1,
        num_nodes=1,
        max_epochs=1000,
        callbacks=[checkpoint_callback, early_stop_callback],
        strategy='ddp',
        precision='32',
        enable_progress_bar=True,
        deterministic=True,
        logger=logger
    )
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    trainer.fit(autoencoder, train_dataloader, val_dataloader)

if __name__ == '__main__':
    train()
    