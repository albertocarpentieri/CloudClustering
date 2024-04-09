import numpy as np
import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm as sn

def activation(act_type="swish"):
    act_dict = {"swish": nn.SiLU(),
                "gelu": nn.GELU(),
                "relu": nn.ReLU(),
                "tanh": nn.Tanh()}
    if act_type:
        if act_type in act_dict:
            return act_dict[act_type]
        else:
            raise NotImplementedError(act_type)
    elif not act_type:
        return nn.Identity()

def normalization(channels, norm_type="group", num_groups=1):
    if norm_type == "batch":
        return nn.BatchNorm3d(channels)
    elif norm_type == "group":
        return nn.GroupNorm(num_groups=num_groups, num_channels=channels)
    elif (not norm_type) or (norm_type.lower() == 'none'):
        return nn.Identity()
    else:
        raise NotImplementedError(norm_type)

class ResBlock2D(nn.Module):
    def __init__(
            self, in_channels, out_channels, resample=None,
            resample_factor=(1, 1), kernel_size=(3, 3),
            act='swish', norm='group', norm_kwargs=None,
            spectral_norm=False, upsampling_mode='nearest',
            **kwargs
    ):
        super().__init__(**kwargs)
        if in_channels != out_channels:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.proj = nn.Identity()

        padding = tuple(k // 2 for k in kernel_size)
        if resample == "down":
            self.resample = nn.AvgPool2d(resample_factor, ceil_mode=True)
            self.conv1 = nn.Conv2d(in_channels, out_channels,
                                   kernel_size=kernel_size, stride=resample_factor,
                                   padding=padding)
            self.conv2 = nn.Conv2d(out_channels, out_channels,
                                   kernel_size=kernel_size, padding=padding)
        elif resample == "up":
            self.resample = nn.Upsample(
                scale_factor=resample_factor, mode=upsampling_mode)
            self.conv1 = nn.ConvTranspose2d(in_channels, out_channels,
                                            kernel_size=kernel_size, padding=padding)
            output_padding = tuple(
                2 * p + s - k for (p, s, k) in zip(padding, resample_factor, kernel_size)
            )
            self.conv2 = nn.ConvTranspose2d(out_channels, out_channels,
                                            kernel_size=kernel_size, stride=resample_factor,
                                            padding=padding, output_padding=output_padding)
        else:
            self.resample = nn.Identity()
            self.conv1 = nn.Conv2d(in_channels, out_channels,
                                   kernel_size=kernel_size, padding=padding)
            self.conv2 = nn.Conv2d(out_channels, out_channels,
                                   kernel_size=kernel_size, padding=padding)

        if isinstance(act, str):
            act = (act, act)
        self.act1 = activation(act_type=act[0])
        self.act2 = activation(act_type=act[1])

        if norm_kwargs is None:
            norm_kwargs = {}
        self.norm1 = normalization(in_channels, norm_type=norm, **norm_kwargs)
        self.norm2 = normalization(out_channels, norm_type=norm, **norm_kwargs)
        if spectral_norm:
            self.conv1 = sn(self.conv1)
            self.conv2 = sn(self.conv2)
            if not isinstance(self.proj, nn.Identity):
                self.proj = sn(self.proj)

    def forward(self, x):
        x_in = self.resample(self.proj(x))
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.conv2(x)
        return x + x_in

class ResBlock3D(nn.Module):
    def __init__(
            self, in_channels, out_channels, resample=None,
            resample_factor=(1, 1, 1), kernel_size=(3, 3, 3),
            act='swish', norm='group', norm_kwargs=None,
            spectral_norm=False, upsampling_mode='nearest',
            **kwargs
    ):
        super().__init__(**kwargs)
        if in_channels != out_channels:
            self.proj = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.proj = nn.Identity()

        padding = tuple(k // 2 for k in kernel_size)
        if resample == "down":
            self.resample = nn.AvgPool3d(resample_factor, ceil_mode=True)
            self.conv1 = nn.Conv3d(in_channels, out_channels,
                                   kernel_size=kernel_size, stride=resample_factor,
                                   padding=padding)
            self.conv2 = nn.Conv3d(out_channels, out_channels,
                                   kernel_size=kernel_size, padding=padding)
        elif resample == "up":
            self.resample = nn.Upsample(
                scale_factor=resample_factor, mode=upsampling_mode)
            self.conv1 = nn.ConvTranspose3d(in_channels, out_channels,
                                            kernel_size=kernel_size, padding=padding)
            output_padding = tuple(
                2 * p + s - k for (p, s, k) in zip(padding, resample_factor, kernel_size)
            )
            self.conv2 = nn.ConvTranspose3d(out_channels, out_channels,
                                            kernel_size=kernel_size, stride=resample_factor,
                                            padding=padding, output_padding=output_padding)
        else:
            self.resample = nn.Identity()
            self.conv1 = nn.Conv3d(in_channels, out_channels,
                                   kernel_size=kernel_size, padding=padding)
            self.conv2 = nn.Conv3d(out_channels, out_channels,
                                   kernel_size=kernel_size, padding=padding)

        if isinstance(act, str):
            act = (act, act)
        self.act1 = activation(act_type=act[0])
        self.act2 = activation(act_type=act[1])

        if norm_kwargs is None:
            norm_kwargs = {}
        self.norm1 = normalization(in_channels, norm_type=norm, **norm_kwargs)
        self.norm2 = normalization(out_channels, norm_type=norm, **norm_kwargs)
        if spectral_norm:
            self.conv1 = sn(self.conv1)
            self.conv2 = sn(self.conv2)
            if not isinstance(self.proj, nn.Identity):
                self.proj = sn(self.proj)

    def forward(self, x):
        x_in = self.resample(self.proj(x))
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.conv2(x)
        return x + x_in


class Encoder(nn.Sequential):
    def __init__(
            self,
            dims=3, 
            in_dim=1, 
            levels=2, 
            min_ch=64, 
            max_ch=64, 
            time_compression_levels=[],
            extra_resblock_levels=[],
            downsampling_mode='resblock',
            norm=None):
        self.max_ch = max_ch 
        self.min_ch = min_ch
        
        sequence = []
        channels = np.hstack((in_dim, np.arange(1, (levels + 1)) * min_ch))
        channels[channels > max_ch] = max_ch
        channels[-1] = max_ch

        if dims == 2:
            res_block_fun = ResBlock2D
        elif dims == 3:
            res_block_fun = ResBlock3D

        for i in range(levels):
            in_channels = int(channels[i])
            out_channels = int(channels[i + 1])
            
            if dims == 2:
                kernel_size = (3, 3)
                resample_factor = (2, 2)
                stride_conv = nn.Conv2d
            elif dims == 3:
                stride_conv = nn.Conv3d
                kernel_size = (1, 3, 3)
                if i in time_compression_levels:
                    resample_factor = (2, 2, 2)
                else:
                    resample_factor = (1, 2, 2)
            
            if i in extra_resblock_levels:
                extra_block = res_block_fun(
                    in_channels, out_channels,
                    resample=None, 
                    kernel_size=kernel_size, 
                    resample_factor=None,
                    norm=norm,
                    norm_kwargs={"num_groups": 1})
                in_channels = out_channels
                sequence.append(extra_block)
            
            if downsampling_mode == 'resblock':
                downsample = res_block_fun(
                    in_channels, out_channels,
                    resample='down', 
                    kernel_size=kernel_size, 
                    resample_factor=resample_factor,
                    norm=None)
            
            elif downsampling_mode == 'stride':
                # kernel_size = (1, 2, 2)
                downsample = stride_conv(in_channels, out_channels,
                                         kernel_size=resample_factor, 
                                         stride=resample_factor)
            sequence.append(downsample)

        super().__init__(*sequence)

class Decoder(nn.Sequential):
    def __init__(
            self, 
            dims=3, 
            in_dim=1, 
            out_dim=1, 
            levels=2, 
            min_ch=64, 
            max_ch=64, 
            time_compression_levels=[],
            extra_resblock_levels=[],
            upsampling_mode='nearest',
            norm=None):
        self.max_ch = max_ch 
        self.min_ch = min_ch
        sequence = []
        channels = np.hstack((in_dim, np.arange(1, (levels + 1)) * min_ch))
        channels[channels > max_ch] = max_ch
        channels[0] = out_dim
        channels[-1] = max_ch
        
        if dims == 2:
            res_block_fun = ResBlock2D
        elif dims == 3:
            res_block_fun = ResBlock3D

        for i in reversed(list(range(levels))):
            in_channels = int(channels[i + 1])
            out_channels = int(channels[i])
            
            if dims == 2:
                kernel_size = (3, 3)
                resample_factor = (2, 2)
                stride_conv = nn.ConvTranspose2d
            elif dims == 3:
                kernel_size = (1, 3, 3)
                stride_conv = nn.ConvTranspose3d
                if i in time_compression_levels:
                    resample_factor = (2, 2, 2)
                else:
                    resample_factor = (1, 2, 2)
            
            if upsampling_mode == 'stride':
                # kernel_size = (1, 2, 2)
                upsample = stride_conv(in_channels, in_channels,
                                       kernel_size=resample_factor, stride=resample_factor)
            else:
                upsample = res_block_fun(
                    in_channels, 
                    in_channels,
                    resample='up', 
                    kernel_size=kernel_size, 
                    resample_factor=resample_factor,
                    norm=None,
                    upsampling_mode=upsampling_mode)
            sequence.append(upsample)
            
            if i in extra_resblock_levels:
                extra_block = res_block_fun(
                    in_channels, out_channels,
                    resample=None, 
                    kernel_size=kernel_size, 
                    resample_factor=None,
                    norm=norm,
                    norm_kwargs={"num_groups": 1})
                sequence.append(extra_block)
        super().__init__(*sequence)

class AutoEncoder(pl.LightningModule):
    def __init__(self,
                 encoder,
                 decoder,
                 lr,
                 encoded_channels,
                 hidden_width,
                 opt_patience,
                 **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=['encoder', 'decoder'])
        self.lr = lr
        self.encoder = encoder
        self.decoder = decoder
        self.hidden_width = hidden_width
        self.opt_patience = opt_patience
        if encoded_channels != hidden_width:
            self.to_latent = nn.Conv2d(encoded_channels, hidden_width,
                                       kernel_size=1)
            self.to_decoder = nn.Conv2d(hidden_width, encoded_channels,
                                        kernel_size=1)
        else:
            self.to_latent = None
            self.to_decoder = None
        

    def encode(self, x):
        h = self.encoder(x)
        if self.to_latent is not None:
            h = self.to_latent(h)
        return h

    def decode(self, z):
        if self.to_decoder is not None:
            z = self.to_decoder(z)
        dec = self.decoder(z)
        return dec

    def forward(self, x, sample_posterior=True):
        z = self.encode(x)
        x = self.decode(z)
        return x

    def _loss(self, batch):
        x = batch

        y_pred = self.forward(x)

        loss = (x - y_pred).abs().mean()
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._loss(batch)
        log_params = {"on_step": True, "on_epoch": True, "prog_bar": True, "sync_dist": True}
        self.log('train_loss', loss, **log_params)
        return loss

    @torch.no_grad()
    def val_test_step(self, batch, batch_idx, split="val"):
        loss = self._loss(batch)
        log_params = {"on_step": False, "on_epoch": True, "prog_bar": True, "sync_dist": True}
        self.log(f"{split}_loss", loss, **log_params)

    def validation_step(self, batch, batch_idx):
        self.val_test_step(batch, batch_idx, split="val")

    def test_step(self, batch, batch_idx):
        self.val_test_step(batch, batch_idx, split="test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr,
                                      betas=(0.5, 0.9), weight_decay=1e-3)
        reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=self.opt_patience, factor=0.25, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": reduce_lr,
                "monitor": "val_loss",
                "frequency": 1,
            },
        }