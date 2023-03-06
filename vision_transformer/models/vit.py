import torch
import torch.nn as nn

from .encoder_block import EncoderBlock
from .input_layer import InputLayer

# import matplotlib.pyplot as plt


class ViT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        latent_dim: int,
        embed_dim: int,
        patch_size: int,
        image_size_h: int,
        image_size_w: int,
        num_blocks: int,
        nb_head: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_layer = InputLayer(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            image_size_h=image_size_h,
            image_size_w=image_size_w,
        )
        self.encoder = nn.Sequential(
            *[
                EncoderBlock(
                    embed_dim=embed_dim,
                    nb_head=nb_head,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )

        self.decoder_lin = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 3 * 32),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1,unflattened_size=(32, 3, 3))

        self.latent_dim = latent_dim

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 10,stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 7, stride=2,padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, 3, stride=2,padding=1, output_padding=1)
        )

        
        
        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(normalized_shape=embed_dim),
        #     nn.Linear(in_features=embed_dim, out_features=num_classes),
        # )

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        result = self.decoder_lin(x)
        result = self.unflatten(result)
        result = self.decoder(result)
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, W, H) -> (B, N, D)
        out = self.input_layer(x)
        # (B, N, D) -> (B, N, D)
        out = self.encoder(out)
        # extract only class token
        # (B, N, D) -> (B, D)
        cls_token = out[:, 0]
        # (B, D) -> (B, M)
        recons = self.decode(cls_token)
        # pred = self.mlp_head(cls_token)
        return recons
