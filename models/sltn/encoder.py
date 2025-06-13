# models/sltn/encoder.py

import torch
import torch.nn as nn

class MultimodalEncoder(nn.Module):
    """
    Encodes visual (image), auditory, and symbolic inputs into a shared latent representation.
    """

    def __init__(self, visual_out=256, audio_out=128, symbol_out=64, fused_latent=256):
        super(MultimodalEncoder, self).__init__()

        # Visual encoder (e.g., CNN)
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, visual_out),
            nn.ReLU()
        )

        # Audio encoder (e.g., raw MFCC vectors)
        self.audio_encoder = nn.Sequential(
            nn.Linear(128, audio_out),
            nn.ReLU()
        )

        # Symbolic encoder (e.g., one-hot or embedding vectors)
        self.symbol_encoder = nn.Sequential(
            nn.Linear(symbol_out, symbol_out),
            nn.ReLU()
        )

        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(visual_out + audio_out + symbol_out, fused_latent),
            nn.ReLU()
        )

    def forward(self, image, audio, symbol):
        """
        :param image: [B, 3, H, W]
        :param audio: [B, 128]
        :param symbol: [B, symbol_dim]
        :return: fused latent vector [B, fused_latent]
        """
        v = self.visual_encoder(image)
        a = self.audio_encoder(audio)
        s = self.symbol_encoder(symbol)
        fused = torch.cat([v, a, s], dim=1)
        return self.fusion_layer(fused)
