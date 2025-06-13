# models/sltn/latent_space.py

import torch
import torch.nn as nn

class LatentTransitionModel(nn.Module):
    """
    Predicts next latent state from current latent state and action embedding.
    """

    def __init__(self, latent_dim=256, action_dim=64):
        super(LatentTransitionModel, self).__init__()
        self.transition_net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, latent_state, action_embedding):
        """
        :param latent_state: [B, latent_dim]
        :param action_embedding: [B, action_dim]
        :return: next_latent_state: [B, latent_dim]
        """
        x = torch.cat([latent_state, action_embedding], dim=1)
        return self.transition_net(x)
