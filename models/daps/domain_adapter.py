# models/daps/domain_adapter.py

import torch
import torch.nn as nn

class DomainAdaptiveModule(nn.Module):
    """
    Learns to adjust latent features or action plans under shifting environment dynamics or rules.
    """

    def __init__(self, latent_dim=256, context_dim=64):
        super(DomainAdaptiveModule, self).__init__()

        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, 64),
            nn.ReLU()
        )

        self.fusion = nn.Sequential(
            nn.Linear(latent_dim + 64, 256),
            nn.ReLU()
        )

    def forward(self, latent_state, env_context):
        """
        :param latent_state: [B, latent_dim]
        :param env_context: [B, context_dim] (e.g., rule vector, level descriptor)
        :return: adapted_latent: [B, 256]
        """
        context_feat = self.context_encoder(env_context)
        fused = torch.cat([latent_state, context_feat], dim=1)
        return self.fusion(fused)
