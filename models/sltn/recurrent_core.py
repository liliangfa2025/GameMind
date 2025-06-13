# models/sltn/recurrent_core.py

import torch
import torch.nn as nn

class StrategicRNN(nn.Module):
    """
    Recurrent architecture to model temporal consistency and strategic memory.
    Uses GRU by default.
    """

    def __init__(self, latent_dim=256, hidden_dim=256, num_layers=1):
        super(StrategicRNN, self).__init__()
        self.rnn = nn.GRU(latent_dim, hidden_dim, num_layers, batch_first=True)
        self.hidden_dim = hidden_dim

    def forward(self, latent_seq, hidden_state=None):
        """
        :param latent_seq: [B, T, latent_dim]
        :param hidden_state: [num_layers, B, hidden_dim] or None
        :return: output_seq: [B, T, hidden_dim], final_hidden
        """
        output_seq, final_hidden = self.rnn(latent_seq, hidden_state)
        return output_seq, final_hidden
