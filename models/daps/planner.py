# models/daps/planner.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalPlanner(nn.Module):
    """
    High-level planner for selecting subgoals or abstract strategies based on latent representations.
    """

    def __init__(self, latent_dim=256, num_subgoals=8):
        super(HierarchicalPlanner, self).__init__()

        self.goal_selector = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_subgoals)  # Output logits for subgoals
        )

    def forward(self, latent_state):
        """
        :param latent_state: [B, latent_dim]
        :return: subgoal logits [B, num_subgoals]
        """
        return self.goal_selector(latent_state)

    def select_subgoal(self, latent_state):
        """
        Returns index of chosen subgoal (greedy for simplicity)
        """
        logits = self.forward(latent_state)
        probs = F.softmax(logits, dim=-1)
        return torch.argmax(probs, dim=-1)
