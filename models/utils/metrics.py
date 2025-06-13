# models/utils/metrics.py

import numpy as np

class EvaluationMetrics:
    """
    A utility class to collect and compute evaluation metrics for SLTN + DAPS agents.
    Supports average reward, success rate, temporal consistency, and generalization score.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets all tracked metrics."""
        self.total_episodes = 0
        self.successful_episodes = 0
        self.total_reward = 0.0
        self.consistency_scores = []      # e.g., action variance / state difference
        self.generalization_scores = []   # e.g., performance on unseen tasks

    def update(self, reward, success, consistency_score=None, generalization_score=None):
        """
        Updates metrics after an episode.
        :param reward: cumulative reward of the episode
        :param success: boolean indicating if goal was achieved
        :param consistency_score: temporal consistency (e.g., action stability)
        :param generalization_score: transferability to novel conditions
        """
        self.total_episodes += 1
        self.total_reward += reward
        if success:
            self.successful_episodes += 1
        if consistency_score is not None:
            self.consistency_scores.append(consistency_score)
        if generalization_score is not None:
            self.generalization_scores.append(generalization_score)

    def compute(self):
        """
        Computes aggregated statistics.
        :return: dictionary of computed metrics
        """
        avg_reward = self.total_reward / max(self.total_episodes, 1)
        success_rate = self.successful_episodes / max(self.total_episodes, 1)
        consistency = np.mean(self.consistency_scores) if self.consistency_scores else 0.0
        generalization = np.mean(self.generalization_scores) if self.generalization_scores else 0.0

        return {
            "avg_reward": avg_reward,
            "success_rate": success_rate,
            "temporal_consistency": consistency,
            "generalization_score": generalization
        }

    def print_summary(self):
        """Prints formatted metric summary to console."""
        metrics = self.compute()
        print("===== Evaluation Summary =====")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
