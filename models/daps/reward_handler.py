# models/daps/reward_handler.py

class RewardHandler:
    """
    Handles modular reward shaping, including injected knowledge and potential-based shaping.
    """

    def __init__(self, use_shaping=True, shaping_weight=0.1):
        self.use_shaping = use_shaping
        self.shaping_weight = shaping_weight

    def compute_reward(self, env_reward, info_dict):
        """
        Computes final reward signal.

        :param env_reward: scalar from environment
        :param info_dict: dict with optional knowledge-based signals
        :return: final_reward (float)
        """
        reward = env_reward
        if self.use_shaping:
            shaping = info_dict.get("shaping_bonus", 0.0)
            reward += self.shaping_weight * shaping
        return reward
