# environments/custom_envs.py

import gym
from gym import spaces
import numpy as np

class GridWorldEnv(gym.Env):
    """
    A simple custom grid-based environment with dynamic rules.
    This is a placeholder template. Can be extended with symbolic/visual/audio cues.
    """

    def __init__(self, grid_size=5):
        super(GridWorldEnv, self).__init__()
        self.grid_size = grid_size
        self.state = None
        self.goal_pos = (grid_size - 1, grid_size - 1)

        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(3, 64, 64), dtype=np.uint8),
            "audio": spaces.Box(low=-1.0, high=1.0, shape=(128,), dtype=np.float32),
            "symbol": spaces.Box(low=0, high=1, shape=(16,), dtype=np.float32),
        })
        self.action_space = spaces.Discrete(4)  # up, down, left, right

    def reset(self):
        self.agent_pos = [0, 0]
        return self._get_obs()

    def step(self, action):
        if action == 0:  # up
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 1:  # down
            self.agent_pos[1] = min(self.grid_size - 1, self.agent_pos[1] + 1)
        elif action == 2:  # left
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 3:  # right
            self.agent_pos[0] = min(self.grid_size - 1, self.agent_pos[0] + 1)

        done = tuple(self.agent_pos) == self.goal_pos
        reward = 1.0 if done else -0.01

        info = {"shaping_bonus": 1.0 / (np.linalg.norm(np.array(self.agent_pos) - np.array(self.goal_pos)) + 1)}

        return self._get_obs(), reward, done, info

    def _get_obs(self):
        # Placeholder multimodal observation
        return {
            "image": np.random.randint(0, 256, (3, 64, 64), dtype=np.uint8),
            "audio": np.random.uniform(-1.0, 1.0, (128,)).astype(np.float32),
            "symbol": np.random.rand(16).astype(np.float32),
        }
