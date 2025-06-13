# environments/wrappers.py

import gym
import numpy as np

class FlattenedObsWrapper(gym.ObservationWrapper):
    """
    Flattens Dict observation into a single 1D vector for baseline compatibility.
    """

    def __init__(self, env):
        super(FlattenedObsWrapper, self).__init__(env)
        flat_dim = np.prod(env.observation_space["image"].shape) + \
                   np.prod(env.observation_space["audio"].shape) + \
                   np.prod(env.observation_space["symbol"].shape)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32)

    def observation(self, obs):
        image = obs["image"].flatten() / 255.0
        audio = obs["audio"].flatten()
        symbol = obs["symbol"].flatten()
        return np.concatenate([image, audio, symbol])
