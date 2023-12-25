from envs.mlg_wb_specs import MLGWB
from envs.standard_navigation_env import Navigation
import gym
import numpy as np


class EnvsManager:
    """
    Provides a common environment manager that makes it easy to create and manage different MineRL & Gym environments.
    """
    envs = {
        "MineRLNavigate": "MineRLNavigate-v0",
        "MineRLNavigateDense": "MineRLNavigateDense-v0",
        "navigation": "navigation-v0",
    }

    human_envs = {"navigation": "MineRLNavigateDense-v0"}
    block_range = 64
    reward_type = 1
    reset_threshold = 2

    @staticmethod
    def register(dense=False, extreme=False, radius=8, block_range=64, reward_type=1, reset_threshold=2.):
        """Register the environments. Default are the MLGWB and Navigation environments."""
        np.float = float
        np.int = int
        EnvsManager.block_range = block_range
        EnvsManager.reward_type = reward_type
        EnvsManager.reset_threshold = reset_threshold

        abs_MLG = MLGWB(
            dense=dense, extreme=extreme, block_range=block_range, compass_radius=radius
        )
        abs_MLG.register()

        nav = Navigation(
            dense=dense, extreme=extreme, block_range=block_range, compass_radius=radius
        )
        nav.register()

    @staticmethod
    def create_env_by_name(name=""):
        """Create an environment by name."""
        env = gym.make(EnvsManager.envs.get(name))
        env.block_range = EnvsManager.block_range
        env.reward_type = EnvsManager.reward_type
        env.reset_threshold = EnvsManager.reset_threshold
        # gym.reset()

        return env
