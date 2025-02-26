import gymnasium as gym
import numpy as np
import cv2
from gymnasium import spaces
import ale_py

gym.register_envs(ale_py)

class PongEnv(gym.Wrapper):
    def __init__(self, render_mode=None):
        env = gym.make("ALE/Pong-v5", render_mode=render_mode, obs_type="grayscale")
        super().__init__(env)
        
        # Define observation space (grayscale image)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(80, 80, 1), dtype=np.uint8
        )
        
        # Define action space (NOOP, FIRE, RIGHT, LEFT)
        # Map to actual Pong actions: NOOP(0), FIRE(1), RIGHT(2), LEFT(3)
        self.action_map = {0: 0, 1: 1, 2: 2, 3: 3}
        self.action_space = spaces.Discrete(4)
        
    def process_observation(self, obs):
        # Crop and resize observation
        obs = obs[34:194, 15:147]  # Crop scores and borders
        obs = cv2.resize(obs, (80, 80))  # Resize to 80x80
        return np.expand_dims(obs, axis=-1)  # Add channel dimension
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.process_observation(obs), info
        
    def step(self, action):
        # Map our action to actual Pong action
        if isinstance(action, (list, np.ndarray)):
            action = action.item()  # Convert to scalar if array
        action = self.action_map[action]
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.process_observation(obs), reward, terminated, truncated, info
