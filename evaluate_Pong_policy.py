import gymnasium as gym
import ale_py
import numpy as np
import ale_py
from pathlib import Path
import numpy as np

from dotenv import load_dotenv
from autogen import config_list_from_json
import gymnasium as gym
import opto.trace as trace
from opto.trace import bundle, node, Module, GRAPH
from opto.optimizers import OptoPrime
from opto.trace.bundle import ExceptionNode
from opto.trace.errors import ExecutionError
from ocatari.core import OCAtari

load_dotenv()
gym.register_envs(ale_py)

class PongOCAtariTracedEnv:
    def __init__(self, 
                 env_name="ALE/Pong-v5",
                 render_mode="human",
                 obs_mode="obj",
                 hud=False):
        self.env_name = env_name
        self.render_mode = render_mode
        self.obs_mode = obs_mode
        self.hud = hud
        self.env = None
        self.init()
    
    def init(self):
        if self.env is not None:
            self.close()
        self.env = OCAtari(self.env_name, 
                           render_mode=self.render_mode, 
                           obs_mode=self.obs_mode, 
                           hud=self.hud)
        self.obs, _ = self.env.reset()
    
    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
            self.obs = None
    
    def __del__(self):
        self.close()

    def extract_obj_state(self, objects):
        obs = dict()
        for object in objects:
            obs[object.category] = {"x": object.x,
                                    "y": object.y,
                                    "w": object.w,
                                    "h": object.h,
                                    "dx": object.dx,
                                    "dy": object.dy,}
        return obs


    @bundle()
    def reset(self):
        """
        Reset the environment and return the initial observation and info.
        """
        _, info = self.env.reset()
        self.obs = self.extract_obj_state(self.env.objects)
        self.obs['reward'] = np.nan

        return self.obs, info
    
    def step(self, action):
        try:
            control = action.data if isinstance(action, trace.Node) else action
            next_obs, reward, termination, truncation, info = self.env.step(control)
            self.obs = self.extract_obj_state(self.env.objects)
            self.obs['reward'] = reward
        except Exception as e:
            e_node = ExceptionNode(
                e,
                inputs={"action": action},
                description="[exception] The operator step raises an exception.",
                name="exception_step",
            )
            raise ExecutionError(e_node)
        @bundle()
        def step(action):
            """
            Take action in the environment and return the next observation
            """
            return self.obs

        self.obs = step(action)
        return self.obs, reward, termination, truncation, info

@trace.model
class Policy(Module):
    def init(self):
        pass

    def __call__(self, obs):
        predicted_ball_y = self.predict_ball_trajectory(obs)
        action = self.select_action(predicted_ball_y, obs)
        return action

    @bundle(trainable=True)
    def predict_ball_trajectory(self, obs):
        """
        Given current observation, compute the velocity of the ball and predict the y coordinate of the ball when it reaches the player's paddle. 
        
        Args:
            obs (dict): Dictionary of current game state, mapping keys ("Player", "Ball", "Enemy") to values (dictionary of keys ('x', 'y', 'w', 'h', 'dx', 'dy') to integer values)
            
        Returns:
            float: The predicted y-coordinate where the ball will reach the player's side
        """
        if 'Ball' in obs:
            ball_y = obs['Ball'].get('y')
            ball_dy = obs['Ball'].get('dy', 0)
            ball_x = obs['Ball'].get('x', 0)
            player_x = obs['Player'].get('x', 0)
            ball_dx = obs['Ball'].get('dx', 0)

            if ball_y is not None and ball_dy is not None:
                if ball_dx == 0:
                    return float(ball_y)
                frames_to_reach = (player_x - ball_x) / max(1, abs(ball_dx))
                predicted_y = ball_y + ball_dy * frames_to_reach

                # Handle bouncing more robustly
                while predicted_y < 0 or predicted_y > 200:
                    if predicted_y < 0:
                        predicted_y *= -1
                    elif predicted_y > 200:
                        predicted_y = 400 - predicted_y

                return float(predicted_y)
        return None
        
    @bundle(trainable=True)
    def select_action(self, predicted_ball_y, obs):
        '''
        Select the optimal action of the player paddle based on current observation and predicted ball y coordinate.

        Args:
            predicted_ball_y (float): predicted y coordinate of the ball or None
            obs(dict): Dictionary of current game state, mapping keys ("Player", "Ball", "Enemy") to values (dictionary of keys ('x', 'y', 'w', 'h', 'dx', 'dy') to integer values)
        Returns:
            int: 0 for NOOP, 2 for DOWN, 3 for UP
        '''

        if predicted_ball_y is not None and "Player" in obs:
            player_y = obs["Player"].get("y", None)
            player_height = obs["Player"].get("h", None)

            if player_y is not None and player_height is not None:
                player_center = player_y + player_height / 2
                
                # Increase tolerance for more stable movement
                center_tolerance = 10  # Central deadzone
                edge_tolerance = 5     # Additional tolerance near paddle edges
                
                # Calculate distance to predicted position
                distance = predicted_ball_y - player_center
                
                # Add extra tolerance near the edges of the paddle
                if abs(distance) <= center_tolerance:
                    return 0  # Stay still if ball is roughly centered
                elif abs(distance) <= player_height/2 + edge_tolerance:
                    # If we're close to intercepting with the paddle edge, don't move
                    return 0
                elif distance > 0:
                    return 3  # UP
                else:
                    return 2  # DOWN
        return 0

def test_policy(env, policy, num_episodes=10, steps_per_episode=4000):
    rewards = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        
        for _ in range(steps_per_episode):
            action = policy(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        print(f"Episode {episode + 1} reward: {episode_reward}")
        rewards.append(episode_reward)
    
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    print(f"\nResults over {num_episodes} episodes:")
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    return mean_reward, std_reward

if __name__ == "__main__":
    env = PongOCAtariTracedEnv(render_mode=None) # No rendering
    policy = Policy()
    try:
        test_policy(env, policy)
    finally:
        env.close()
