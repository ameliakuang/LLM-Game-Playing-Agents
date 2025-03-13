import os
import ale_py
import logging
import datetime
from pathlib import Path
import numpy as np
import io
import contextlib
import cv2
import pandas as pd
import warnings
import argparse
import sys
import random
import imageio.v2 as imageio

# os.environ['TRACE_CUSTOMLLM_MODEL'] = "anthropic.claude-3-5-haiku-20241022-v1:0"
os.environ['TRACE_CUSTOMLLM_MODEL'] = "anthropic.claude-3-5-sonnet-20240620-v1:0"

from dotenv import load_dotenv
load_dotenv()

import gymnasium as gym
import opto.trace as trace
from opto.trace import bundle, node, Module, GRAPH
from opto.optimizers import OptoPrime
from opto.trace.bundle import ExceptionNode
from opto.trace.errors import ExecutionError
from ocatari.core import OCAtari

gym.register_envs(ale_py)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
base_trace_ckpt_dir = Path("trace_ckpt")
base_trace_ckpt_dir.mkdir(exist_ok=True)



class SpaceInvadersOCAtariTracedEnv:
    def __init__(self, 
                 env_name="SpaceInvadersNoFrameskip-v4",
                #  render_mode="human",
                 render_mode=None,
                 obs_mode="obj",
                 hud=False,
                 frameskip=4,
                 repeat_action_probability=0.0):
        self.env_name = env_name
        self.render_mode = render_mode
        self.obs_mode = obs_mode
        self.hud = hud
        self.frameskip = frameskip
        self.repeat_action_probability = repeat_action_probability
        self.env = None
        self.init()
    
    def init(self):
        if self.env is not None:
            self.close()
        self.env = OCAtari(self.env_name, 
                           render_mode=self.render_mode, 
                           obs_mode=self.obs_mode, 
                           hud=self.hud,
                           frameskip=self.frameskip,
                           repeat_action_probability=self.repeat_action_probability)
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
        # Count objects by category to create unique keys
        category_counts = {}
        
        for object in objects:
            category = object.category

            if category == "NoObject":
                continue

            # For aliens and other objects that might have multiple instances
            if category in ["Alien", "Shield", "Bullet"]:
                if category not in category_counts:
                    category_counts[category] = 0
                else:
                    category_counts[category] += 1
                
                # Create indexed key for multiple objects of same category
                key = f"{category}{category_counts[category]}"
            else:
                key = category
                
            obs[key] = {"x": object.x,
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
        """
        Step the environment with the given action.
        """
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
        shoot_decision = self.decide_shoot(obs)
        move_decision = self.decide_movement(obs)
        return self.combine_actions(shoot_decision, move_decision)

    @bundle(trainable=True)
    def decide_shoot(self, obs):
        '''
        Decide whether to shoot based on enemy positions and existing projectiles.
         
        Args:
            obs (dict): Game state observation containing object states for "Player", "Shield0", "Shield1", "Alien0", "Alien1", etc.
            Each object has position (x,y), size (w,h), and velocity (dx,dy).
            Player bullets have negative dy velocity and alien bullets have positive dy velocity
        
        Strategy tips:
        - You can only have one missile at a time
        - Try to shoot when aliens are aligned with your ship
        - Prioritize shooting at lower aliens as they're closer to you
        - Consider the movement of aliens when deciding to shoot
        
        Returns:
            bool: True if should shoot, False otherwise
        '''
        
        # There can only be one player bullet on the field at a time
        # Check for player bullets (which have negative dy velocity)
        for key, obj in obs.items():
            if key.startswith('Bullet') and obj.get('dy', 0) < 0:
                return False
            
        player = obs['Player']
        for key, obj in obs.items():
            if key.startswith('Alien'):
                # Check if alien is aligned with player (within 5 pixels)
                if abs(obj['x'] - player['x']) < 5:
                    # Prioritize lower aliens (higher y value)
                    if obj['y'] > 60:  # Adjust this threshold as needed
                        return True
        return False            

    @bundle(trainable=True)
    def decide_movement(self, obs):
        '''
        Decide movement direction based on enemy positions and projectiles.
         
        Args:
            obs (dict): Game state observation containing object states for "Player", "Shield0", "Shield1", "Alien0", "Alien1", etc.
            Each object has position (x,y), size (w,h), and velocity (dx,dy).
            Player bullets have negative dy velocity and alien bullets have positive dy velocity
        
        Strategy tips:
        - Move to dodge enemy projectiles
        - Position yourself under aliens to shoot them
        - Stay away from the edges of the screen
        - Consider moving toward areas with more aliens to increase score
        
        Returns:
            int: -1 for left, 1 for right, 0 for no movement
        '''

        player = obs['Player']
        move = 0
        threat_left = 0
        threat_right = 0
        aliens_left = 0
        aliens_right = 0
        
        for key, obj in obs.items():
            if key.startswith('Alien'):
                if obj['x'] < player['x']:
                    aliens_left += 1
                else:
                    aliens_right += 1
            elif key.startswith('Bullet') and obj['dy'] > 0:  # Enemy bullet
                if obj['x'] < player['x']:
                    threat_left += 1
                else:
                    threat_right += 1
        
        # Move away from threats
        if threat_left > threat_right:
            move = 1
        elif threat_right > threat_left:
            move = -1
        # If no immediate threat, move towards more aliens
        elif aliens_left > aliens_right:
            move = -1
        elif aliens_right > aliens_left:
            move = 1
        
        return move    

    @bundle(trainable=True)
    def combine_actions(self, shoot, movement):
        '''
        Combine shooting and movement decisions into final action.
        
        Args:
            shoot (bool): Whether to shoot
            movement (int): Movement direction
        
        Action mapping:
        - 0: NOOP (no operation)
        - 1: FIRE (shoot without moving)
        - 2: RIGHT (move right without shooting)
        - 3: LEFT (move left without shooting)
        - 4: RIGHT+FIRE (move right while shooting)
        - 5: LEFT+FIRE (move left while shooting)
        
        Returns:
            int: Final action (0: NOOP, 1: FIRE, 2: RIGHT, 3: LEFT, 4: RIGHT+FIRE, 5: LEFT+FIRE)
        '''
        
        if shoot and movement > 0:
            return 4  # RIGHT+FIRE
        elif shoot and movement < 0:
            return 5  # LEFT+FIRE
        elif shoot:
            return 1  # FIRE
        elif movement > 0:
            return 2  # RIGHT
        elif movement < 0:
            return 3  # LEFT
        return 0  # NOOP

def display_terminal_debug(obs, step_num=None):
    """
    Display debug information directly in the terminal.
    
    Args:
        obs (dict): Game state observation
        step_num (int, optional): Current step number
    """
    # Get debug info as string
    debug_info = print_debug_info(obs, step_num)
    
    # Print to terminal with a separator for visibility
    print("\n" + "="*50)
    print("DEBUG INFORMATION")
    print("="*50)
    print(debug_info)
    print("="*50 + "\n")

def rollout(env, horizon, policy, visualize=False, debug=False, vis_dir=None, terminal_debug=False, vis_frequency=20, create_gif=True, gif_fps=10):
    """Rollout a policy in an env for horizon steps."""
    try:
        obs, _ = env.reset()
        trajectory = dict(observations=[], actions=[], rewards=[], terminations=[], truncations=[], infos=[], steps=0)
        trajectory["observations"].append(obs)
        
        # Create visualization directory if needed
        if visualize and vis_dir:
            os.makedirs(vis_dir, exist_ok=True)
            
            # Create debug log file
            if debug:
                with open(os.path.join(vis_dir, "debug_log.txt"), "w") as f:
                    f.write("Debug Log\n")
                    f.write("=========\n\n")
        
        # For GIF creation
        frames = []
        
        for step in range(horizon):
            error = None
            try:
                # Display debug info in terminal if requested
                if terminal_debug and step % 10 == 0:  # Only show every 10 steps to avoid clutter
                    display_terminal_debug(obs, step)
                
                # Visualize current state if requested
                if visualize:
                    try:
                        # Create visualization frame
                        frame = visualize_game_state(obs, step)
                        
                        if create_gif:
                            # Store frame for GIF
                            frames.append(frame)
                        
                        # Save individual frame if requested and at the right frequency
                        if vis_dir and not create_gif and (step % vis_frequency == 0 or step < 5 or step > horizon - 5):
                            vis_path = os.path.join(vis_dir, f"step_{step:04d}.png")
                            cv2.imwrite(vis_path, frame)
                    except Exception as e:
                        # Log visualization error but continue with rollout
                        if vis_dir:
                            with open(os.path.join(vis_dir, "errors.txt"), "a") as f:
                                f.write(f"Visualization error at step {step}: {str(e)}\n")
                
                # Print debug info if requested
                if debug and vis_dir:
                    try:
                        debug_info = print_debug_info(obs, step)
                        with open(os.path.join(vis_dir, "debug_log.txt"), "a") as f:
                            f.write(debug_info + "\n\n")
                    except Exception as e:
                        # Log debug error but continue with rollout
                        with open(os.path.join(vis_dir, "errors.txt"), "a") as f:
                            f.write(f"Debug error at step {step}: {str(e)}\n")
                
                action = policy(obs)
                next_obs, reward, termination, truncation, info = env.step(action)
            except trace.ExecutionError as e:
                error = e
                reward = np.nan
                termination = True
                truncation = False
                info = {}
            except Exception as e:
                # Create a custom error node for non-trace errors
                error = trace.ExecutionError(
                    ExceptionNode(
                        e,
                        inputs={"step": step},
                        description=f"[exception] Error during rollout at step {step}: {str(e)}",
                        name="exception_rollout",
                    )
                )
                reward = np.nan
                termination = True
                truncation = False
                info = {}
            
            if error is None:
                trajectory["observations"].append(next_obs)
                trajectory["actions"].append(action)
                trajectory["rewards"].append(reward)
                trajectory["terminations"].append(termination)
                trajectory["truncations"].append(truncation)
                trajectory["infos"].append(info)
                trajectory["steps"] += 1
                if termination or truncation:
                    break
                obs = next_obs
        
        # Create GIF if requested
        if visualize and create_gif and frames and vis_dir:
            try:
                # Save GIF
                gif_path = os.path.join(vis_dir, "animation.gif")
                print(f"Creating GIF with {len(frames)} frames...")
                imageio.mimsave(gif_path, frames, fps=gif_fps)
                print(f"GIF created: {gif_path}")
            except Exception as e:
                # Log GIF creation error
                with open(os.path.join(vis_dir, "errors.txt"), "a") as f:
                    f.write(f"GIF creation error: {str(e)}\n")
                    
    except Exception as e:
        # Handle any other exceptions during rollout
        error = trace.ExecutionError(
            ExceptionNode(
                e,
                inputs={},
                description=f"[exception] Error during rollout setup: {str(e)}",
                name="exception_rollout_setup",
            )
        )
        return None, error
    
    return trajectory, error

def test_policy(policy, 
                num_episodes=10, 
                steps_per_episode=4000,
                frameskip=1,
                repeat_action_probability=0.0,
                visualize=False,
                debug=False,
                vis_dir=None,
                terminal_debug=False,
                vis_frequency=20,
                create_gif=True,
                gif_fps=10):
    """
    Test a policy over multiple episodes and return the mean and standard deviation of rewards.
    
    Args:
        policy: The policy to test
        num_episodes: Number of episodes to run
        steps_per_episode: Maximum steps per episode
        frameskip: Number of frames to skip between actions
        repeat_action_probability: Probability of repeating the previous action
        visualize: Whether to visualize the first evaluation episode
        debug: Whether to save debug information
        vis_dir: Directory to save visualization files
        terminal_debug: Whether to display debug info in terminal
        vis_frequency: Save visualization every N steps
        create_gif: Whether to create a GIF instead of individual frames
        gif_fps: Frames per second for GIF
        
    Returns:
        tuple: (mean_reward, std_reward)
    """
    print(f"  Testing policy over {num_episodes} episodes...", end="", flush=True)
    
    env = None
    rewards = []
    
    try:
        env = SpaceInvadersOCAtariTracedEnv(render_mode=None,
                                           frameskip=frameskip,
                                           repeat_action_probability=repeat_action_probability)
        
        for episode in range(num_episodes):
            episode_reward = 0
            frames = [] if visualize and episode == 0 and create_gif else None
            
            try:
                obs, _ = env.reset()
                
                for step in range(steps_per_episode):
                    try:
                        # Visualize the first episode if requested
                        if visualize and episode == 0:
                            # Display debug info in terminal if requested
                            if terminal_debug and step % 10 == 0:
                                display_terminal_debug(obs, step)
                            
                            # Create visualization frame
                            if create_gif or (step % vis_frequency == 0 or step < 5 or step > steps_per_episode - 5):
                                frame = visualize_game_state(obs, step)
                                
                                if create_gif:
                                    # Store frame for GIF
                                    frames.append(frame)
                                
                                # Save individual frame if requested and at the right frequency
                                if vis_dir and not create_gif and (step % vis_frequency == 0 or step < 5):
                                    eval_vis_path = os.path.join(vis_dir, f"eval_step_{step:04d}.png")
                                    cv2.imwrite(eval_vis_path, frame)
                            
                            # Print debug info if requested
                            if debug and vis_dir:
                                try:
                                    debug_info = print_debug_info(obs, step)
                                    with open(os.path.join(vis_dir, "eval_debug_log.txt"), "a") as f:
                                        f.write(debug_info + "\n\n")
                                except Exception as e:
                                    # Log debug error but continue with evaluation
                                    with open(os.path.join(vis_dir, "eval_errors.txt"), "a") as f:
                                        f.write(f"Debug error at step {step}: {str(e)}\n")
                        
                        action = policy(obs)
                        obs, reward, terminated, truncated, _ = env.step(action)
                        episode_reward += reward
                        
                        if terminated or truncated:
                            break
                    except Exception as e:
                        # Log error but continue with next episode
                        logging.warning(f"Error during test episode {episode} step: {str(e)}")
                        break
                
                # Create GIF for the first episode if requested
                if visualize and episode == 0 and create_gif and frames and vis_dir:
                    try:
                        # Save GIF
                        gif_path = os.path.join(vis_dir, "eval_animation.gif")
                        print(f"\n  Creating evaluation GIF with {len(frames)} frames...")
                        imageio.mimsave(gif_path, frames, fps=gif_fps)
                        print(f"  Evaluation GIF created: {gif_path}")
                    except Exception as e:
                        # Log GIF creation error
                        if vis_dir:
                            with open(os.path.join(vis_dir, "eval_errors.txt"), "a") as f:
                                f.write(f"GIF creation error: {str(e)}\n")
                
                rewards.append(episode_reward)
            except Exception as e:
                # Log error but continue with next episode
                logging.warning(f"Error during test episode {episode}: {str(e)}")
                continue
    except Exception as e:
        logging.error(f"Error during policy testing: {str(e)}")
        # Return default values if testing fails completely
        if not rewards and env is not None:
            # Try to get a single episode reward as fallback
            try:
                obs, _ = env.reset()
                episode_reward = 0
                for _ in range(100):  # Just try a few steps
                    action = policy(obs)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    episode_reward += reward
                    if terminated or truncated:
                        break
                rewards = [episode_reward]
            except:
                rewards = [0.0]  # Last resort
    finally:
        if env is not None:
            try:
                env.close()
            except:
                pass  # Ignore errors during environment closing
    
    # Calculate statistics
    if rewards:
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
    else:
        mean_reward = 0.0
        std_reward = 0.0
    
    print(f" done. Mean: {mean_reward:.1f}, StdDev: {std_reward:.1f}")
    return mean_reward, std_reward

def visualize_game_state(obs, step_num=None, save_path=None):
    """
    Visualize the game state from object observations.
    
    Args:
        obs (dict): Game state observation
        step_num (int, optional): Current step number for labeling
        save_path (str, optional): Path to save visualization image
    """
    # Create a blank canvas (black background)
    canvas = np.zeros((210, 160, 3), dtype=np.uint8)
    
    # Draw objects
    for key, obj in obs.items():
        # Skip reward or non-dict objects
        if key == 'reward' or not isinstance(obj, dict) and not hasattr(obj, 'data'):
            continue
        
        # Handle MessageNode objects for keys
        key_str = str(key.data) if hasattr(key, 'data') else str(key)
        
        # Handle MessageNode objects for values
        if hasattr(obj, 'data'):
            obj_data = obj.data
        else:
            obj_data = obj
            
        # Skip if obj_data is not a dict
        if not isinstance(obj_data, dict):
            continue
            
        # Extract coordinates safely
        try:
            x = float(obj_data.get('x', 0))
            y = float(obj_data.get('y', 0))
            w = float(obj_data.get('w', 5))
            h = float(obj_data.get('h', 5))
            dx = float(obj_data.get('dx', 0))
            dy = float(obj_data.get('dy', 0))
        except (ValueError, TypeError):
            continue
        
        # Different colors for different object types
        if key_str == 'Player':
            color = (0, 255, 0)  # Green for player
        elif key_str.startswith('Alien'):
            color = (255, 0, 0)  # Red for aliens
        elif key_str.startswith('Bullet'):
            # Determine if it's a player bullet (moving up) or enemy bullet (moving down)
            if dy < 0:  # Moving up (player bullet)
                color = (0, 255, 255)  # Cyan for player bullets
            else:  # Moving down or stationary (enemy bullet)
                color = (255, 255, 0)  # Yellow for enemy bullets
        elif key_str.startswith('Missile'):
            color = (0, 255, 255)  # Cyan for player missiles
        elif key_str.startswith('EnemyMissile'):
            color = (255, 255, 0)  # Yellow for enemy missiles
        elif key_str.startswith('Shield'):
            color = (0, 0, 255)  # Blue for shields
        else:
            color = (255, 255, 255)  # White for other objects
            
        # Draw rectangle for the object
        cv2.rectangle(canvas, (int(x), int(y)), (int(x+w), int(y+h)), color, 1)
        
        # Add label
        cv2.putText(canvas, key_str, (int(x), int(y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    
    # Add step number if provided
    if step_num is not None:
        cv2.putText(canvas, f"Step: {step_num}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Add reward if available
    if 'reward' in obs:
        reward = obs['reward']
        # Handle MessageNode for reward
        if hasattr(reward, 'data'):
            reward = reward.data
            
        # Fix the isnan check to handle non-numeric types
        try:
            if isinstance(reward, (int, float)) and not np.isnan(reward):
                cv2.putText(canvas, f"Reward: {reward}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        except:
            pass  # Skip if reward can't be displayed
    
    # Save or display
    if save_path:
        cv2.imwrite(save_path, canvas)
    
    return canvas

def print_debug_info(obs, step_num=None):
    """
    Print debug information about the game state.
    
    Args:
        obs (dict): Game state observation
        step_num (int, optional): Current step number
    """
    # Count objects by type
    object_counts = {}
    for key in obs.keys():
        if key == 'reward':
            continue
            
        # Handle MessageNode objects or other trace objects
        if hasattr(key, 'data'):
            key_str = str(key.data)
        else:
            key_str = str(key)
            
        # Extract the base object type (remove digits)
        obj_type = ''.join([c for c in key_str if not (c.isdigit() if hasattr(c, 'isdigit') else False)])
        
        if obj_type not in object_counts:
            object_counts[obj_type] = 1
        else:
            object_counts[obj_type] += 1
    
    # Format debug string
    debug_str = []
    if step_num is not None:
        debug_str.append(f"Step {step_num}:")
    
    for obj_type, count in object_counts.items():
        debug_str.append(f"{obj_type}: {count}")
    
    # Add alien positions if any
    aliens = []
    for key, obj in obs.items():
        # Handle MessageNode objects
        key_str = str(key.data) if hasattr(key, 'data') else str(key)
        
        if key_str.startswith('Alien'):
            aliens.append((key_str, obj))

    shields = []
    for key, obj in obs.items():
        # Handle MessageNode objects
        key_str = str(key.data) if hasattr(key, 'data') else str(key)
        
        if key_str.startswith('Shield'):
            shields.append((key_str, obj))
    
    if aliens:
        debug_str.append("Alien positions:")
        for key, alien in aliens:
            # Handle MessageNode objects in the alien data
            if hasattr(alien, 'data'):
                alien_data = alien.data
            else:
                alien_data = alien
                
            # Extract x and y coordinates safely
            x = alien_data.get('x', 'unknown') if isinstance(alien_data, dict) else 'unknown'
            y = alien_data.get('y', 'unknown') if isinstance(alien_data, dict) else 'unknown'
            
            debug_str.append(f"  {key}: x={x}, y={y}")

    if shields:
        debug_str.append("Shield positions:")
        for key, shield in shields:
            # Handle MessageNode objects in the shield data
            if hasattr(shield, 'data'):
                shield_data = shield.data
            else:
                shield_data = shield
                
            # Extract x and y coordinates safely
            x = shield_data.get('x', 'unknown') if isinstance(shield_data, dict) else 'unknown'
            y = shield_data.get('y', 'unknown') if isinstance(shield_data, dict) else 'unknown'
            
            debug_str.append(f"  {key}: x={x}, y={y}")
    
    return "\n".join(debug_str)

def optimize_policy(
    env_name="SpaceInvadersNoFrameskip-v4",
    horizon=400,
    memory_size=5,
    n_optimization_steps=10,
    verbose=False,
    frame_skip=4,
    sticky_action_p=0.00,
    logger=None,
    visualize=True,  # Enable visualization by default
    debug=True,      # Enable debug info by default
    terminal_debug=False,  # Display debug info in terminal
    vis_frequency=20,  # Save visualization every N steps
    create_gif=True,  # Create GIF instead of individual PNG files
    gif_fps=10,  # Frames per second for GIF
    enable_rollback=False,  # Enable policy rollback on error (default: False)
):
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        # Use file handler instead of stream handler to reduce terminal clutter
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_dir / f"{env_name.replace('/', '_')}_{timestamp}_skip{frame_skip}_sticky{sticky_action_p}_horizon{horizon}_optimSteps{n_optimization_steps}_mem{memory_size}_rollback{enable_rollback}.log")
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Remove any existing handlers to avoid duplicate output
        for handler in logger.handlers[:]:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                logger.removeHandler(handler)
    
    # Create directories for visualization and debug info
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path("results")
    base_dir.mkdir(exist_ok=True)
    
    vis_base_dir = base_dir / "visualizations"
    vis_base_dir.mkdir(exist_ok=True)
    
    # Create animations directory if needed
    if create_gif:
        animations_dir = base_dir / "animations"
        animations_dir.mkdir(exist_ok=True)
    
    log_dir = base_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    
    base_trace_ckpt_dir = base_dir / "checkpoints"
    base_trace_ckpt_dir.mkdir(exist_ok=True)
    
    policy = Policy()
    optimizer = OptoPrime(policy.parameters(), memory_size=memory_size)
    env = None
    
    # Print a clean header to the console
    print("\n" + "="*50)
    print(f"Space Invaders AI Training - {timestamp}")
    print("="*50)
    print(f"Environment: {env_name}")
    print(f"Horizon: {horizon} steps")
    print(f"Optimization steps: {n_optimization_steps}")
    print(f"Frame skip: {frame_skip}")
    print(f"Sticky action probability: {sticky_action_p}")
    print(f"Visualization: {'Enabled' if visualize else 'Disabled'}")
    print(f"Debug info: {'Enabled' if debug else 'Disabled'}")
    print(f"Terminal debug: {'Enabled' if terminal_debug else 'Disabled'}")
    print(f"Create GIF: {'Enabled' if create_gif else 'Disabled'}")
    if create_gif:
        print(f"GIF FPS: {gif_fps}")
    else:
        print(f"Visualization frequency: Every {vis_frequency} steps")
    print(f"Policy rollback: {'Enabled' if enable_rollback else 'Disabled'}")
    print("="*50 + "\n")
    
    perf_csv_filename = log_dir / f"perf_{env_name.replace('/', '_')}_{timestamp}_skip{frame_skip}_sticky{sticky_action_p}_horizon{horizon}_optimSteps{n_optimization_steps}_mem{memory_size}_rollback{enable_rollback}.csv"
    trace_ckpt_dir = base_trace_ckpt_dir / f"{env_name.replace('/', '_')}_{timestamp}_skip{frame_skip}_sticky{sticky_action_p}_horizon{horizon}_optimSteps{n_optimization_steps}_mem{memory_size}_rollback{enable_rollback}"
    trace_ckpt_dir.mkdir(exist_ok=True)
    
    # Create visualization directory for this run
    vis_run_dir = vis_base_dir / f"{env_name.replace('/', '_')}_{timestamp}"
    if visualize:
        vis_run_dir.mkdir(exist_ok=True)
    
    try:
        rewards = []
        optimization_data = []
        logger.info("Optimization Starts")
        print("Starting optimization...")
        
        # Initialize environment
        env = SpaceInvadersOCAtariTracedEnv(env_name=env_name,
                                           render_mode="human" if visualize else None,
                                           frameskip=frame_skip,
                                           repeat_action_probability=sticky_action_p)
        
        for i in range(n_optimization_steps):
            print(f"\nIteration {i+1}/{n_optimization_steps}:")
            
            # Maximum number of retry attempts for this iteration
            max_retries = 3
            retry_count = 0
            rollout_success = False
            
            while not rollout_success and retry_count < max_retries:
                try:
                    env.init()
                    
                    # Create visualization directory for this iteration
                    if visualize:
                        iter_vis_dir = vis_run_dir / f"iteration_{i}"
                        if retry_count > 0:
                            iter_vis_dir = vis_run_dir / f"iteration_{i}_retry_{retry_count}"
                        iter_vis_dir.mkdir(exist_ok=True)
                    else:
                        iter_vis_dir = None
                    
                    # Run rollout with visualization if enabled
                    traj, error = rollout(env, horizon, policy, 
                                         visualize=False,  # Disable visualization for training rollout
                                         debug=debug,
                                         vis_dir=iter_vis_dir,
                                         terminal_debug=terminal_debug,
                                         vis_frequency=vis_frequency,
                                         create_gif=False,  # Disable GIF creation for training rollout
                                         gif_fps=gif_fps)

                    if error is None:
                        rollout_success = True
                        episode_score = sum(traj['rewards'])
                        feedback = f"Episode ends after {traj['steps']} steps with total score: {episode_score:.1f}"
                        
                        # Test policy performance
                        try:
                            mean_rewards, std_rewards = test_policy(policy,
                                                                frameskip=frame_skip,
                                                                repeat_action_probability=sticky_action_p,
                                                                visualize=visualize,
                                                                debug=debug,
                                                                vis_dir=iter_vis_dir,
                                                                terminal_debug=terminal_debug,
                                                                vis_frequency=vis_frequency,
                                                                create_gif=create_gif,
                                                                gif_fps=gif_fps)
                        except Exception as e:
                            logger.error(f"Error during policy testing: {e}")
                            mean_rewards = episode_score
                            std_rewards = 0.0
                        
                        # Provide feedback based on performance
                        if mean_rewards >= 2000:
                            logger.info(f"Excellent performance! Average score: {mean_rewards} with std dev {std_rewards}. Ending optimization early.")
                            print(f"Excellent performance! Average score: {mean_rewards:.1f}")
                            rewards.append(episode_score)
                            optimization_data.append({
                                "Optimization Step": i,
                                "Mean Reward": mean_rewards,
                                "Std Dev Reward": std_rewards
                            })
                            df = pd.DataFrame(optimization_data)
                            df.to_csv(perf_csv_filename, index=False)
                            policy.save(os.path.join(trace_ckpt_dir, f"{i}.pkl"))
                            break
                            
                        if mean_rewards >= 300:
                            feedback += (f"\nGreat job! You're performing well with an average score of {mean_rewards} "
                                        f"with std dev {std_rewards}. Try to improve your shooting accuracy and dodging.")
                        elif mean_rewards >= 100:
                            feedback += (f"\nGood progress! Your average score is {mean_rewards} with std dev {std_rewards}. "
                                        f"Focus on better timing for shooting and avoiding enemy projectiles.")
                        else:
                            feedback += (f"\nYour average score is {mean_rewards} with std dev {std_rewards}. "
                                        f"Try to improve your strategy for shooting aliens and dodging projectiles.")
                        
                        target = traj['observations'][-1]
                        
                        rewards.append(episode_score)
                        optimization_data.append({
                            "Optimization Step": i,
                            "Mean Reward": mean_rewards,
                            "Std Dev Reward": std_rewards
                        })
                        df = pd.DataFrame(optimization_data)
                        df.to_csv(perf_csv_filename, index=False)
                        
                        # Print a clean summary to console
                        print(f"  Episode Score: {episode_score:.1f}")
                        print(f"  Average Test Score: {mean_rewards:.1f} (±{std_rewards:.1f})")
                        print(f"  Steps Completed: {traj['steps']}")
                        
                        # Save detailed info to log file
                        logger.info(f"Iteration: {i}, Feedback: {feedback}, target: {target}")
                    else:
                        feedback = error.exception_node.create_feedback()
                        target = error.exception_node
                        logger.info(f"Iteration: {i}, Error: {feedback}, target: {target}")
                        print(f"  Error occurred during rollout")
                        
                        # Try to roll back to previous iteration's policy if available
                        if enable_rollback and i > 0 and retry_count < max_retries - 1:
                            prev_policy_path = os.path.join(trace_ckpt_dir, f"{i-1}.pkl")
                            if os.path.exists(prev_policy_path):
                                print(f"  Attempting to roll back to previous iteration's policy (retry {retry_count+1}/{max_retries-1})...")
                                try:
                                    # Create a new policy instance and load the previous checkpoint
                                    policy = Policy()
                                    policy.load(prev_policy_path)
                                    # Recreate optimizer with the loaded policy
                                    optimizer = OptoPrime(policy.parameters(), memory_size=memory_size)
                                    logger.info(f"Successfully rolled back to policy from iteration {i-1}")
                                    print(f"  Successfully rolled back to policy from iteration {i-1}")
                                    retry_count += 1
                                    continue
                                except Exception as load_error:
                                    logger.error(f"Failed to load previous policy: {load_error}")
                                    print(f"  Failed to load previous policy: {str(load_error)[:100]}...")
                        
                        # If we reach here, either rollback failed or we've exhausted retries
                        rollout_success = True  # Mark as success to move on to next iteration
                
                except Exception as e:
                    logger.exception(f"Error during iteration {i}, retry {retry_count}: {e}")
                    print(f"  Error during iteration {i}, retry {retry_count}: {str(e)[:100]}...")
                    
                    # Try to roll back to previous iteration's policy if available
                    if enable_rollback and i > 0 and retry_count < max_retries - 1:
                        prev_policy_path = os.path.join(trace_ckpt_dir, f"{i-1}.pkl")
                        if os.path.exists(prev_policy_path):
                            print(f"  Attempting to roll back to previous iteration's policy (retry {retry_count+1}/{max_retries-1})...")
                            try:
                                # Create a new policy instance and load the previous checkpoint
                                policy = Policy()
                                policy.load(prev_policy_path)
                                # Recreate optimizer with the loaded policy
                                optimizer = OptoPrime(policy.parameters(), memory_size=memory_size)
                                logger.info(f"Successfully rolled back to policy from iteration {i-1}")
                                print(f"  Successfully rolled back to policy from iteration {i-1}")
                                retry_count += 1
                                continue
                            except Exception as load_error:
                                logger.error(f"Failed to load previous policy: {load_error}")
                                print(f"  Failed to load previous policy: {str(load_error)[:100]}...")
                    
                    retry_count += 1
                    if retry_count >= max_retries:
                        print(f"  Maximum retry attempts ({max_retries}) reached. Moving to next iteration.")
                        break
            
            # Save policy checkpoint (even if there was an error)
            try:
                policy.save(os.path.join(trace_ckpt_dir, f"{i}.pkl"))
            except Exception as save_error:
                logger.error(f"Failed to save policy checkpoint: {save_error}")
                print(f"  Failed to save policy checkpoint: {str(save_error)[:100]}...")

            instruction = "In Space Invaders, you control a ship at the bottom of the screen and defend against waves of descending aliens. "
            instruction += "The goal is to shoot as many aliens as possible while avoiding their projectiles. "
            instruction += "You score points by destroying aliens, with higher rows worth more points. "
            instruction += "The policy should decide when to shoot and how to move to maximize score while avoiding enemy fire. "
            instruction += "There are shields that absorb both player and alien projectiles. "
            instruction += "Analyze the trace to figure out how to improve your strategy for shooting aliens and dodging projectiles."
            instruction += "When you generate code, don't write any thing like '\\\\n' in your response. Also keep the function comments and docstrings as they are."
            
            optimizer.objective = optimizer.default_objective + instruction 
            
            optimizer.zero_feedback()
            optimizer.backward(target, feedback, visualize=True)
            logger.info(optimizer.problem_instance(optimizer.summarize()))
            
            stdout_buffer = io.StringIO()
            with contextlib.redirect_stdout(stdout_buffer):
                optimizer.step(verbose=verbose)
                llm_output = stdout_buffer.getvalue()
                if llm_output:
                    logger.info(f"LLM response:\n {llm_output}")
            
            # Create a summary visualization of this iteration if enabled
            if visualize and iter_vis_dir:
                # Create a summary image showing the final state
                if rollout_success and error is None and len(traj['observations']) > 0:
                    final_obs = traj['observations'][-1]
                    summary_path = os.path.join(iter_vis_dir, "final_state.png")
                    visualize_game_state(final_obs, traj['steps'], summary_path)
                    
                    # Create a text file with iteration summary
                    with open(os.path.join(iter_vis_dir, "summary.txt"), "w") as f:
                        f.write(f"Iteration: {i}\n")
                        f.write(f"Total Score: {episode_score:.1f}\n")
                        f.write(f"Steps: {traj['steps']}\n")
                        f.write(f"Average Test Score: {mean_rewards:.1f}\n")
                        f.write(f"Test Score Std Dev: {std_rewards:.1f}\n\n")
                        f.write(f"Feedback: {feedback}\n")
                        
    except Exception as e:
        logger.exception(f"Error during optimization: {e}")
        print(f"Error during optimization: {str(e)[:100]}...")
    finally:
        if env is not None:
            env.close()
    
    # Print final summary
    if rewards:
        avg_reward = sum(rewards) / len(rewards)
        logger.info(f"Final Average Reward: {avg_reward}")
        print("\n" + "="*50)
        print(f"Training Complete")
        print(f"Final Average Reward: {avg_reward:.1f}")
        print(f"Best Reward: {max(rewards):.1f}")
        print("="*50)
    else:
        logger.warning("No rewards collected during optimization")
        print("\n" + "="*50)
        print("Training failed - no rewards collected")
        print("="*50)
    
    return policy, rewards

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train an AI agent to play Space Invaders")
    parser.add_argument("--env", type=str, default="SpaceInvadersNoFrameskip-v4", help="Environment name")
    parser.add_argument("--horizon", type=int, default=10, help="Maximum steps per episode")
    parser.add_argument("--steps", type=int, default=20, help="Number of optimization steps")
    parser.add_argument("--memory", type=int, default=5, help="Memory size for optimization")
    parser.add_argument("--frameskip", type=int, default=4, help="Number of frames to skip")
    parser.add_argument("--sticky", type=float, default=0.0, help="Sticky action probability")
    parser.add_argument("--no-vis", action="store_true", help="Disable visualization")
    parser.add_argument("--no-debug", action="store_true", help="Disable debug info")
    parser.add_argument("--terminal-debug", action="store_true", help="Display debug info in terminal")
    parser.add_argument("--show-debug-files", action="store_true", help="Show paths to debug files after completion")
    parser.add_argument("--vis-frequency", type=int, default=1, help="Save visualization every N steps")
    parser.add_argument("--no-gif", action="store_true", help="Disable GIF creation (save individual PNGs instead)")
    parser.add_argument("--gif-fps", type=int, default=10, help="Frames per second for GIF")
    parser.add_argument("--enable-rollback", action="store_true", help="Enable policy rollback on error")
    
    args = parser.parse_args()
    
    # Setup logging
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"{args.env.replace('/', '_')}_{timestamp}_skip{args.frameskip}_sticky{args.sticky}_horizon{args.horizon}_optimSteps{args.steps}_mem{args.memory}_rollback{args.enable_rollback}.log"
    
    # Configure logging to file
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Space Invaders AI training...")
    
    try:
        # Run optimization
        policy, rewards = optimize_policy(
            env_name=args.env,
            horizon=args.horizon,
            memory_size=args.memory,
            n_optimization_steps=args.steps,
            frame_skip=args.frameskip,
            sticky_action_p=args.sticky,
            logger=logger,
            visualize=not args.no_vis,
            debug=not args.no_debug,
            terminal_debug=args.terminal_debug,
            vis_frequency=args.vis_frequency,
            create_gif=not args.no_gif,
            gif_fps=args.gif_fps,
            enable_rollback=args.enable_rollback,
        )
        
        # Show paths to debug files if requested
        if args.show_debug_files:
            vis_dir = Path("results/visualizations") / f"{args.env.replace('/', '_')}_{timestamp}"
            print("\n" + "="*50)
            print("Debug Files Locations:")
            print("="*50)
            print(f"Log file: {log_file}")
            if not args.no_vis:
                print(f"Visualizations directory: {vis_dir}")
                for i in range(args.steps):
                    iter_dir = vis_dir / f"iteration_{i}"
                    if iter_dir.exists():
                        print(f"  Iteration {i} debug log: {iter_dir}/debug_log.txt")
                        print(f"  Iteration {i} errors: {iter_dir}/errors.txt")
                        if not args.no_gif:
                            print(f"  Iteration {i} animation: {iter_dir}/animation.gif")
            print("="*50)
            
    except Exception as e:
        logger.exception(f"Fatal error during training: {e}")
        print(f"\nFatal error: {str(e)}")
        print(f"See log file for details: {log_file}")
        sys.exit(1)