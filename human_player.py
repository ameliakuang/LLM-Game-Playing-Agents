import os
import sys
import time
import json
import datetime
import pygame
import numpy as np
import cv2
from pathlib import Path
import gymnasium as gym
import ale_py
from ocatari.core import OCAtari
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Register Atari environments
gym.register_envs(ale_py)

class AtariHumanPlayer:
    def __init__(self, 
                 save_dir=None,
                 env_name=None,
                 render_mode="rgb_array",
                 obs_mode="obj",
                 hud=True,
                 frameskip=1,
                 repeat_action_probability=0.0):
        
        # Environment setup
        self.env_name = env_name
        self.render_mode = render_mode
        self.obs_mode = obs_mode
        self.hud = hud
        self.frameskip = frameskip
        self.repeat_action_probability = repeat_action_probability
        
        # Create output directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if save_dir is None:
            save_dir = Path(f"human_play_{timestamp}")
        else:
            save_dir = Path(save_dir) / timestamp
        
        # Create separate directories for raw frames, visualization frames, and state data
        self.raw_frames_dir = save_dir / "raw_frames"
        self.vis_frames_dir = save_dir / "vis_frames"
        self.states_dir = save_dir / "states"
        
        # Create all directories
        self.raw_frames_dir.mkdir(parents=True, exist_ok=True)
        self.vis_frames_dir.mkdir(parents=True, exist_ok=True)
        self.states_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize pygame for keyboard input
        pygame.init()
        self.screen = pygame.display.set_mode((160, 210))
        pygame.display.set_caption(self.get_game_title())
        
        # Environment
        self.env = None
        self.obs = None
        self.frame_count = 0
        self.total_reward = 0
        
        # Initialize environment
        self.init_env()
        
        # Print instructions
        self.print_instructions()
    
    def get_game_title(self):
        """Get the game title for the window caption."""
        raise NotImplementedError
    
    def print_instructions(self):
        """Print game instructions to the console."""
        raise NotImplementedError
    
    def init_env(self):
        """Initialize the environment."""
        if self.env is not None:
            self.env.close()
        
        # Create two environments - one for display and one for capturing frames
        # Main environment is for gameplay
        self.env = OCAtari(self.env_name, 
                          render_mode="human",  # Always use "human" for the interactive display
                          obs_mode=self.obs_mode, 
                          hud=self.hud,
                          frameskip=self.frameskip,
                          repeat_action_probability=self.repeat_action_probability)
        
        # Create a secondary environment just for capturing raw frames
        self.frame_env = OCAtari(self.env_name,
                                render_mode="rgb_array",
                                obs_mode=self.obs_mode,
                                hud=self.hud,
                                frameskip=self.frameskip,
                                repeat_action_probability=self.repeat_action_probability)
        
        # Initialize both environments with the same seed for synchronization
        seed = np.random.randint(0, 2**31)
        self.obs, _ = self.env.reset(seed=seed)
        self.frame_env.reset(seed=seed)
        
        self.frame_count = 0
        self.total_reward = 0
    
    def extract_obj_state(self, objects):
        """Extract object state information from OCAtari objects."""
        obs = dict()
        # Count objects by category to create unique keys
        category_counts = {}
        
        for obj in objects:
            category = obj.category

            if category == "NoObject":
                continue

            # For objects that might have multiple instances
            if category in self.get_multiple_instance_categories():
                if category not in category_counts:
                    category_counts[category] = 0
                else:
                    category_counts[category] += 1
                
                # Create indexed key for multiple objects of same category
                key = f"{category}{category_counts[category]}"
            else:
                key = category
                
            obs[key] = {"x": obj.x,
                       "y": obj.y,
                       "w": obj.w,
                       "h": obj.h,
                       "dx": obj.dx,
                       "dy": obj.dy,}
        return obs
    
    def get_multiple_instance_categories(self):
        """Get list of object categories that can have multiple instances."""
        raise NotImplementedError
    
    def get_action_from_keys(self):
        """Get the current action based on keyboard input."""
        raise NotImplementedError
    
    def get_object_colors(self):
        """Get color mapping for different object types."""
        raise NotImplementedError
    
    def visualize_game_state(self, obs, step_num=None, save_path=None):
        """
        Visualize the game state from object observations.
        """
        # Create a blank canvas (black background)
        canvas = np.zeros((210, 160, 3), dtype=np.uint8)
        
        # Draw objects
        for key, obj in obs.items():
            # Skip reward or non-dict objects
            if key == 'reward' or not isinstance(obj, dict):
                continue
            
            # Extract coordinates safely
            try:
                x = float(obj.get('x', 0))
                y = float(obj.get('y', 0))
                w = float(obj.get('w', 5))
                h = float(obj.get('h', 5))
                dx = float(obj.get('dx', 0))
                dy = float(obj.get('dy', 0))
            except (ValueError, TypeError):
                continue
            
            # Get color for this object type
            color = self.get_object_colors().get(key, (255, 255, 255))  # Default to white
                
            # Draw rectangle for the object
            cv2.rectangle(canvas, (int(x), int(y)), (int(x+w), int(y+h)), color, 1)
            
            # Add label
            cv2.putText(canvas, key, (int(x), int(y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Add step number if provided
        if step_num is not None:
            cv2.putText(canvas, f"Step: {step_num}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add reward if available
        if 'reward' in obs:
            reward = obs['reward']
            try:
                if isinstance(reward, (int, float)) and not np.isnan(reward):
                    cv2.putText(canvas, f"Reward: {reward}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            except:
                pass  # Skip if reward can't be displayed
        
        # Save if path is provided
        if save_path:
            cv2.imwrite(save_path, canvas)
        
        return canvas
    
    def save_raw_frame(self, frame, frame_num):
        """Save a raw RGB frame to disk."""
        frame_path = self.raw_frames_dir / f"frame_{frame_num:06d}.png"
        cv2.imwrite(str(frame_path), frame)
        return frame_path
    
    def save_vis_frame(self, frame, frame_num):
        """Save a visualization frame to disk."""
        frame_path = self.vis_frames_dir / f"vis_{frame_num:06d}.png"
        cv2.imwrite(str(frame_path), frame)
        return frame_path
    
    def save_state(self, state_data, frame_num):
        """Save state data to disk as JSON."""
        state_path = self.states_dir / f"state_{frame_num:06d}.json"
        
        # Convert any numpy values to Python types for JSON serialization
        cleaned_data = {}
        for key, value in state_data.items():
            if isinstance(value, dict):
                cleaned_data[key] = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        cleaned_data[key][k] = v.tolist()
                    elif isinstance(v, np.integer):
                        cleaned_data[key][k] = int(v)
                    elif isinstance(v, np.floating):
                        cleaned_data[key][k] = float(v)
                    else:
                        cleaned_data[key][k] = v
            elif isinstance(value, np.ndarray):
                cleaned_data[key] = value.tolist()
            elif isinstance(value, np.integer):
                cleaned_data[key] = int(value)
            elif isinstance(value, np.floating):
                cleaned_data[key] = float(value)
            else:
                cleaned_data[key] = value
        
        with open(state_path, 'w') as f:
            json.dump(cleaned_data, f, indent=2)
        
        return state_path
    
    def run(self):
        """Main game loop."""
        running = True
        clock = pygame.time.Clock()
        
        try:
            while running:
                # Process pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                
                # Get action from keyboard
                action = self.get_action_from_keys()
                if action == -1:  # Quit signal
                    running = False
                    break
                
                # Step the main environment
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                
                # Also step the frame capture environment with the same action
                _, _, _, _, _ = self.frame_env.step(action)
                
                # Extract object state from the main environment
                obj_state = self.extract_obj_state(self.env.objects)
                obj_state['reward'] = reward
                
                # Get raw frame from the frame capture environment
                raw_frame = self.frame_env.render()
                
                # Create visualization frame
                vis_frame = self.visualize_game_state(obj_state, self.frame_count)
                
                # Save all data for this frame
                
                # 1. Save raw frame (if available)
                if raw_frame is not None:
                    # Convert from RGB to BGR if needed (OpenCV uses BGR)
                    if raw_frame.shape[-1] == 3:  # If it's a color image
                        raw_frame = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)
                    
                    self.save_raw_frame(raw_frame, self.frame_count)
                else:
                    print("\nWarning: Could not capture raw frame")
                
                # 2. Save visualization frame
                self.save_vis_frame(vis_frame, self.frame_count)
                
                # 3. Save state data
                self.save_state(obj_state, self.frame_count)
                
                # Update state
                self.frame_count += 1
                self.total_reward += reward
                
                # Print info
                sys.stdout.write(f"\rFrame: {self.frame_count} | Action: {action} | Reward: {reward:.1f} | Total: {self.total_reward:.1f}")
                sys.stdout.flush()
                
                # Check if episode is done
                if terminated or truncated:
                    print(f"\nEpisode finished after {self.frame_count} frames with total reward {self.total_reward:.1f}")
                    self.init_env()  # Reset for next episode
                
                # Cap the frame rate
                clock.tick(30)  # 30 FPS
        
        except Exception as e:
            print(f"\nError: {e}")
        
        finally:
            # Clean up
            if self.env is not None:
                self.env.close()
            if hasattr(self, 'frame_env') and self.frame_env is not None:
                self.frame_env.close()
            pygame.quit()
            
            print("\n" + "="*50)
            print(f"Game session ended. Saved {self.frame_count} frames.")
            print(f"Raw frames saved to: {self.raw_frames_dir}")
            print(f"Visualization frames saved to: {self.vis_frames_dir}")
            print(f"State data saved to: {self.states_dir}")
            print("="*50)

class SpaceInvadersHumanPlayer(AtariHumanPlayer):
    def __init__(self, **kwargs):
        if 'env_name' not in kwargs:
            kwargs['env_name'] = "SpaceInvadersNoFrameskip-v4"
        super().__init__(**kwargs)
        
        # Action mapping for Space Invaders
        # 0: NOOP, 1: FIRE, 2: RIGHT, 3: LEFT, 4: RIGHT+FIRE, 5: LEFT+FIRE
        self.key_action_map = {
            pygame.K_SPACE: 1,     # FIRE
            pygame.K_RIGHT: 2,     # RIGHT
            pygame.K_LEFT: 3,      # LEFT
        }
    
    def get_game_title(self):
        return "Space Invaders Human Player"
    
    def print_instructions(self):
        """Print game instructions to the console."""
        print("\n" + "="*50)
        print("SPACE INVADERS HUMAN PLAYER")
        print("="*50)
        print("Controls:")
        print("  LEFT ARROW: Move Left")
        print("  RIGHT ARROW: Move Right")
        print("  SPACE: Fire")
        print("  ESC or Q: Quit")
        print("="*50)
        print(f"Saving raw frames to: {self.raw_frames_dir}")
        print(f"Saving visualization frames to: {self.vis_frames_dir}")
        print(f"Saving state data to: {self.states_dir}")
        print("="*50 + "\n")
    
    def get_multiple_instance_categories(self):
        return ["Alien", "Shield", "Bullet"]
    
    def get_action_from_keys(self):
        """Get the current action based on keyboard input."""
        keys = pygame.key.get_pressed()
        
        # Check for quit
        if keys[pygame.K_ESCAPE] or keys[pygame.K_q]:
            return -1  # Special value to indicate quit
        
        # Check for combined actions
        if keys[pygame.K_SPACE] and keys[pygame.K_RIGHT]:
            return 4  # RIGHT+FIRE
        elif keys[pygame.K_SPACE] and keys[pygame.K_LEFT]:
            return 5  # LEFT+FIRE
        # Check for single actions
        elif keys[pygame.K_SPACE]:
            return 1  # FIRE
        elif keys[pygame.K_RIGHT]:
            return 2  # RIGHT
        elif keys[pygame.K_LEFT]:
            return 3  # LEFT
        
        # Default - no action
        return 0  # NOOP
    
    def get_object_colors(self):
        return {
            'Player': (0, 255, 0),  # Green for player
            'Alien': (255, 0, 0),   # Red for aliens
            'Bullet': (0, 255, 255),  # Cyan for bullets
            'Shield': (0, 0, 255),   # Blue for shields
        }

class PongHumanPlayer(AtariHumanPlayer):
    def __init__(self, **kwargs):
        if 'env_name' not in kwargs:
            kwargs['env_name'] = "PongNoFrameskip-v4"
        super().__init__(**kwargs)
        
        # Action mapping for Pong
        # 0: NOOP, 2: DOWN, 3: UP
        self.key_action_map = {
            pygame.K_UP: 3,    # UP
            pygame.K_DOWN: 2,  # DOWN
        }
    
    def get_game_title(self):
        return "Pong Human Player"
    
    def print_instructions(self):
        """Print game instructions to the console."""
        print("\n" + "="*50)
        print("PONG HUMAN PLAYER")
        print("="*50)
        print("Controls:")
        print("  UP ARROW: Move Paddle Up")
        print("  DOWN ARROW: Move Paddle Down")
        print("  ESC or Q: Quit")
        print("="*50)
        print(f"Saving raw frames to: {self.raw_frames_dir}")
        print(f"Saving visualization frames to: {self.vis_frames_dir}")
        print(f"Saving state data to: {self.states_dir}")
        print("="*50 + "\n")
    
    def get_multiple_instance_categories(self):
        # In Pong, we don't have multiple instances of any objects
        return []
    
    def get_action_from_keys(self):
        """Get the current action based on keyboard input."""
        keys = pygame.key.get_pressed()
        
        # Check for quit
        if keys[pygame.K_ESCAPE] or keys[pygame.K_q]:
            return -1  # Special value to indicate quit
        
        # Check for paddle movement
        if keys[pygame.K_UP]:
            return 3  # UP
        elif keys[pygame.K_DOWN]:
            return 2  # DOWN
        
        # Default - no action
        return 0  # NOOP
    
    def get_object_colors(self):
        return {
            'Player': (0, 255, 0),    # Green for player paddle
            'Enemy': (255, 0, 0),     # Red for enemy paddle
            'Ball': (255, 255, 0),    # Yellow for ball
        }
    
    def visualize_game_state(self, obs, step_num=None, save_path=None):
        """
        Visualize the game state from object observations.
        """
        # Create a blank canvas (black background)
        canvas = np.zeros((210, 160, 3), dtype=np.uint8)
        
        # Draw objects
        for key, obj in obs.items():
            # Skip reward or non-dict objects
            if key == 'reward' or not isinstance(obj, dict):
                continue
            
            # Extract coordinates safely
            try:
                x = float(obj.get('x', 0))
                y = float(obj.get('y', 0))
                w = float(obj.get('w', 5))
                h = float(obj.get('h', 5))
                dx = float(obj.get('dx', 0))
                dy = float(obj.get('dy', 0))
            except (ValueError, TypeError):
                continue
            
            # Get color for this object type
            color = self.get_object_colors().get(key, (255, 255, 255))  # Default to white
                
            # Draw rectangle for the object
            cv2.rectangle(canvas, (int(x), int(y)), (int(x+w), int(y+h)), color, 1)
            
            # Add label with adjusted position based on object type
            if key == 'Player':
                # Position label to the right of the player paddle
                label_x = int(x + w - 20)
            elif key == 'Enemy':
                # Position label to the left of the enemy paddle
                label_x = int(x - 10)
            else:
                # For ball, position label above it
                label_x = int(x)
            
            cv2.putText(canvas, key, (label_x, int(y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Add step number if provided
        if step_num is not None:
            cv2.putText(canvas, f"Step: {step_num}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add reward if available
        if 'reward' in obs:
            reward = obs['reward']
            try:
                if isinstance(reward, (int, float)) and not np.isnan(reward):
                    cv2.putText(canvas, f"Reward: {reward}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            except:
                pass  # Skip if reward can't be displayed
        
        # Save if path is provided
        if save_path:
            cv2.imwrite(save_path, canvas)
        
        return canvas

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Human player for Atari games with frame and state capture")
    parser.add_argument("--game", type=str, default="pong", choices=["space_invaders", "pong"], help="Game to play")
    parser.add_argument("--save_dir", type=str, default="my_gameplay", help="Directory to save frames and state data")
    parser.add_argument("--env_name", type=str, default=None, help="Environment name (will be set based on game)")
    parser.add_argument("--render_mode", type=str, default="human", help="Render mode (human, rgb_array)")
    parser.add_argument("--obs_mode", type=str, default="obj", help="Observation mode (obj, ram, rgb)")
    parser.add_argument("--hud", action="store_true", help="Show HUD (heads-up display)")
    parser.add_argument("--frameskip", type=int, default=1, help="Frame skip (1 for smoother human play)")
    parser.add_argument("--sticky_actions", type=float, default=0.0, help="Sticky action probability")
    
    args = parser.parse_args()
    
    # Set default environment name based on game
    if args.env_name is None:
        if args.game == "pong":
            args.env_name = "PongNoFrameskip-v4"
        else:  # space_invaders
            args.env_name = "SpaceInvadersNoFrameskip-v4"
    
    # Create appropriate player based on game
    if args.game == "pong":
        player = PongHumanPlayer(
            save_dir=args.save_dir,
            env_name=args.env_name,
            render_mode="human",  # Override to always use human for main env
            obs_mode=args.obs_mode,
            hud=args.hud,
            frameskip=args.frameskip,
            repeat_action_probability=args.sticky_actions
        )
    else:  # space_invaders
        player = SpaceInvadersHumanPlayer(
            save_dir=args.save_dir,
            env_name=args.env_name,
            render_mode="human",  # Override to always use human for main env
            obs_mode=args.obs_mode,
            hud=args.hud,
            frameskip=args.frameskip,
            repeat_action_probability=args.sticky_actions
        )
    
    player.run() 