import gymnasium as gym
import ale_py
import numpy as np
import cv2

def process_frame(gray):
    """
    Processes a grayscale frame from Space Invaders.
    
    This function thresholds the frame to extract game objects,
    then uses contour detection to estimate the positions of the player's spaceship,
    enemy invaders, and enemy lasers.
    """
    # Threshold to highlight bright objects (spaceship, invaders, lasers, etc.)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    
    # Visualize the thresholded image
    cv2.imshow("Threshold", thresh)
    cv2.waitKey(1)
    
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Visualize the detected contours
    contour_image = gray.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 1)
    cv2.imshow("Contours", contour_image)
    cv2.waitKey(1)
    
    spaceship_pos = None
    invaders = []
    lasers = []
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        
        # Heuristic: the player's spaceship is usually larger and located near the bottom
        if y > 150 and area > 10:
            spaceship_pos = (x + w // 2, y + h // 2)
        # Heuristic: enemy invaders are typically small and appear in the upper half
        elif y < 100 and 2 < area < 50:
            invaders.append((x + w // 2, y + h // 2))
        # Heuristic: enemy lasers might appear in the mid-region (between invaders and spaceship)
        # and have a relatively small area.
        elif 100 <= y <= 150 and 5 < area < 15:
            lasers.append((x + w // 2, y + h // 2))
    
    return spaceship_pos, invaders, lasers

def decide_movement(spaceship_pos, invaders, lasers, laser_margin=10, chase_margin=30):
    """
    Determines the movement command based on laser avoidance and chasing monsters.
    
    1. If any laser is detected ahead (within 'laser_margin' pixels horizontally),
       return a movement command to move right (action 2) to avoid it.
    2. Otherwise, if invaders are detected, compute their average x-position and
       only move if the spaceship's x-position is far (beyond 'chase_margin') from that average.
       
    Returns:
        2: move RIGHT, 3: move LEFT, 0: NOOP (no movement).
    """
    # First, check for laser threat.
    for laser in lasers:
        if abs(laser[0] - spaceship_pos[0]) < laser_margin:
            return 2  # Immediate avoidance: move right.
    
    # If no immediate laser threat, check if chasing monsters is necessary.
    if invaders:
        avg_invader_x = np.mean([inv[0] for inv in invaders])
        ship_x = spaceship_pos[0]
        if ship_x < avg_invader_x - chase_margin:
            return 2  # Move right to chase the monsters.
        elif ship_x > avg_invader_x + chase_margin:
            return 3  # Move left to chase the monsters.
    
    return 0  # No movement needed.

def decide_firing():
    """
    Firing decision: continuously fire.
    
    Returns:
        1: FIRE action.
    """
    return 1

def simple_space_invaders_ai(observation):
    """
    Heuristic policy for Space Invaders.
    
    It uses two independent functions:
      - A movement function that checks for laser avoidance and chases monsters.
      - A firing function that continuously fires.
    
    The high-level decision logic:
      1. If the movement function indicates a movement (non-0 action), that action is chosen.
      2. Otherwise, the agent fires.
    """
    spaceship_pos, invaders, lasers = process_frame(observation)
    action = 0  # Default action: NOOP (do nothing)
    
    if spaceship_pos:
        move_action = decide_movement(spaceship_pos, invaders, lasers)
        if move_action != 0:
            action = move_action
            print(f"\rMovement: {move_action} (laser avoidance or chasing), Spaceship: {spaceship_pos}", end="")
        else:
            action = decide_firing()
            print(f"\rFiring continuously. Spaceship: {spaceship_pos}", end="")
    else:
        print("\rNo detection - Spaceship not found", end="")
    
    return action

if __name__ == "__main__":
    
    gym.register_envs(ale_py)
    
    # Create the Space Invaders environment.
    env = gym.make("ALE/SpaceInvaders-v5", render_mode="human", obs_type="grayscale")
    
    # Initialize the environment.
    obs, _ = env.reset()
    print(f"Initial obs shape: {obs.shape}")
    episode_steps = 0
    total_reward = 0
    
    done = False
    while not done:
        # Get action from the heuristic AI.
        action = simple_space_invaders_ai(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        print(f" info: {info}, reward: {reward}")
        total_reward += reward
        episode_steps += 1
        
        if terminated or truncated:
            print(f"\nEpisode finished after {episode_steps} steps. Total reward: {total_reward}")
        
        done = terminated or truncated
    
    env.close()
    cv2.destroyAllWindows()

