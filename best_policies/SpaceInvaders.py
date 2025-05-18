import opto.trace as trace
from opto.trace import bundle, node, Module, GRAPH
import random

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
        - Shoot aliens and satellites and consider their movement when deciding to shoot (remember aliens and satellites move! so need to account for speed and shooting position)
        - Prioritize shooting aliens with larger y coordinates (closer to you)
        - Consider the movement of aliens when deciding to shoot
        
        Returns:
            bool: True if should shoot, False otherwise
        '''
        # There can only be one player bullet on the field at a time
        # Check for player bullets (which have negative dy velocity)
        for key, obj in obs.items():
            if key.startswith("Bullet") and obj.get("dy", 0) < 0:
                return False

        player = obs["Player"]
        closest_alien_distance = float("inf")
        closest_alien = None

        for key, obj in obs.items():
            if key.startswith("Alien"):
                distance = abs(obj["x"] - player["x"])
                if distance < closest_alien_distance:
                    closest_alien_distance = distance
                    closest_alien = obj

        if closest_alien:
            # Check if alien is aligned with player (within 15 pixels)
            if abs(closest_alien["x"] - player["x"]) < 15:
                # Prioritize lower aliens (higher y value)
                if (
                    closest_alien["y"] > 30
                ):  # Lowered threshold for more aggressive shooting
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
        - Move to dodge enemy projectiles. Use shields as covers, because they shield the alien bullets for the player but shields can be damaged and disappear.
        - Position yourself to shoot aliens and satellites (remember aliens and satellites move! so need to account for speed and shooting position)
        - Prioritize shooting aliens with larger y coordinates
        - Stay away from the edges of the screen
        
        Returns:
            int: -1 for left, 1 for right, 0 for no movement
        '''
        import random

        player = obs["Player"]
        move = 0
        threat_left = 0
        threat_right = 0
        aliens_left = 0
        aliens_right = 0
        screen_width = 160  # Assuming standard Space Invaders screen width

        for key, obj in obs.items():
            if key.startswith("Alien"):
                if obj["x"] < player["x"]:
                    aliens_left += 1
                else:
                    aliens_right += 1
            elif key.startswith("Bullet") and obj["dy"] > 0:  # Enemy bullet
                if obj["x"] < player["x"]:
                    threat_left += 1
                else:
                    threat_right += 1
                # Consider vertical position of bullets
                if abs(obj["x"] - player["x"]) < 10 and obj["y"] > player["y"] - 30:
                    move = 1 if obj["x"] < player["x"] else -1

        # Move away from threats if no immediate vertical threat
        if move == 0:
            if threat_left > threat_right:
                move = 1
            elif threat_right > threat_left:
                move = -1
            # If no immediate threat, move towards more aliens
            elif aliens_left > aliens_right:
                move = -1
            elif aliens_right > aliens_left:
                move = 1

        # Stay away from screen edges
        if player["x"] < 10 and move == -1:
            move = 1
        elif player["x"] > screen_width - 10 and move == 1:
            move = -1

        # Add small random movement
        if random.random() < 0.1:
            move = random.choice([-1, 0, 1])

        return move

    @bundle(trainable=True)
    def combine_actions(self, shoot, movement):
        '''
        Combine shooting and movement decisions into final action.
        
        Args:
            shoot (bool): Whether to shoot
            movement (int): Movement direction
        
        Strategy tips:
        - Move to dodge enemy projectiles
        - Position yourself to shoot aliens and satellites (remember aliens and satellites move! so need to account for speed and shooting position)
        - Prioritize shooting aliens with larger y coordinates
        - Stay away from the edges of the screen
        
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

