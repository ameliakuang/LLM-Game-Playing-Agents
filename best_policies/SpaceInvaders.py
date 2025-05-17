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
        for key, obj in obs.items():
            if key.startswith("Bullet") and obj.get("dy", 0) < 0:
                return False

        player = obs["Player"]
        for key, obj in obs.items():
            if key.startswith("Alien"):
                if abs(obj["x"] - player["x"]) < 7:  # Increased alignment threshold
                    if obj["y"] > 60 or (
                        obj["y"] > 40 and obj.get("dy", 0) > 0
                    ):  # More aggressive for lower aliens
                        lead_factor = 0.3  # Increased lead factor
                        predicted_x = obj["x"] + obj.get("dx", 0) * lead_factor
                        if abs(predicted_x - player["x"]) < 8:
                            return True
                elif abs(obj["x"] - player["x"]) < 15 and (
                    (obj["x"] > player["x"] and obj.get("dx", 0) < 0)
                    or (obj["x"] < player["x"] and obj.get("dx", 0) > 0)
                ):
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
        player = obs["Player"]
        move = 0
        threat_left = 0
        threat_right = 0
        aliens_left = 0
        aliens_right = 0

        screen_width = 210
        edge_buffer = 25  # Increased edge buffer

        for key, obj in obs.items():
            if key.startswith("Alien"):
                if obj["x"] < player["x"]:
                    aliens_left += 1
                else:
                    aliens_right += 1
            elif key.startswith("Bullet") and obj["dy"] > 0:
                time_to_player = (player["y"] - obj["y"]) / obj["dy"]
                predicted_x = obj["x"] + obj["dx"] * time_to_player
                distance = abs(predicted_x - player["x"])
                if distance < 20:  # Increased dodging threshold
                    threat_weight = max(0, 20 - distance)
                    if obj["x"] < player["x"]:
                        threat_left += threat_weight
                    else:
                        threat_right += threat_weight

        if threat_left > 0 or threat_right > 0:
            if threat_left > threat_right:
                move = 1
            else:
                move = -1
        else:
            alien_density_left = aliens_left / max(1, player["x"])
            alien_density_right = aliens_right / max(1, screen_width - player["x"])
            if alien_density_left > alien_density_right:
                move = -1
            elif alien_density_right > alien_density_left:
                move = 1

        if player["x"] < edge_buffer and move == -1:
            move = 1
        elif player["x"] > screen_width - edge_buffer and move == 1:
            move = -1

        if random.random() < 0.05:  # Reduced random movement probability
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

