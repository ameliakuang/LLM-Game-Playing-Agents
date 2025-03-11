import opto.trace as trace
from opto.trace import bundle, Module

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
        Predict the y-coordinate where the ball will intersect with the player's paddle by calculating its trajectory,
        using ball's (x, y) and (dx, dy) and accounting for bounces off the top and bottom walls.

        Game Setup:
        - Screen dimensions: The game screen has boundaries where the ball bounces
          - Top boundary: approximately y=30
          - Bottom boundary: approximately y=190
        - Paddle positions:
          - Player paddle: right side of screen (x = 140)
          - Enemy paddle: left side of screen (x = 16)

        Args:
            obs (dict): Dictionary containing object states for "Player", "Ball", and "Enemy".
                       Each object has position (x,y), size (w,h), and velocity (dx,dy).

        Returns:
            float: Predicted y-coordinate where the ball will intersect the player's paddle plane.
                  Returns None if ball position cannot be determined.

        """
        if 'Ball' in obs:
            ball = obs['Ball']
            # If ball moving away from player, return None
            if ball.get('dx', 0) < 0:
                return None
                
            # Calculate time to reach paddle
            paddle_x = 140
            ball_x = ball.get('x', 0)
            ball_dx = ball.get('dx', 0)
            if ball_dx == 0:
                return ball.get('y', None)
                
            time_to_paddle = (paddle_x - ball_x) / ball_dx
            
            # Calculate predicted y position with improved accuracy
            ball_y = ball.get('y', 0)
            ball_dy = ball.get('dy', 0)
            predicted_y = ball_y + ball_dy * time_to_paddle
            
            # Account for bounces with improved accuracy
            num_bounces = 0
            while predicted_y < 30 or predicted_y > 190:
                if predicted_y < 30:
                    predicted_y = 30 + (30 - predicted_y)
                if predicted_y > 190:
                    predicted_y = 190 - (predicted_y - 190)
                num_bounces += 1
                if num_bounces > 4:  # Limit bounce calculations
                    break
                    
            return predicted_y
        return None
    
    @bundle(trainable=True)
    def select_action(self, predicted_ball_y, obs):
        '''
        Select the optimal action to move player paddle by comparing current player position and predicted_ball_y.
        
        IMPORTANT! Movement Logic:
        - If the player paddle's y position is GREATER than predicted_ball_y: Move DOWN (action 2)
          (because the paddle needs to move downward to meet the ball)
        - If the player paddle's y position is LESS than predicted_ball_y: Move UP (action 3)
          (because the paddle needs to move upward to meet the ball)
        - If the player paddle is already aligned with predicted_ball_y: NOOP (action 0)
          (to stabilize the paddle when it's in position)
        Ensure stable movement to avoid missing the ball when close by.

        Args:
            predicted_ball_y (float): predicted y coordinate of the ball or None
            obs(dict): Dictionary of current game state, mapping keys ("Player", "Ball", "Enemy") to values (dictionary of keys ('x', 'y', 'w', 'h', 'dx', 'dy') to integer values)
        Returns:
            int: 0 for NOOP, 2 for DOWN, 3 for UP
        '''
        if predicted_ball_y is not None and 'Player' in obs:
            # Calculate center of paddle
            paddle_center = obs['Player']['y'] + obs['Player']['h']/2
            
            # Increase margin and add dynamic adjustment based on ball distance
            base_margin = 4
            if 'Ball' in obs:
                ball_x = obs['Ball'].get('x', 0)
                dist_factor = (140 - ball_x) / 140  # Normalized distance factor
                margin = base_margin * (1 + dist_factor)  # Larger margin when ball is far
                
                # Add momentum-based adjustment
                if obs['Ball'].get('dx', 0) > 0:
                    ball_dy = obs['Ball'].get('dy', 0)
                    # Scale adjustment based on distance
                    predicted_ball_y += ball_dy * dist_factor
            else:
                margin = base_margin
            
            # More aggressive movement thresholds
            if paddle_center > predicted_ball_y + margin:
                return 2  # Move down
            elif paddle_center < predicted_ball_y - margin:
                return 3  # Move up
            return 0  # Stay in position
        return 0