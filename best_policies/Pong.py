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
        - Top boundary: y=30
        - Bottom boundary: y=190
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
        if "Ball" not in obs:
            return None

        ball = obs["Ball"]
        ball_x = float(ball.get("x", 0))
        ball_y = float(ball.get("y", 0))
        ball_dx = float(ball.get("dx", 0))
        ball_dy = float(ball.get("dy", 0))

        if ball_dx == 0:
            # Special handling for vertical movement
            if ball_dy > 0:
                # Ball moving down
                return min(190.0, ball_y + 4.0)
            elif ball_dy < 0:
                # Ball moving up
                return max(30.0, ball_y - 4.0)
            return ball_y

        # Calculate time to reach paddle
        paddle_x = 140.0
        time_to_paddle = (paddle_x - ball_x) / ball_dx

        # Calculate predicted y without bounces
        predicted_y = ball_y + ball_dy * time_to_paddle

        # Handle bounces with improved precision
        while predicted_y < 30 or predicted_y > 190:
            if predicted_y < 30:
                predicted_y = 60.0 - predicted_y  # Reflect off top
            elif predicted_y > 190:
                predicted_y = 380.0 - predicted_y  # Reflect off bottom

        # Adjust prediction near boundaries
        if predicted_y < 40:
            predicted_y = 40.0
        elif predicted_y > 180:
            predicted_y = 180.0

        return predicted_y
    
    @bundle(trainable=True)
    def select_action(self, predicted_ball_y, obs):
        """
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
            obs(dict): Dictionary of current game state, mapping keys ("Player", "Ball", "Enemy") to values
        Returns:
            int: 0 for NOOP, 2 for DOWN, 3 for UP
        """
        if predicted_ball_y is None or "Player" not in obs or "Ball" not in obs:
            return 0

        paddle_y = float(obs["Player"].get("y", 0))
        paddle_h = float(obs["Player"].get("h", 15))  # Default paddle height

        # Calculate center of paddle with improved precision
        paddle_center = paddle_y + paddle_h / 2.0

        ball = obs["Ball"]
        ball_x = float(ball.get("x", 0))
        ball_dx = float(ball.get("dx", 0))
        ball_dy = float(ball.get("dy", 0))

        # Base tolerance increased for faster response
        base_tolerance = 4.0

        # Distance-based momentum - be more aggressive when ball is close
        distance = abs(140.0 - ball_x)
        distance_factor = max(0.5, min(2.0, distance / 70.0))  # Scale with distance

        # Velocity-based momentum
        speed_momentum = min(abs(ball_dy) / 2.0, 3.0)

        # Combined adaptive tolerance
        tolerance = base_tolerance * distance_factor + speed_momentum

        # Early movement when ball is far and moving slowly
        if distance > 100 and abs(ball_dy) < 2:
            tolerance *= 0.5

        # Special handling for straight ball movement
        if ball_dx == 0:
            if abs(ball_dy) > 0:
                # Move towards predicted intersection more aggressively
                tolerance *= 0.5

        # Tighter tolerance near paddle edges
        if paddle_y < 40 or paddle_y > 180:
            tolerance *= 0.7

        # Decision making with improved positioning
        diff = paddle_center - predicted_ball_y
        if abs(diff) < tolerance:
            return 0  # Stay in position
        elif diff > 0:
            return 2  # Move down
        else:
            return 3  # Move up