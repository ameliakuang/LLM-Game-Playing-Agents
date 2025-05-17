import opto.trace as trace
from opto.trace import bundle, node, Module, GRAPH

@trace.model
class Policy(Module):
    def init(self):
        pass

    def __call__(self, obs):
        pre_ball_x = self.predict_ball_trajectory(obs)
        target_paddle_pos = self.generate_paddle_target(pre_ball_x, obs)
        action = self.select_paddle_action(target_paddle_pos, obs)
        return action
    @bundle(trainable=True)
    def predict_ball_trajectory(self, obs):
        """
        Predict the x-coordinate where the ball will intersect with the player's paddle by calculating its trajectory,
        using ball's (x, y) and (dx, dy) and accounting for bounces off the right and left walls and brick blocks.

        Game setup: 
        - Screen dimensions: The game screen has left and right walls and brick wall where the ball bounces 
          - Left wall: x=9
          - Right wall: x=152
          - brick wall: y is dynamically determined by the lowest brick blocks y coordinates at the x coordinate intersecting with ball
        - Paddle positions:
          - Player paddle: bottom of screen (y=189)
        - Ball speed:
          - Ball deflects from higher-scoring bricks would have a higher speed and is harder to catch.
        - The paddle would deflect the ball at different angles depending on where the ball lands on the paddle
        
        Args:
            obs (dict): Dictionary containing object states for "Player", "Ball", and "{color}B" (color in [R/O/Y/G/A/B]).
                       Each object has position (x,y), size (w,h), and velocity (dx,dy).
        Returns:
            float: Predicted x-coordinate where the ball will intersect the player's paddle plane.
                  Returns None if ball position cannot be determined.
        """
        if "Ball" not in obs:
            return None

        ball = obs["Ball"]
        ball_x = ball["x"]
        ball_y = ball["y"]
        ball_dx = ball["dx"]
        ball_dy = ball["dy"]

        # If ball is not moving or moving up, can't predict
        if ball_dy <= 0:
            return None

        # Calculate time to reach paddle
        paddle_y = 189
        time_to_paddle = (paddle_y - ball_y) / ball_dy

        # Calculate x position considering wall bounces
        num_bounces = 0
        pred_x = ball_x + (ball_dx * time_to_paddle)

        while pred_x < 9 or pred_x > 152:
            if pred_x < 9:
                pred_x = 9 + (9 - pred_x)
                num_bounces += 1
            elif pred_x > 152:
                pred_x = 152 - (pred_x - 152)
                num_bounces += 1
            if num_bounces > 10:  # Avoid infinite bounces
                return None

        return pred_x
    @bundle(trainable=True)
    def generate_paddle_target(self, pre_ball_x, obs):
        """
        Calculate the optimal x coordinate to move the paddle to catch the ball (at predicted_ball_x)
        and deflect the ball to hit bricks with higher scores in the brick wall.

        Logic:
        - Prioritize returning the ball when the ball is closer (larger y-coordinate) and coming down (positive dy)
        - Given the observation of blocks of different colors, try to return the ball in such a way that breaks through a tunnel in the brick wall, so the ball can then bounce around at the top of the brick walls to hit many bricks and score more points
        - Adaptively balance between safely returning the ball and aiming to hit the higher-scoring bricks in the brick block
        - Ball deflects from higher-scoring bricks would have a higher speed and is harder to catch

        Args:
            predicted_ball_x (float): predicted x coordinate of the ball intersecting with the paddle or None
            obs (dict): Dictionary containing object states for "Player", "Ball", and "{color}B" where color is Red(7pts)/Orange(7)/Yellow(4)/Green(4)/Aqua(1)/Blue(1).
                       Each object has position (x,y), size (w,h), and velocity (dx,dy).
        Returns:
            float: Predicted x-coordinate to move the paddle to. 
                Returns None if ball position cannot be determined.
        """
        if pre_ball_x is None or "Ball" not in obs:
            return None

        paddle = obs["Player"]
        paddle_w = paddle["w"]
        ball = obs["Ball"]
        ball_dx = ball["dx"]
        ball_y = ball["y"]

        # Find gaps in brick rows to aim for
        gaps = []
        for y in [87, 81, 75, 69, 63, 57]:  # Bottom to top rows
            row_blocks = [
                b
                for b in obs.get(
                    f'{"B" if y == 87 else "A" if y == 81 else "G" if y == 75 else "Y" if y == 69 else "O" if y == 63 else "R"}B',
                    [],
                )
            ]
            if not row_blocks:
                continue
            for i in range(len(row_blocks)):
                if i > 0:
                    gap_start = row_blocks[i - 1]["x"] + row_blocks[i - 1]["w"]
                    gap_end = row_blocks[i]["x"]
                    if gap_end - gap_start > 6:  # Min gap width
                        gaps.append((gap_start + gap_end) / 2)

        # Base offset that ensures reliable ball return
        base_offset = -3 if ball_dx > 0 else 3

        # Adjust offset based on ball height and gaps
        if ball_y < 90:  # Ball near brick wall
            if gaps:  # Aim for closest gap
                closest_gap = min(gaps, key=lambda x: abs(x - pre_ball_x))
                if abs(closest_gap - pre_ball_x) < 30:  # Gap within reach
                    return closest_gap

        # When ball is low or no good gaps available, focus on safe return
        return pre_ball_x + base_offset
    
    @bundle(trainable=True)
    def select_paddle_action(self, target_paddle_pos, obs):
        """
        Select the optimal action to move player paddle by comparing current player position and target_paddle_pos.

        Movement Logic:
        - If the player paddle's x position is GREATER than target_paddle_pos: Move LEFT (action 3)
          (because the paddle needs to move left to meet the ball)
        - If the player paddle's x position is LESS than target_paddle_pos: Move RIGHT (action 2)
          (because the paddle needs to move right to meet the ball)
        - If the player paddle is already aligned with target_paddle_pos: NOOP (action 0)
          (to stabilize the paddle when it's in position)
        Ensure stable movement to avoid missing the ball when close by.

        Args:
            target_paddle_pos (float): predicted x coordinate of the position to best position the paddle to catch the ball,
                and hit the ball to break brick wall.
            obs (dict): Dictionary containing object states for "Player", "Ball", and "{color}B" (color in [R/O/Y/G/A/B]).
                Each object has position (x,y), size (w,h), and velocity (dx,dy).
        Returns:
            int: 0 for NOOP, 2 for RIGHT, 3 for LEFT
        """
        if target_paddle_pos is None or "Player" not in obs:
            return 0

        paddle = obs["Player"]
        paddle_x = paddle["x"]
        paddle_w = paddle["w"]
        paddle_center = paddle_x + (paddle_w / 2)
        ball = obs.get("Ball", {})

        # Adaptive deadzone based on ball position and speed
        base_deadzone = 3
        ball_y = ball.get("y", 189)
        ball_dy = abs(ball.get("dy", 0))

        # Larger deadzone for faster balls and higher positions
        height_factor = (189 - ball_y) / 189
        speed_factor = ball_dy / 4
        deadzone = base_deadzone * (1 + height_factor + speed_factor)

        if abs(paddle_center - target_paddle_pos) < deadzone:
            return 0  # NOOP if close enough
        elif paddle_center > target_paddle_pos:
            return 3  # LEFT
        else:
            return 2  # RIGHT