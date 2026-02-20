import numpy as np
import opto.trace as trace
from opto.trace import bundle
from opto.trace.bundle import ExceptionNode
from opto.trace.errors import ExecutionError
from ocatari.core import OCAtari


class TracedEnv:
    def __init__(self,
                 env_name="EnduroNoFrameskip-v4",
                 render_mode="rgb_array",
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

    def render(self):
        """Render the environment by delegating to the underlying environment."""
        return self.env.render()

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
            self.obs = None

    def __del__(self):
        self.close()

    def extract_game_state(self, objects, info):
        """Extract structured game state from OCAtari objects.

        Returns a dict with:
          - "Player": {x, y, w, h}
          - "EnemyCars": list of enemy car states sorted by proximity (closest first):
              [{x, y, rel_x, rel_y, w, h}, ...]
              rel_x: enemy x - player x (negative = enemy is left, positive = right)
              rel_y: enemy y - player y (positive = enemy is close/below, negative = far/above)
          - "nearest_enemy_rel_x": rel_x of the closest enemy (quick access for steering)
          - "nearest_enemy_rel_y": rel_y of the closest enemy
        """
        obs = {}
        enemy_cars = []
        player_x = None
        player_y = None

        for obj in objects:
            if obj is None or obj.category == "NoObject":
                continue
            elif obj.category == "Player":
                player_x = obj.x
                player_y = obj.y
                obs["Player"] = {
                    "x": obj.x, "y": obj.y,
                    "w": obj.w, "h": obj.h,
                }
            elif obj.category == "Car":
                enemy_cars.append({
                    "x": obj.x, "y": obj.y,
                    "w": obj.w, "h": obj.h,
                })

        # Add relative positions and sort by proximity to player
        if player_x is not None and player_y is not None:
            for car in enemy_cars:
                car["rel_x"] = car["x"] - player_x
                car["rel_y"] = car["y"] - player_y
            # Sort by y descending (closest/highest y first)
            enemy_cars.sort(key=lambda c: -c["y"])

        if enemy_cars:
            obs["EnemyCars"] = enemy_cars
            obs["nearest_enemy_rel_x"] = enemy_cars[0]["rel_x"]
            obs["nearest_enemy_rel_y"] = enemy_cars[0]["rel_y"]

        return obs

    @bundle()
    def reset(self, seed=None, options=None):
        """
        Reset the environment and return the initial observation and info.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        self.obs = self.extract_game_state(self.env.objects, info)
        self.obs['reward'] = np.nan

        return self.obs, info

    def step(self, action):
        try:
            control = action.data if isinstance(action, trace.Node) else action
            next_obs, reward, termination, truncation, info = self.env.step(control)
            self.obs = self.extract_game_state(self.env.objects, info)
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
