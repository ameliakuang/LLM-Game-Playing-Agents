import numpy as np
import opto.trace as trace
from opto.trace import bundle
from opto.trace.bundle import ExceptionNode
from opto.trace.errors import ExecutionError
from ocatari.core import OCAtari


class TracedEnv:
    def __init__(self,
                 env_name="FreewayNoFrameskip-v4",
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

    def extract_game_state(self, objects):
        """Extract structured game state from OCAtari objects.

        Returns a dict with:
          - "Chicken": player 1 chicken state {x, y, w, h, dx, dy}
          - "Cars": list of car states sorted by lane (top to bottom),
                    each {x, y, w, h, dx, dy}
        """
        obs = {}
        cars = []
        for obj in objects:
            if obj.category == "NoObject":
                continue
            elif obj.category == "Chicken" and obj.x < 80:
                # Player 1 chicken (left side, x~44)
                obs["Chicken"] = {
                    "x": obj.x, "y": obj.y,
                    "w": obj.w, "h": obj.h,
                    "dx": obj.dx, "dy": obj.dy,
                }
            elif obj.category == "Car":
                cars.append({
                    "x": obj.x, "y": obj.y,
                    "w": obj.w, "h": obj.h,
                    "dx": obj.dx, "dy": obj.dy,
                })
        if cars:
            # Sort by lane position (top to bottom)
            cars.sort(key=lambda c: c["y"])
            obs["Cars"] = cars
        return obs

    @bundle()
    def reset(self):
        """
        Reset the environment and return the initial observation and info.

        Freeway has no lives and no FIRE action, so reset is straightforward.
        """
        _, info = self.env.reset()
        self.obs = self.extract_game_state(self.env.objects)
        self.obs['reward'] = np.nan

        return self.obs, info

    def step(self, action):
        try:
            control = action.data if isinstance(action, trace.Node) else action
            next_obs, reward, termination, truncation, info = self.env.step(control)
            self.obs = self.extract_game_state(self.env.objects)
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
