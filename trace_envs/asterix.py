import numpy as np
import opto.trace as trace
from opto.trace import bundle
from opto.trace.bundle import ExceptionNode
from opto.trace.errors import ExecutionError
from ocatari.core import OCAtari

LANE_Y_POSITIONS = [26, 42, 58, 74, 90, 106, 122, 138]

def _y_to_lane(y):
    """Convert a y-coordinate to lane index (0-7)."""
    return min(range(8), key=lambda i: abs(LANE_Y_POSITIONS[i] - y))


class TracedEnv:
    def __init__(self,
                 env_name="AsterixNoFrameskip-v4",
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
        self.lives = 0
        self.was_real_done = True
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
        self.lives = self.env._env.unwrapped.ale.lives()
        self.was_real_done = True

    def render(self):
        """Render the environment by delegating to the underlying environment."""
        return self.env.render()

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
            self.obs = None
            self.lives = 0

    def __del__(self):
        self.close()

    def extract_game_state(self, objects, info):
        """Extract structured game state from OCAtari objects.

        Returns a dict with:
          - "Player": {x, y}
          - "Enemies": list sorted by distance to player [{x, y, type}, ...]
          - "Consumables": list sorted by distance to player [{x, y, type}, ...]
          - "Rewards": list sorted by distance to player [{x, y, type}, ...]
          - "lives": remaining lives
        """
        obs = {}
        enemies = []
        consumables = []
        rewards = []
        player_x = None
        player_y = None

        for obj in objects:
            if obj is None or obj.category == "NoObject":
                continue
            elif obj.category == "Player":
                player_x = obj.x
                player_y = obj.y
                obs["Player"] = {"x": obj.x, "y": obj.y}
            elif obj.category == "Enemy":
                enemies.append({"x": obj.x, "y": obj.y, "type": "Enemy"})
            elif obj.category == "Consumable":
                consumables.append({"x": obj.x, "y": obj.y, "type": "Consumable"})
            elif obj.category == "Reward":
                rewards.append({"x": obj.x, "y": obj.y, "type": "Reward"})

        # Sort by Manhattan distance from player (nearest first)
        if player_x is not None and player_y is not None:
            for item_list in [enemies, consumables, rewards]:
                item_list.sort(key=lambda item: abs(item["x"] - player_x) + abs(item["y"] - player_y))

        if enemies:
            obs["Enemies"] = enemies
        if consumables:
            obs["Consumables"] = consumables
        if rewards:
            obs["Rewards"] = rewards

        if info:
            obs['lives'] = info.get('lives', None)
        return obs

    @bundle()
    def reset(self, seed=None, options=None):
        """
        Reset the environment and return the initial observation and info.

        Implements EpisodicLifeEnv logic (see openai/baselines):
          On true game-over, do a real env.reset().
          On life loss, just NOOP to advance past the death frame.
        Asterix has no FIRE action, so no FireResetEnv phase is needed.
        """
        # EpisodicLifeEnv
        if self.was_real_done:
            obs, info = self.env.reset(seed=seed, options=options)
        else:
            # NOOP to advance past the death frame
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset(seed=seed, options=options)
        self.lives = self.env._env.unwrapped.ale.lives()

        self.obs = self.extract_game_state(self.env.objects, info)
        self.obs['reward'] = np.nan

        return self.obs, info

    def step(self, action):
        try:
            control = action.data if isinstance(action, trace.Node) else action
            next_obs, reward, termination, truncation, info = self.env.step(control)
            # Check for life loss to implement episodic life
            self.was_real_done = termination or truncation
            lives = info.get('lives')

            if lives is not None and lives < self.lives and lives > 0:
                # Life was lost, terminate the episode
                termination = True
            if lives is not None:
                self.lives = lives

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
