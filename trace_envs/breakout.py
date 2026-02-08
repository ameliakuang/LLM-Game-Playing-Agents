import numpy as np
import opto.trace as trace
from opto.trace import bundle
from opto.trace.bundle import ExceptionNode
from opto.trace.errors import ExecutionError
from ocatari.core import OCAtari


class TracedEnv:
    def __init__(self, 
                 env_name="BreakoutNoFrameskip-v4",
                 render_mode="rgb_array",
                 obs_mode="ori",
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

    def extract_game_state(self, objects, rgb, info):
        obs = dict()
        color_blocks = {
            "Red": [], "Orange": [], "Yellow": [],
            "Green": [], "Aqua": [], "Blue": []
        }
        for object in objects:
            if object.category == "NoObject":
                continue
            elif object.category == "Block":
                color = None
                if object.y == 57: color = "Red"
                elif object.y == 63: color = "Orange"
                elif object.y == 69: color = "Yellow"
                elif object.y == 75: color = "Green"
                elif object.y == 81: color = "Aqua"
                elif object.y == 87: color = "Blue"
                else: continue  # Skip unknown y-positions
            
                color_blocks[color].append({
                    "x": object.x,
                    "y": object.y,
                    "w": object.w,
                    "h": object.h,
                })
            else:
                obs[object.category] = {"x": object.x,
                                        "y": object.y,
                                        "w": object.w,
                                        "h": object.h,
                                        "dx": object.dx,
                                        "dy": object.dy,}
        for color, blocks in color_blocks.items():
            if blocks:
                obs[f"{color[0]}B"] = blocks
        if info:
            obs['lives'] = info.get('lives', None)
        return obs


    @bundle()
    def reset(self, seed=None, options=None):
        """
        Reset the environment and return the initial observation and info.

        Combines EpisodicLifeEnv + FireResetEnv logic (see openai/baselines):
          Phase 1 (EpisodicLifeEnv): On true game-over, do a real env.reset().
                  On life loss, just NOOP to advance past the death frame.
          Phase 2 (FireResetEnv):    FIRE to launch the ball, every time.
        """
        # Phase 1 – EpisodicLifeEnv
        if self.was_real_done:
            obs, info = self.env.reset(seed=seed, options=options)
        else:
            # NOOP to advance past the death frame (standard EpisodicLifeEnv)
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset(seed=seed, options=options)
        self.lives = self.env._env.unwrapped.ale.lives()

        # Phase 2 – FireResetEnv: launch the ball
        obs, _, terminated, truncated, info = self.env.step(1)  # FIRE
        if terminated or truncated:
            obs, info = self.env.reset(seed=seed, options=options)
        self.lives = self.env._env.unwrapped.ale.lives()

        self.obs = self.extract_game_state(self.env.objects, obs, info)
        self.obs['reward'] = np.nan

        return self.obs, info
    
    
    def step(self, action):
        try:
            control = action.data if isinstance(action, trace.Node) else action
            next_obs, reward, termination, truncation, info = self.env.step(control)
            # Check for life loss to implement episodic life
            self.was_real_done = termination or truncation
            lives = info.get('lives')
            
            if lives < self.lives and lives > 0:
                # Life was lost, terminate the episode
                termination = True
            self.lives = lives

            self.obs = self.extract_game_state(self.env.objects, next_obs, info)
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
