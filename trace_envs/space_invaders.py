import numpy as np
import opto.trace as trace
from opto.trace import bundle
from opto.trace.bundle import ExceptionNode
from opto.trace.errors import ExecutionError
from ocatari.core import OCAtari


class TracedEnv:
    def __init__(self, 
                 env_name="SpaceInvadersNoFrameskip-v4",
                 render_mode=None,
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
    
    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
            self.obs = None
    
    def __del__(self):
        self.close()

    def extract_obj_state(self, objects):
        obs = dict()
        # Count objects by category to create unique keys
        category_counts = {}
        
        for object in objects:
            category = object.category

            if category == "NoObject":
                continue

            # For aliens and other objects that might have multiple instances
            if category in ["Alien", "Shield", "Bullet"]:
                if category not in category_counts:
                    category_counts[category] = 0
                else:
                    category_counts[category] += 1
                
                # Create indexed key for multiple objects of same category
                key = f"{category}{category_counts[category]}"
            else:
                key = category
                
            obs[key] = {"x": object.x,
                        "y": object.y,
                        "w": object.w,
                        "h": object.h,
                        "dx": object.dx,
                        "dy": object.dy,}
        return obs

    @bundle()
    def reset(self):
        """
        Reset the environment and return the initial observation and info.
        """
        _, info = self.env.reset()
        self.obs = self.extract_obj_state(self.env.objects)
        self.obs['reward'] = np.nan
        return self.obs, info
    
    def step(self, action):
        """
        Step the environment with the given action.
        """
        try:
            control = action.data if isinstance(action, trace.Node) else action
            next_obs, reward, termination, truncation, info = self.env.step(control)

            self.obs = self.extract_obj_state(self.env.objects)
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
