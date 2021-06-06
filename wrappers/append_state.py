"""A wrapper for appending state to info for safety envs for gym.Env"""
import gym
from mujoco_py import MjSim, load_model_from_xml


class SafetyEnvStateAppender(gym.Wrapper):
    def __init__(self, env, capture_state=False):
        super().__init__(env)
        self.capture_state = capture_state

    def set_capture_state(self, capture_state):
        self.capture_state = capture_state

    def reset(self):
        obs = self.env.reset()
        self.model_xml = self.model.get_xml()
        return obs

    def step(self, action):

        if self.capture_state:
            state = self.env.world.sim.get_state()

        obs, reward, done, info = self.env.step(action)

        if self.capture_state:
            info['state'] = state.flatten()

        info['model_xml'] = self.model_xml
        if self.model_xml is not None:
            self.model_xml = None

        return obs, reward, done, info

    def load_model(self, model_xml):
        self.env.world.sim = MjSim(load_model_from_xml(model_xml))

    def render_state(self,
                     state,
                     overlay_bl=None,
                     overlay_tl=None,
                     overlay_tr=None):
        self.env.world.sim.set_state_from_flattened(state)
        self.env.sim.forward()
        self.env.render_lidar_markers = False
        rgb_array = self.env.render('rgb_array',
                                    overlay_bl=overlay_bl,
                                    overlay_tl=overlay_tl,
                                    overlay_tr=overlay_tr,
                                    )
        return rgb_array
