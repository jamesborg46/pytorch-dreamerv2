"""A wrapper for appending state to info for safety envs for gym.Env"""
import gym
import os
from collections import defaultdict
from gym.wrappers.monitoring import video_recorder


class Renderer(gym.Wrapper):
    def __init__(self, env, directory):
        super().__init__(env)

        self.video_enabled = False
        self._video_directory = directory
        self._video_filename = None
        self.video_recorder = None
        self.names = defaultdict(int)
        self.file_prefix = ""

        if not os.path.isdir(self._video_directory):
            os.makedirs(self._video_directory)

    def enable_rendering(self, rendering_enabled, file_prefix=""):
        self.video_enabled = rendering_enabled
        self.file_prefix = file_prefix

        if not rendering_enabled and self.video_recorder:
            self.video_recorder.close()

    def step(self, action):

        obs, reward, done, info = self.env.step(action)
        if self.video_enabled:
            self.video_recorder.capture_frame()

        # Sets filename on the first step then must continue to attach Nones
        # to get a full batch of entries, Otherwise the garage EpisodeBatch
        # data-structure would complain
        info['video_filename'] = None
        if self._video_filename is not None:
            info['video_filename'] = os.path.join(
                self._video_directory,
                self._video_filename + '.mp4'
            )
            self._video_filename = None

        return obs, reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        if self.video_enabled:
            self.reset_video_recorder()
        return observation

    def close_renderer(self):
        self.video_enabled = False
        if self.video_recorder:
            self.video_recorder.close()
            self.video_recorder = None

    def close(self):
        super().close()
        self.close_renderer()

    def __del__(self):
        self.close()

    def reset_video_recorder(self):
        # Close any existing video recorder
        if self.video_recorder:
            self.video_recorder.close()

        name = self.file_prefix
        skill = self.metadata.get('skill')
        if skill is not None:
            name = name + f'_skill_{skill:02}_'
        self.names[name] += 1

        ep_id = self.names[name]
        self._video_filename = name + f"id_{ep_id:002}"

        # Start recording the next video.
        self.video_recorder = video_recorder.VideoRecorder(
            env=self.env,
            base_path=os.path.join(
                self._video_directory,
                self._video_filename,
            ),
            metadata=self.metadata,
            enabled=self.video_enabled,
        )
        self.video_recorder.capture_frame()
