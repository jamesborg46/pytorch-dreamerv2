"""Pickle wrapper for MineRL environments"""
import gym


class MineRLPickler(gym.Wrapper):
    """gym.Env wrapper that allows for pickling of the MineRLEnvironment

    Example:

    Args:

    Raises:

    """

    def __init__(self, env):

        super().__init__(env)

    def __getstate__(self):
        if len(self.unwrapped.instances) == 1:
            instance = self.unwrapped.instances.pop()
            self.unwrapped._TO_MOVE_clean_connection(instance)
            if instance.running:
                instance.kill()
        elif len(self.unwrapped.instances) == 0:
            pass
        else:
            raise Exception()
        return self.__dict__
