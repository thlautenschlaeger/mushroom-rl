import gym

try:
    import pybullet_envs
    import time
except ImportError:
    pass

from gym import spaces as gym_spaces
from mushroom_rl.environments import Environment, MDPInfo
from mushroom_rl.utils.spaces import *


class Gym(Environment):
    """
    Interface for OpenAI Gym environments. It makes it possible to use every
    Gym environment just providing the id, except for the Atari games that
    are managed in a separate class.

    """
    def __init__(self, name, horizon, gamma, **kwargs):
        """
        Constructor.

        Args:
             name (str): gym id of the environment;
             horizon (int): the horizon;
             gamma (float): the discount factor.

        """
        # MDP creation
        self._close_at_stop = True
        if '- ' + name in pybullet_envs.getList():
            import pybullet
            pybullet.connect(pybullet.DIRECT)
            self._close_at_stop = False

        self.env = gym.make(name)

        # set to 100Hz for pendulum
        if name == 'Pendulum-ID-v1' or name == 'Cartpole-ID-v1':
            self.env._max_episode_steps = 1000
            self.env.unwrapped._dt = 0.01
            self.env.unwrapped._sigma = 1e-4

        self.env._max_episode_steps = np.inf  # Hack to ignore gym time limit.

        # MDP properties
        assert not isinstance(self.env.observation_space,
                              gym_spaces.MultiDiscrete)
        assert not isinstance(self.env.action_space, gym_spaces.MultiDiscrete)

        action_space = self._convert_gym_space(self.env.action_space)
        observation_space = self._convert_gym_space(self.env.observation_space)
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        if isinstance(action_space, Discrete):
            self._convert_action = lambda a: a[0]
        else:
            self._convert_action = lambda a: a

        super().__init__(mdp_info)

    def reset(self, state=None):
        if state is None:
            return self.env.reset()
        else:
            self.env.reset()
            self.env.state = state

            return state

    def step(self, action):
        action = self._convert_action(action)

        return self.env.step(action)

    def render(self, mode='human'):
        self.env.render(mode=mode)

    def stop(self):
        try:
            if self._close_at_stop:
                self.env.close()
        except:
            pass

    @staticmethod
    def _convert_gym_space(space):
        if isinstance(space, gym_spaces.Discrete):
            return Discrete(space.n)
        elif isinstance(space, gym_spaces.Box):
            return Box(low=space.low, high=space.high, shape=space.shape)
        else:
            raise ValueError
