class MDPInfo:
    """
    This class is used to store the information of the environment.

    """
    def __init__(self, observation_space, action_space, gamma, horizon):
        """
        Constructor.

        Args:
             observation_space ([Box, Discrete]): the state space;
             action_space ([Box, Discrete]): the action space;
             gamma (float): the discount factor;
             horizon (int): the horizon.

        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.gamma = gamma
        self.horizon = horizon

    @property
    def size(self):
        """
        Returns:
            The sum of the number of discrete states and discrete actions. Only
            works for discrete spaces.

        """
        return self.observation_space.size + self.action_space.size

    @property
    def shape(self):
        """
        Returns:
            The concatenation of the shape tuple of the state and action
            spaces.

        """
        return self.observation_space.shape + self.action_space.shape


class Environment(object):
    def __init__(self, mdp_info):
        """
        Constructor.

        Args:
             mdp_info (MDPInfo): an object containing the info of the
                environment.

        """
        # MDP initialization
        self.reset()

        self._mdp_info = mdp_info

    def seed(self, seed):
        """
        Set the seed of the environment.

        Args:
            seed (float): the value of the seed.

        """
        if hasattr(self, 'env'):
            self.env.seed(seed)
        else:
            raise NotImplementedError

    def reset(self, state=None):
        """
        Reset the current state.

        Args:
            state (np.ndarray, None): the state to set to the current state.

        Returns:
            The current state.

        """
        raise NotImplementedError

    def step(self, action):
        """
        Move the agent from its current state according to the action.

        Args:
            action (np.ndarray): the action to execute.

        Returns:
            The state reached by the agent executing `action` in its current
            state.

        """
        raise NotImplementedError

    def render(self, mode='human', close=False):
        raise NotImplementedError

    @property
    def info(self):
        """
        Returns:
             An object containing the info of the environment.

        """
        return self._mdp_info

    def __str__(self):
        return self.__name__
