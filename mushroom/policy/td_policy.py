import numpy as np

from mushroom.utils.parameters import Parameter


class EpsGreedy:
    """
    Epsilon greedy policy.

    """
    def __init__(self, epsilon, observation_space, action_space):
        """
        Constructor.

        Args:
        epsilon (Parameter): the exploration coefficient. It indicates
            the probability of performing a random actions in the current step;
        observation_space (object): the state space;
        action_space (object): the action_space.

        """
        self.__name__ = 'EpsGreedy'

        assert isinstance(epsilon, Parameter)

        self._epsilon = epsilon
        self.observation_space = observation_space
        self.action_space = action_space

        self._approximator = None

    def __call__(self, state):
        """
        Compute an action according to the policy.

        Args:
            state (np.array): the state where the agent is.

        Returns:
            The selected action.

        """
        if not np.random.uniform() < self._epsilon(state):
            q = self._approximator.predict(np.expand_dims(state, axis=0))
            max_a = np.argmax(q, axis=1)

            return max_a

        return self.action_space.sample()

    def set_epsilon(self, epsilon):
        """
        Setter.

        Args:
            epsilon (Parameter): the exploration coefficient. It indicates the
            probability of performing a random actions in the current step.

        """
        assert isinstance(epsilon, Parameter)

        self._epsilon = epsilon

    def update(self, *idx):
        """
        Update the value of the epsilon parameter (e.g. in case of different
        values of epsilon for each visited state according to the number of
        visits).

        Args:
            idx (int): value to use to update epsilon.

        """
        self._epsilon.update(*idx)

    def set_q(self, approximator):
        """
        Args:
            approximator (object): the approximator to use.

        """
        self._approximator = approximator

    def get_q(self):
        """
        Returns:
             the approximator used by the policy.

        """
        return self._approximator

    def __str__(self):
        return self.__name__
