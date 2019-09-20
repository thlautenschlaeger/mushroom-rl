import numpy as np

from mushroom.algorithms.value.td import TD
from mushroom.approximators import Regressor
from mushroom.approximators.parametric import LinearApproximator
from mushroom.features import get_action_features


class SARSALambdaContinuous(TD):
    """
    Continuous version of SARSA(lambda) algorithm.

    """
    def __init__(self, approximator, policy, mdp_info, learning_rate,
                 lambda_coeff, features, approximator_params=None):
        """
        Constructor.

        Args:
            lambda_coeff (float): eligibility trace coefficient.

        """
        self._approximator_params = dict() if approximator_params is None else \
            approximator_params

        self.Q = Regressor(approximator, **self._approximator_params)
        self.e = np.zeros(self.Q.weights_size)
        self._lambda = lambda_coeff

        super().__init__(self.Q, policy, mdp_info, learning_rate, features)

    def _update(self, state, action, reward, next_state, absorbing):
        phi_state = self.phi(state)
        q_current = self.Q.predict(phi_state, action)

        alpha = self.alpha(state, action)

        self.e = self.mdp_info.gamma * self._lambda * self.e + self.Q.diff(
            phi_state, action)

        self.next_action = self.draw_action(next_state)
        phi_next_state = self.phi(next_state)
        q_next = self.Q.predict(phi_next_state,
                                self.next_action) if not absorbing else 0.

        delta = reward + self.mdp_info.gamma * q_next - q_current

        theta = self.Q.get_weights()
        theta += alpha * delta * self.e
        self.Q.set_weights(theta)

    def episode_start(self):
        self.e = np.zeros(self.Q.weights_size)

        super().episode_start()


class TrueOnlineSARSALambda(TD):
    """
    True Online SARSA(lambda) with linear function approximation.
    "True Online TD(lambda)". Seijen H. V. et al.. 2014.

    """
    def __init__(self, policy, mdp_info, learning_rate, lambda_coeff,
                 features, approximator_params=None):
        """
        Constructor.

        Args:
            lambda_coeff (float): eligibility trace coefficient.

        """
        self._approximator_params = dict() if approximator_params is None else \
            approximator_params

        self.Q = Regressor(LinearApproximator, **self._approximator_params)
        self.e = np.zeros(self.Q.weights_size)
        self._lambda = lambda_coeff
        self._q_old = None

        super().__init__(self.Q, policy, mdp_info, learning_rate, features)

    def _update(self, state, action, reward, next_state, absorbing):
        phi_state = self.phi(state)
        phi_state_action = get_action_features(phi_state, action,
                                               self.mdp_info.action_space.n)
        q_current = self.Q.predict(phi_state, action)

        if self._q_old is None:
            self._q_old = q_current

        alpha = self.alpha(state, action)

        e_phi = self.e.dot(phi_state_action)
        self.e = self.mdp_info.gamma * self._lambda * self.e + alpha * (
            1. - self.mdp_info.gamma * self._lambda * e_phi) * phi_state_action

        self.next_action = self.draw_action(next_state)
        phi_next_state = self.phi(next_state)
        q_next = self.Q.predict(phi_next_state,
                                self.next_action) if not absorbing else 0.

        delta = reward + self.mdp_info.gamma * q_next - self._q_old

        theta = self.Q.get_weights()
        theta += delta * self.e + alpha * (
            self._q_old - q_current) * phi_state_action
        self.Q.set_weights(theta)

        self._q_old = q_next

    def episode_start(self):
        self._q_old = None
        self.e = np.zeros(self.Q.weights_size)

        super().episode_start()