from mushroom_rl.algorithms.value.td import TD
from mushroom_rl.utils.table import Table


class SARSA(TD):
    """
    SARSA algorithm.

    """
    def __init__(self, mdp_info, policy, learning_rate):
        self.Q = Table(mdp_info.size)
        super().__init__(mdp_info, policy, self.Q, learning_rate)

    def _update(self, state, action, reward, next_state, absorbing):
        q_current = self.Q[state, action]

        self.next_action = self.draw_action(next_state)
        q_next = self.Q[next_state, self.next_action] if not absorbing else 0.

        self.Q[state, action] = q_current + self.alpha(state, action) * (
            reward + self.mdp_info.gamma * q_next - q_current)
