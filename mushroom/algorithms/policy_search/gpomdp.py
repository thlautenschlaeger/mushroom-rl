import numpy as np
from mushroom.algorithms.policy_search import PolicyGradient


class GPOMDP(PolicyGradient):
    """
    GPOMDP algorithm.
    "Infinite-Horizon Policy-Gradient Estimation", Baxter J., Bartlett P. L.  2001.

    """
    def __init__(self, policy, mdp_info, params, features):
        self.__name__ = 'GPOMDP'

        super(GPOMDP, self).__init__(policy, mdp_info, params, features)

        self.sum_d_log_pi = None
        self.list_sum_d_log_pi = list()
        self.list_sum_d_log_pi_ep = list()

        self.list_reward = list()
        self.list_reward_ep = list()

        self.baseline_num = list()
        self.baseline_den = list()

        self.step_count = 0

        # Ignore divide by zero
        np.seterr(divide='ignore', invalid='ignore')

    def _compute_gradient(self, J):
        gradient = np.zeros(self.policy.weights_size)

        n_episodes = len(self.list_sum_d_log_pi_ep)

        for i in xrange(n_episodes):
            list_sum_d_log_pi = self.list_sum_d_log_pi_ep[i]
            list_reward = self.list_reward_ep[i]

            n_steps = len(list_sum_d_log_pi)

            for t in xrange(n_steps):
                step_grad = list_sum_d_log_pi[t]
                step_reward = list_reward[t]
                baseline = self.baseline_num[t]/ self.baseline_den[t]
                baseline[np.logical_not(np.isfinite(baseline))] = 0.
                gradient += (step_reward - baseline)*step_grad

        gradient /= n_episodes

        self.list_reward_ep = list()
        self.list_sum_d_log_pi_ep = list()

        self.baseline_num = list()
        self.baseline_den = list()

        return gradient,

    def _step_update(self, x, u, r):
        discounted_reward = self.df*r
        self.list_reward.append(discounted_reward)

        d_log_pi = self.policy.diff_log(x, u)
        self.sum_d_log_pi += d_log_pi

        self.list_sum_d_log_pi.append(self.sum_d_log_pi)

        squared_sum_d_log_pi = np.square(self.sum_d_log_pi)

        if self.step_count < len(self.baseline_num):
            self.baseline_num[self.step_count] += discounted_reward * squared_sum_d_log_pi
            self.baseline_den[self.step_count] += squared_sum_d_log_pi
        else:
            self.baseline_num.append(discounted_reward * squared_sum_d_log_pi)
            self.baseline_den.append(squared_sum_d_log_pi)

        self.step_count += 1

    def _episode_end_update(self):
        self.list_reward_ep.append(self.list_reward)
        self.list_reward = list()

        self.list_sum_d_log_pi_ep.append(self.list_sum_d_log_pi)
        self.list_sum_d_log_pi = list()

    def _init_update(self):
        self.sum_d_log_pi = np.zeros(self.policy.weights_size)
        self.list_sum_d_log_pi = list()
        self.step_count = 0
