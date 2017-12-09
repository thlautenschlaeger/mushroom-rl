import numpy as np

from mushroom.environments.finite_mdp import FiniteMDP


def generate_simple_chain(state_n, goal_states, prob, rew, mu=None, gamma=.9):
    """
    Simple chain generator.

    Args:
        state_n (int): number of states;
        goal_states (list): list of goal states;
        prob (float): probability of success of an action;
        rew (float): reward obtained in goal states;
        mu (np.ndarray): initial state probability distribution;
        gamma (float): discount factor.

    Returns:
        a FiniteMDP object built with the provided parameters.

    """
    p = compute_probabilities(state_n, prob)
    r = compute_reward(state_n, goal_states, rew)

    return FiniteMDP(p, r, mu, gamma)


def compute_probabilities(state_n, prob):
    """
    Compute the transition probability matrix.

    Args:
        state_n (int): number of states;
        prob (float): probability of success of an action.

    Returns:
        the transition probability matrix;

    """
    p = np.zeros((state_n, 2, state_n))

    for i in xrange(state_n):
        if i == 0:
            p[i, 1, i] = 1.0
        else:
            p[i, 1, i] = 1.0 - prob
            p[i, 1, i - 1] = prob

        if i == state_n - 1:
            p[i, 0, i] = 1.0
        else:
            p[i, 0, i] = 1.0 - prob
            p[i, 0, i + 1] = prob

    return p


def compute_reward(state_n, goal_states, rew):
    """
    Compute the reward matrix.

    Args:
        state_n (int): number of states;
        goal_states (list): list of goal states;
        rew (float): reward obtained in goal states.

    Returns:
        the reward matrix.

    """
    r = np.zeros((state_n, 2, state_n))

    for g in goal_states:
        if g != 0:
            r[g - 1, 0, g] = rew

        if g != state_n - 1:
            r[g + 1, 1, g] = rew

    return r
