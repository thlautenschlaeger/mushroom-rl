import numpy as np
import torch

from mushroom.algorithms.actor_critic import StochasticAC, StochasticAC_AVG
from mushroom.core import Core
from mushroom.environments import *
from mushroom.features import Features
from mushroom.features.tiles import Tiles
from mushroom.approximators import Regressor
from mushroom.approximators.parametric import LinearApproximator
from mushroom.policy import StateLogStdGaussianPolicy
from mushroom.utils.parameters import Parameter


def learn(alg):
    n_steps = 50
    mdp = InvertedPendulum(horizon=n_steps)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    # Agent
    n_tilings = 2
    alpha_r = Parameter(.0001)
    alpha_theta = Parameter(.001 / n_tilings)
    alpha_v = Parameter(.1 / n_tilings)
    tilings = Tiles.generate(n_tilings-1, [1, 1],
                             mdp.info.observation_space.low,
                             mdp.info.observation_space.high + 1e-3)

    phi = Features(tilings=tilings)

    tilings_v = tilings + Tiles.generate(1, [1, 1],
                                         mdp.info.observation_space.low,
                                         mdp.info.observation_space.high + 1e-3)
    psi = Features(tilings=tilings_v)

    input_shape = (phi.size,)

    mu = Regressor(LinearApproximator, input_shape=input_shape,
                   output_shape=mdp.info.action_space.shape)

    std = Regressor(LinearApproximator, input_shape=input_shape,
                    output_shape=mdp.info.action_space.shape)

    std_0 = np.sqrt(1.)
    std.set_weights(np.log(std_0) / n_tilings * np.ones(std.weights_size))

    policy = StateLogStdGaussianPolicy(mu, std)

    if alg is StochasticAC:
        agent = alg(policy, mdp.info, alpha_theta, alpha_v, lambda_par=.5,
                    value_function_features=psi, policy_features=phi)
    elif alg is StochasticAC_AVG:
        agent = alg(policy, mdp.info, alpha_theta, alpha_v, alpha_r,
                    lambda_par=.5, value_function_features=psi,
                    policy_features=phi)

    core = Core(agent, mdp)

    core.learn(n_episodes=2, n_episodes_per_fit=1)

    return policy


def test_stochastic_ac():
    policy = learn(StochasticAC)

    w = policy.get_weights()
    w_test = np.array([-0.0026135, 0.01222979])

    assert np.allclose(w, w_test)


def test_stochastic_ac_avg():
    policy = learn(StochasticAC_AVG)

    w = policy.get_weights()
    w_test = np.array([-0.00295433, 0.01325534])

    assert np.allclose(w, w_test)
