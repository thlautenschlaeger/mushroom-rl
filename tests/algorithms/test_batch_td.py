import numpy as np
import shutil
from datetime import datetime
from helper.utils import TestUtils as tu

from mushroom_rl.algorithms import Agent
from mushroom_rl.algorithms.value import LSPI
from mushroom_rl.core import Core
from mushroom_rl.environments import *
from mushroom_rl.features import Features
from mushroom_rl.features.basis import PolynomialBasis
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.parameters import Parameter


def learn_lspi():
    mdp = CartPole()
    np.random.seed(1)

    # Policy
    epsilon = Parameter(value=1.)
    pi = EpsGreedy(epsilon=epsilon)

    # Agent
    basis = [PolynomialBasis()]
    features = Features(basis_list=basis)

    approximator_params = dict(input_shape=(features.size,),
                               output_shape=(mdp.info.action_space.n,),
                               n_actions=mdp.info.action_space.n)
    agent = LSPI(mdp.info, pi, fit_params=dict(),
                 approximator_params=approximator_params, features=features)

    # Algorithm
    core = Core(agent, mdp)

    # Train
    core.learn(n_episodes=100, n_episodes_per_fit=100)
    return agent


def test_lspi():

    w = learn_lspi().approximator.get_weights()
    w_test = np.array([-2.23880597, -2.27427603, -2.25])

    assert np.allclose(w, w_test)


def test_lspi_save():

    agent_path = './agentdir{}/'.format(datetime.now().strftime("%H%M%S%f"))

    agent_save = learn_lspi()

    agent_save.save(agent_path)
    agent_load = Agent.load(agent_path)

    shutil.rmtree(agent_path)

    for att, method in agent_save.__dict__.items():
        save_attr = getattr(agent_save, att)
        load_attr = getattr(agent_load, att)
        #print('{}: {}'.format(att, type(save_attr)))

        tu.assert_eq(save_attr, load_attr)
