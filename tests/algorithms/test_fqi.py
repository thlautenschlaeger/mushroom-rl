import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

import shutil
from datetime import datetime
from helper.utils import TestUtils as tu

from mushroom_rl.algorithms import Agent
from mushroom_rl.algorithms.value import DoubleFQI, FQI
from mushroom_rl.core import Core
from mushroom_rl.environments import *
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.utils.parameters import Parameter


def learn(alg, alg_params):
    mdp = CarOnHill()
    np.random.seed(1)

    # Policy
    epsilon = Parameter(value=1.)
    pi = EpsGreedy(epsilon=epsilon)

    # Approximator
    approximator_params = dict(input_shape=mdp.info.observation_space.shape,
                               n_actions=mdp.info.action_space.n,
                               n_estimators=50,
                               min_samples_split=5,
                               min_samples_leaf=2)
    approximator = ExtraTreesRegressor

    # Agent
    agent = alg(mdp.info, pi, approximator,
                approximator_params=approximator_params, **alg_params)

    # Algorithm
    core = Core(agent, mdp)

    # Train
    core.learn(n_episodes=5, n_episodes_per_fit=5)

    test_epsilon = Parameter(0.75)
    agent.policy.set_epsilon(test_epsilon)
    dataset = core.evaluate(n_episodes=2)

    return agent, np.mean(compute_J(dataset, mdp.info.gamma))


def test_fqi():
    params = dict(n_iterations=10)
    _, j = learn(FQI, params)
    j_test = -0.0874123073618985

    assert j == j_test


def test_fqi_save():
    params = dict(n_iterations=10)
    agent_save, _ = learn(FQI, params)

    agent_path = './agentdir{}/'.format(datetime.now().strftime("%H%M%S%f"))

    agent_save.save(agent_path)
    agent_load = Agent.load(agent_path)

    shutil.rmtree(agent_path)

    for att, method in agent_save.__dict__.items():
        save_attr = getattr(agent_save, att)
        load_attr = getattr(agent_load, att)
        #print('{}: {}'.format(att, type(save_attr)))

        tu.assert_eq(save_attr, load_attr)


def test_fqi_boosted():
    params = dict(n_iterations=10, boosted=True)
    _, j = learn(FQI, params)
    j_test = -0.09201295511778791

    assert j == j_test


def test_fqi_boosted_save():
    params = dict(n_iterations=10, boosted=True)
    agent_save, _ = learn(FQI, params)

    agent_path = './agentdir{}/'.format(datetime.now().strftime("%H%M%S%f"))

    agent_save.save(agent_path)
    agent_load = Agent.load(agent_path)

    shutil.rmtree(agent_path)

    for att, method in agent_save.__dict__.items():
        save_attr = getattr(agent_save, att)
        load_attr = getattr(agent_load, att)
        #print('{}: {}'.format(att, type(save_attr)))

        tu.assert_eq(save_attr, load_attr)


def test_double_fqi():
    params = dict(n_iterations=10)
    _, j = learn(DoubleFQI, params)
    j_test = -0.19933233708925654

    assert j == j_test


def test_double_fqi_save():
    params = dict(n_iterations=10)
    agent_save, _ = learn(DoubleFQI, params)

    agent_path = './agentdir{}/'.format(datetime.now().strftime("%H%M%S%f"))

    agent_save.save(agent_path)
    agent_load = Agent.load(agent_path)

    shutil.rmtree(agent_path)

    for att, method in agent_save.__dict__.items():
        save_attr = getattr(agent_save, att)
        load_attr = getattr(agent_load, att)
        #print('{}: {}'.format(att, type(save_attr)))

        tu.assert_eq(save_attr, load_attr)
