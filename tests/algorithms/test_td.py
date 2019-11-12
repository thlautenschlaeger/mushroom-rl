import numpy as np

from mushroom.algorithms.value import *
from mushroom.approximators.parametric import LinearApproximator
from mushroom.core import Core
from mushroom.environments.grid_world import GridWorld
from mushroom.environments.gym_env import Gym
from mushroom.features import Features
from mushroom.features.tiles import Tiles
from mushroom.policy.td_policy import EpsGreedy
from mushroom.utils.parameters import Parameter


def initialize():
    np.random.seed(1)
    return EpsGreedy(Parameter(1)), GridWorld(2, 2, start=(0, 0), goal=(1, 1)),\
           Gym(name='MountainCar-v0', horizon=np.inf, gamma=1.)


def test_q_learning():
    pi, mdp, _ = initialize()
    agent = QLearning(pi, mdp.info, Parameter(.5))

    core = Core(agent, mdp)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    test_q = np.array([[7.82042542, 8.40151978, 7.64961548, 8.82421875],
                       [8.77587891, 9.921875, 7.29316406, 8.68359375],
                       [7.7203125, 7.69921875, 4.5, 9.84375],
                       [0., 0., 0., 0.]])

    assert np.allclose(agent.Q.table, test_q)


def test_double_q_learning():
    pi, mdp, _ = initialize()
    agent = DoubleQLearning(pi, mdp.info, Parameter(.5))

    core = Core(agent, mdp)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    test_q_0 = np.array([[2.6578125, 6.94757812, 3.73359375, 7.171875],
                         [2.25, 7.5, 3.0375, 3.375],
                         [3.0375, 5.4140625, 2.08265625, 8.75],
                         [0., 0., 0., 0.]])
    test_q_1 = np.array([[2.72109375, 4.5, 4.36640625, 6.609375],
                         [4.5, 9.375, 4.49296875, 4.5],
                         [1.0125, 5.0625, 5.625, 8.75],
                         [0., 0., 0., 0.]])

    assert np.allclose(agent.Q[0].table, test_q_0)
    assert np.allclose(agent.Q[1].table, test_q_1)


def test_weighted_q_learning():
    pi, mdp, _ = initialize()
    agent = WeightedQLearning(pi, mdp.info, Parameter(.5))

    core = Core(agent, mdp)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    test_q = np.array([[7.1592415, 4.07094744, 7.10518702, 8.5467274],
                       [8.08689916, 9.99023438, 5.77871216, 7.51059129],
                       [6.52294537, 0.86087671, 3.70431496, 9.6875],
                       [0., 0., 0., 0.]])

    assert np.allclose(agent.Q.table, test_q)


def test_speedy_q_learning():
    pi, mdp, _ = initialize()
    agent = SpeedyQLearning(pi, mdp.info, Parameter(.5))

    core = Core(agent, mdp)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    test_q = np.array([[7.82042542, 8.40151978, 7.64961548, 8.82421875],
                       [8.77587891, 9.921875, 7.29316406, 8.68359375],
                       [7.7203125, 7.69921875, 4.5, 9.84375],
                       [0., 0., 0., 0.]])

    assert np.allclose(agent.Q.table, test_q)


def test_sarsa():
    pi, mdp, _ = initialize()
    agent = SARSA(pi, mdp.info, Parameter(.1))

    core = Core(agent, mdp)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    test_q = np.array([[4.31368701e-2, 3.68037689e-1, 4.14040445e-2, 1.64007642e-1],
                       [6.45491436e-1, 4.68559000, 8.07603735e-2, 1.67297938e-1],
                       [4.21445838e-2, 3.71538042e-3, 0., 3.439],
                       [0., 0., 0., 0.]])

    assert np.allclose(agent.Q.table, test_q)


def test_sarsa_lambda_discrete():
    pi, mdp, _ = initialize()
    agent = SARSALambda(pi, mdp.info, Parameter(.1), .9)

    core = Core(agent, mdp)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    test_q = np.array([[1.88093529, 2.42467354, 1.07390687, 2.39288988],
                       [2.46058746, 4.68559, 1.5661933, 2.56586018],
                       [1.24808966, 0.91948465, 0.47734152, 3.439],
                       [0., 0., 0., 0.]])

    assert np.allclose(agent.Q.table, test_q)


def test_sarsa_lambda_continuous():
    pi, _, mdp_continuous = initialize()
    mdp_continuous.seed(1)
    n_tilings = 1
    tilings = Tiles.generate(n_tilings, [2, 2],
                             mdp_continuous.info.observation_space.low,
                             mdp_continuous.info.observation_space.high)
    features = Features(tilings=tilings)

    approximator_params = dict(
        input_shape=(features.size,),
        output_shape=(mdp_continuous.info.action_space.n,),
        n_actions=mdp_continuous.info.action_space.n
    )
    agent = SARSALambdaContinuous(LinearApproximator, pi, mdp_continuous.info,
                                  Parameter(.1), .9, features=features,
                                  approximator_params=approximator_params)

    core = Core(agent, mdp_continuous)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    test_w = np.array([-16.38428419, 0., -14.31250136, 0., -15.68571525, 0.,
                       -10.15663821, 0., -15.0545445, 0., -8.3683605, 0.])

    assert np.allclose(agent.Q.get_weights(), test_w)


def test_expected_sarsa():
    pi, mdp, _ = initialize()
    agent = ExpectedSARSA(pi, mdp.info, Parameter(.1))

    core = Core(agent, mdp)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    test_q = np.array([[0.10221208, 0.48411449, 0.07688765, 0.64002317],
                       [0.58525881, 5.217031, 0.06047094, 0.48214145],
                       [0.08478224, 0.28873536, 0.06543094, 4.68559],
                       [0., 0., 0., 0.]])

    assert np.allclose(agent.Q.table, test_q)


def test_true_online_sarsa_lambda():
    pi, _, mdp_continuous = initialize()
    mdp_continuous.seed(1)
    n_tilings = 1
    tilings = Tiles.generate(n_tilings, [2, 2],
                             mdp_continuous.info.observation_space.low,
                             mdp_continuous.info.observation_space.high)
    features = Features(tilings=tilings)

    approximator_params = dict(
        input_shape=(features.size,),
        output_shape=(mdp_continuous.info.action_space.n,),
        n_actions=mdp_continuous.info.action_space.n
    )
    agent = TrueOnlineSARSALambda(pi, mdp_continuous.info,
                                  Parameter(.1), .9, features=features,
                                  approximator_params=approximator_params)

    core = Core(agent, mdp_continuous)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    test_w = np.array([-17.27410736, 0., -15.04386343, 0., -16.6551805, 0.,
                       -11.31383707, 0., -16.11782002, 0., -9.6927357, 0.])

    assert np.allclose(agent.Q.get_weights(), test_w)


def test_r_learning():
    pi, mdp, _ = initialize()
    agent = RLearning(pi, mdp.info, Parameter(.1), Parameter(.5))

    core = Core(agent, mdp)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    test_q = np.array([[-6.19137991, -3.9368055, -5.11544257, -3.43673781],
                       [-2.52319391, 1.92201829, -2.77602918, -2.45972955],
                       [-5.38824415, -2.43019918, -1.09965936, 2.04202511],
                       [0., 0., 0., 0.]])

    assert np.allclose(agent.Q.table, test_q)


def test_rq_learning():
    pi, mdp, _ = initialize()

    agent = RQLearning(pi, mdp.info, Parameter(.1), beta=Parameter(.5))

    core = Core(agent, mdp)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    test_q = np.array([[0.32411217, 2.9698436, 0.46474438, 1.10269504],
                       [2.99505139, 5.217031, 0.40933461, 0.37687883],
                       [0.41942675, 0.32363486, 0., 4.68559],
                       [0., 0., 0., 0.]])

    assert np.allclose(agent.Q.table, test_q)

    agent = RQLearning(pi, mdp.info, Parameter(.1), delta=Parameter(.5))

    core = Core(agent, mdp)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    test_q = np.array([[1.04081115e-2, 5.14662188e-1, 1.73951634e-2, 1.24081875e-01],
                       [0., 2.71, 1.73137500e-4, 4.10062500e-6],
                       [0., 4.50000000e-2, 0., 4.68559],
                       [0., 0., 0., 0.]])

    assert np.allclose(agent.Q.table, test_q)

    agent = RQLearning(pi, mdp.info, Parameter(.1), off_policy=True,
                       beta=Parameter(.5))

    core = Core(agent, mdp)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    test_q = np.array([[3.55204022, 4.54235939, 3.42601165, 2.95170908],
                       [2.73877031, 3.439, 2.42031528, 2.86634531],
                       [3.43274708, 3.8592342, 3.72637395, 5.217031],
                       [0., 0., 0., 0.]])

    assert np.allclose(agent.Q.table, test_q)

    agent = RQLearning(pi, mdp.info, Parameter(.1), off_policy=True,
                       delta=Parameter(.5))

    core = Core(agent, mdp)

    # Train
    core.learn(n_steps=100, n_steps_per_fit=1, quiet=True)

    test_q = np.array([[0.18947806, 1.57782254, 0.21911489, 1.05197011],
                       [0.82309759, 5.217031, 0.04167492, 0.61472604],
                       [0.23620541, 0.59828262, 1.25299991, 5.217031],
                       [0., 0., 0., 0.]])

    assert np.allclose(agent.Q.table, test_q)
