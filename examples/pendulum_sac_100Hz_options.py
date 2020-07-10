import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

from mushroom_rl.algorithms.actor_critic import SAC
from mushroom_rl.algorithms.actor_critic.deep_actor_critic.sac import OptionSAC
from mushroom_rl.core import Core
from mushroom_rl.core.core import OptionCore
from mushroom_rl.environments.gym_env import Gym
from mushroom_rl.sds.envs.option_model import OptionSwitchingModel
from mushroom_rl.utils.dataset import compute_J

import mushroom_rl.sds


class CriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action):
        state_action = torch.cat((state.float(), action.float()), dim=1)
        features1 = F.relu(self._h1(state_action))
        features2 = F.relu(self._h2(features1))
        q = self._h3(features2)

        return torch.squeeze(q)


class ActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(ActorNetwork, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state):
        features1 = F.relu(self._h1(torch.squeeze(state, 1).float()))
        features2 = F.relu(self._h2(features1))
        a = self._h3(features2)

        return a


def experiment(alg, n_epochs, n_steps, n_steps_test, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # MDP
    horizon = 1000
    gamma = 0.99
    mdp = Gym('Pendulum-ID-v1', horizon, gamma)
    mdp.seed(seed)
    rarhmm = torch.load(
        os.path.abspath(os.path.join(__file__, '..', '..')) + '/mushroom_rl/sds/envs/hybrid/models/neural_rarhmm_pendulum_cart.pkl',
        map_location='cpu')
    # mdp = Gym('Pendulum-v0', horizon, gamma)

    # Settings
    initial_replay_size = 256
    max_replay_size = 50000 * 4
    batch_size = 512
    n_features = 20
    warmup_transitions = 500
    tau = 0.005
    lr_alpha = 3e-4

    use_cuda = torch.cuda.is_available()

    # Approximator
    actor_input_shape = mdp.info.observation_space.shape
    actor_mu_params = dict(network=ActorNetwork,
                           n_features=n_features,
                           input_shape=actor_input_shape,
                           output_shape=mdp.info.action_space.shape,
                           use_cuda=use_cuda)
    actor_sigma_params = dict(network=ActorNetwork,
                              n_features=n_features,
                              input_shape=actor_input_shape,
                              output_shape=mdp.info.action_space.shape,
                              use_cuda=use_cuda)

    actor_optimizer = {'class': optim.Adam,
                       'params': {'lr': 3e-4}}

    critic_input_shape = (actor_input_shape[0] + mdp.info.action_space.shape[0],)
    critic_params = dict(network=CriticNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': 3e-4}},
                         loss=F.mse_loss,
                         n_features=n_features,
                         input_shape=critic_input_shape,
                         output_shape=(1,),
                         use_cuda=use_cuda)

    # Agent
    agent = alg(mdp.info, actor_mu_params, actor_sigma_params,
                actor_optimizer, critic_params, batch_size, initial_replay_size,
                max_replay_size, warmup_transitions, tau, lr_alpha,
                critic_fit_params=None, rarhmm=rarhmm)

    option_switch_model = OptionSwitchingModel(rarhmm)

    # Algorithm
    core = OptionCore(agent, mdp, option_switch_model=option_switch_model)

    core.learn(n_steps=initial_replay_size, n_steps_per_fit=initial_replay_size)

    # RUN
    dataset = core.evaluate(n_steps=n_steps_test, render=False)
    J = compute_J(dataset, gamma)
    print('J: ', np.mean(J))

    for n in range(n_epochs):
        print('Epoch: ', n)
        core.learn(n_steps=n_steps, n_steps_per_fit=1)
        dataset = core.evaluate(n_steps=n_steps_test, render=False)
        J = compute_J(dataset, gamma)
        print('J: ', np.mean(J))

    print('Press a button to visualize pendulum')
    # input()
    return core.evaluate(n_episodes=15, render=False)


if __name__ == '__main__':
    seed = 69420
    algs = [
        OptionSAC
    ]

    for alg in algs:
        print('Algorithm: ', alg.__name__)
        dataset = experiment(alg=alg, n_epochs=5, n_steps=4000, n_steps_test=2000, seed=seed)
        # dataset = experiment(alg=alg, n_epochs=40, n_steps=5000, n_steps_test=2000)

    import matplotlib.pyplot as plt

    lol = [d[0] for d in dataset[0:1000]]
    plt.plot(lol)
    plt.show()
