import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from tqdm import tqdm, trange

from mushroom.core import Core
from mushroom.environments import Gym
from mushroom.algorithms.actor_critic import A2C

from mushroom.policy import GaussianTorchPolicy
from mushroom.utils.dataset import compute_J


class Network(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(Network, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state, **kwargs):
        features1 = torch.tanh(self._h1(torch.squeeze(state, 1).float()))
        features2 = torch.tanh(self._h2(features1))
        a = self._h3(features2)

        return a


def experiment(alg, env_id, horizon, gamma, n_epochs, n_steps, n_steps_per_fit, n_step_test,
               alg_params, policy_params):
    print(alg.__name__)

    mdp = Gym(env_id, horizon, gamma)

    critic_params = dict(network=Network,
                         optimizer={'class': optim.RMSprop,
                                    'params': {'lr': 7e-4,
                                               'eps': 1e-5}},
                         loss=F.mse_loss,
                         n_features=64,
                         input_shape=mdp.info.observation_space.shape,
                         output_shape=(1,))

    policy = GaussianTorchPolicy(Network,
                                 mdp.info.observation_space.shape,
                                 mdp.info.action_space.shape,
                                 **policy_params)

    agent = alg(mdp.info, policy, critic_params, **alg_params)

    core = Core(agent, mdp)

    for it in trange(n_epochs):
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit)
        dataset = core.evaluate(n_steps=n_step_test, render=False)

        J = np.mean(compute_J(dataset, mdp.info.gamma))
        R = np.mean(compute_J(dataset))
        E = agent.policy.entropy()

        tqdm.write('END OF EPOCH ' + str(it))
        tqdm.write('J: {}, R: {}, entropy: {}'.format(J, R, E))
        tqdm.write('##################################################################################################')

    print('Press a button to visualize')
    input()
    core.evaluate(n_episodes=5, render=True)


if __name__ == '__main__':
    policy_params = dict(
        std_0=1.,
        n_features=64,
        use_cuda=False
    )

    a2c_params = dict(actor_optimizer={'class': optim.RMSprop,
                                       'params': {'lr': 7e-4,
                                                  'eps': 3e-3}},
                      max_grad_norm=0.5,
                      ent_coeff=0.01)

    algs_params = [
        (A2C, 'a2c', a2c_params)
     ]

    for alg, alg_name, alg_params in algs_params:
        experiment(alg=alg, env_id='Pendulum-v0', horizon=200, gamma=.99,
                   n_epochs=40, n_steps=30000, n_steps_per_fit=5,
                   n_step_test=5000, alg_params=alg_params,
                   policy_params=policy_params)
