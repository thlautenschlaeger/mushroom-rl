import matplotlib.pyplot as plt
import torch
import numpy as np

option_sac = torch.load('option_sac_cartpole100Hz_experiments.pkl')
sac = torch.load('sac_cartpole100Hz_experiments.pkl')

os_rewards = np.zeros((len(option_sac[0]['J_results'])))
s_rewards = np.zeros((len(sac[0]['J_results'])))

for i in range(len(option_sac)):
    for j in range(os_rewards.shape[0]):
        os_rewards[j] += option_sac[i]['J_results'][j]['J_mean']
        s_rewards[j] += sac[i]['J_results'][j]['J_mean']

os_rewards /= len(option_sac[0]['J_results'])
s_rewards /= len(sac[0]['J_results'])

plt.plot(os_rewards, color='green')
plt.plot(s_rewards, color='orange')

plt.show()
