import matplotlib.pyplot as plt
import torch
import numpy as np

option_sac = torch.load('option_sac_pendulum100Hz_experiments_tanh.pkl')
sac = torch.load('sac_pendulum100Hz_experiments_tanh.pkl')

os_rewards = np.zeros((len(option_sac[0]['J_results']) - 1))
s_rewards = np.zeros((len(sac[0]['J_results']) - 1))
os_means = np.zeros((len(option_sac[0]['J_results']) - 1))
s_means = np.zeros((len(sac[0]['J_results']) - 1))
os_std = np.zeros((len(option_sac[0]['J_results']) - 1))
s_std = np.zeros((len(sac[0]['J_results']) - 1))

# for i in range(len(option_sac)):
#     for j in range(os_rewards.shape[0]):
#         os_rewards[j] += option_sac[i]['J_results'][j + 1]['J_mean']
#         s_rewards[j] += sac[i]['J_results'][j + 1]['J_mean']

for j in range(os_rewards.shape[0]):
    os_tmp = [option_sac[i]['J_results'][j + 1]['J_mean'] for i in range(len(option_sac))]
    s_tmp = [sac[i]['J_results'][j + 1]['J_mean'] for i in range(len(option_sac))]
    os_means[j] = np.mean(os_tmp)
    s_means[j] = np.mean(s_tmp)
    os_std[j] = np.std(os_tmp)
    s_std[j] = np.std(s_tmp)

# os_rewards /= (len(option_sac[0]['J_results']) - 1)
# s_rewards /= (len(sac[0]['J_results']) - 1)

plt.plot(os_means, color='green')
plt.plot(s_means, color='orange')

plt.show()
