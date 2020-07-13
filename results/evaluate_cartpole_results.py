import matplotlib.pyplot as plt
import torch
import numpy as np
import seaborn as sns

# sns.set()

option_sac = torch.load('option_sac_cartpole100Hz_experiments.pkl')
sac = torch.load('sac_cartpole100Hz_experiments.pkl')

os_means = np.zeros((len(option_sac[0]['J_results'])))
s_means = np.zeros((len(sac[0]['J_results'])))
os_std = np.zeros((len(option_sac[0]['J_results'])))
s_std = np.zeros((len(sac[0]['J_results'])))

for j in range(os_means.shape[0]):
    os_tmp = [option_sac[i]['J_results'][j]['J_mean'] for i in range(len(option_sac))]
    s_tmp = [sac[i]['J_results'][j]['J_mean'] for i in range(len(option_sac))]
    os_means[j] = np.mean(os_tmp)
    s_means[j] = np.mean(s_tmp)
    os_std[j] = np.std(os_tmp)
    s_std[j] = np.std(s_tmp)

fig, ax = plt.subplots()
ax.grid(alpha=0.5, linestyle='-')
ax.plot(os_means, color='green', label='rARHMM SAC')
ax.fill_between(np.arange(os_means.shape[0]), os_means - os_std, os_means + os_std, alpha=0.2, color='green',
                )

ax.plot(s_means, color='orange', label='SAC')
ax.fill_between(np.arange(s_means.shape[0]), s_means - s_std, s_means + s_std, alpha=0.2, color='orange',
                )

ax.set_title("Cartpole100Hz - evaluation")
ax.legend()
ax.set_xlabel('Episode')
ax.set_ylabel('Expected return')

plt.show()
