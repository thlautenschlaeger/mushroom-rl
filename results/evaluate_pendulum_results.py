import matplotlib.pyplot as plt
import torch
import numpy as np
import scipy.stats

option_sac = torch.load('option_sac_pendulum100Hz_experiments_tanh.pkl')
sac = torch.load('sac_pendulum100Hz_experiments_tanh.pkl')

os_means = np.zeros((len(option_sac[0]['J_results'])))
s_means = np.zeros((len(sac[0]['J_results'])))
os_confidence = np.zeros((len(option_sac[0]['J_results'])))
s_confidence = np.zeros((len(sac[0]['J_results'])))

for j in range(os_means.shape[0]):
    os_tmp = np.array([option_sac[i]['J_results'][j]['J_mean'] for i in range(len(option_sac))])
    s_tmp = np.array([sac[i]['J_results'][j]['J_mean'] for i in range(len(option_sac))])
    os_means[j] = np.mean(os_tmp)
    s_means[j] = np.mean(s_tmp)
    os_confidence[j] = scipy.stats.sem(os_tmp) * scipy.stats.t.ppf((1 + 0.95) / 2., os_tmp.shape[0] - 1)
    s_confidence[j] = scipy.stats.sem(s_tmp) * scipy.stats.t.ppf((1 + 0.95) / 2., s_tmp.shape[0] - 1)

fig, ax = plt.subplots()
ax.grid(alpha=0.5, linestyle='-')
ax.plot(os_means, color='green', label='rARHMM SAC')
ax.fill_between(np.arange(os_means.shape[0]), os_means - os_confidence, os_means + os_confidence, alpha=0.2, color='green',
                )

ax.plot(s_means, color='orange', label='SAC')
ax.fill_between(np.arange(s_means.shape[0]), s_means - s_confidence, s_means + s_confidence, alpha=0.2, color='orange',
                )

ax.set_title("Pendulum100Hz - evaluation")
ax.legend()
ax.set_xlabel('Episode')
ax.set_ylabel('Expected return')

plt.show()
