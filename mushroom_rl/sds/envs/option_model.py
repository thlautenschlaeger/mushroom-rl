import numpy as np
import math

from mushroom_rl.sds import rARHMM


class OptionSwitchingModel(object):

    def __init__(self, rarhmm: rARHMM):
        self.rarhmm = rarhmm
        self.n_options = self.rarhmm.nb_states
        self.reset()

    def reset(self):
        self.hist_obs = np.zeros((0, self.rarhmm.dm_obs))
        self.hist_act = np.zeros((0, self.rarhmm.dm_act))
        self.hist_inits_obs_loglik = None
        self.hist_trans_loglik = np.zeros((0, self.n_options, self.n_options))
        self.hist_obs_loglik = np.zeros((0, self.n_options))
        self.last_weights = None

    def get_transition_weights(self, obs, act):
        self.hist_act = np.vstack((self.hist_act, act))
        self.hist_obs = np.vstack((self.hist_obs, obs))
        weights = self.rarhmm.filter(self.hist_obs, self.hist_act)[0][-1, ...]

        return weights

    def get_transition_weights_reuse_logliks(self, obs, act):
        self.hist_act = np.vstack((self.hist_act, act))
        self.hist_obs = np.vstack((self.hist_obs, obs))

        weights, tmp_logliks = self.rarhmm.filter_reuse_logliks(self.hist_obs[-2:], self.hist_act[-2:],
                                                                logliks=[self.hist_inits_obs_loglik, self.hist_trans_loglik,
                                                                 self.hist_obs_loglik])
        weights = weights[0][-1, ...]

        # a = [math.isnan(w) for w in weights]
        # if any(a):
        #     print("")
        self.last_weights = weights
        self.hist_inits_obs_loglik = tmp_logliks[0]
        self.hist_trans_loglik = tmp_logliks[1]
        self.hist_obs_loglik = tmp_logliks[2]

        return weights
