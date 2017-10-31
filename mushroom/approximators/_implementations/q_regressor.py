import numpy as np


class QRegressor:
    """
    This class is used to create a regressor that approximates the Q-function
    using a multi-dimensional output where each output corresponds to the
    Q-value of each action. This is used, for instance, by the `ConvNet` used
    in examples/atari_dqn.

    """
    def __init__(self, approximator, n_actions, approximator_params, **params):
        """
        Constructor.

        Args:
            approximator (object): the model class to approximate the
                Q-function;
            n_actions (int): number of different actions of the problem. It
                determines the output shape of the model;
            approximator_params (dict): parameters dictionary to the regressor;
            **params (dict): parameters dictionary used by the `QRegressor`.

        """
        approximator_params['n_actions'] = n_actions
        self.model = approximator(**approximator_params)
        self._input_preprocessor = params.get('input_preprocessor', list())
        self._output_preprocessor = params.get('output_preprocessor', list())

    def fit(self, state, action, q, **fit_params):
        """
        Fit the model.

        Args:
            state (np.array): states;
            action (np.array): actions;
            q (np.array): target q-values;
            **fit_params (dict): other parameters used by the fit method of the
                regressor.

        """
        state, q = self._preprocess(state, q)
        self.model.fit(state, action, q, **fit_params)

    def predict(self, *z, **predict_params):
        """
        Predict.

        Args:
            *z (list): a list containing states or states and actions depending
                on whether the call requires to predict all q-values or only
                one q-value corresponding to the provided action;
            **predict_params (dict): other parameters used by the predict method
                of each regressor.

        Returns:
            The predictions of the model.

        """
        state = self._preprocess(z[0])
        q = self.model.predict(state, **predict_params)

        if len(z) == 2:
            action = z[1]
            if q.ndim == 1:
                return q[action]
            else:
                return q[np.arange(q.shape[0]), action]
        else:
            assert len(z) == 1

            return q

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, w):
        self.model.set_weights(w)

    def _preprocess(self, state, q=None):
        for p in self._input_preprocessor:
            state = p(state)

        if q is not None:
            for p in self._output_preprocessor:
                q = p(q)

            return state, q
        return state

    def __len__(self):
        return len(self.model)

    def __str__(self):
        return 'QRegressor of ' + str(self.model) + '.'
