from mushroom_rl.algorithms import Agent


class DeepAC(Agent):
    """
    Base class for algorithms that uses the reparametrization trick, such as
    SAC, DDPG and TD3.

    """
    def __init__(self, mdp_info, policy, actor_optimizer, parameters):
        """
        Constructor.

        Args:
            actor_optimizer (dict): parameters to specify the actor optimizer
                algorithm;
            parameters: policy parameters to be optimized.
        """
        if actor_optimizer is not None:
            self._optimizer = actor_optimizer['class'](
                parameters, **actor_optimizer['params']
            )

            self._clipping = None
            self._parameters = parameters

            if 'clipping' in actor_optimizer:
                self._clipping = actor_optimizer['clipping']['method']
                self._clipping_params = actor_optimizer['clipping']['params']

        super().__init__(mdp_info, policy)

    def fit(self, dataset):
        """
        Fit step.

        Args:
            dataset (list): the dataset.

        """
        raise NotImplementedError('DeepAC is an abstract class')

    def _optimize_actor_parameters(self, loss):
        """
        Method used to update actor parameters to maximize a given loss.

        Args:
            loss (torch.tensor): the loss computed by the algorithm.

        """
        self._optimizer.zero_grad()
        loss.backward()
        self._clip_gradient()
        self._optimizer.step()

    def _clip_gradient(self):
        if self._clipping:
            self._clipping(self._parameters, **self._clipping_params)

    @staticmethod
    def _init_target(online, target):
        for i in range(len(target)):
            target[i].set_weights(online[i].get_weights())

    def _update_target(self, online, target):
        for i in range(len(target)):
            weights = self._tau * online[i].get_weights()
            weights += (1 - self._tau) * target[i].get_weights()
            target[i].set_weights(weights)
