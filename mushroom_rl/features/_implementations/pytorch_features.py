import numpy as np
import torch

from .features_implementation import FeaturesImplementation


class PyTorchFeatures(FeaturesImplementation):
    def __init__(self, tensor_list, device=None):
        self._phi = tensor_list
        self._device = device

    def __call__(self, *args):
        x = self._concatenate(args)

        x = torch.from_numpy(np.atleast_2d(x))

        y_list = [self._phi[i].forward(x) for i in range(len(self._phi))]
        y = torch.stack(y_list, dim=-1)

        y = y.detach().numpy()

        if y.shape[0] == 1:
            return y[0]
        else:
            return y

    @property
    def size(self):
        return len(self._phi)
