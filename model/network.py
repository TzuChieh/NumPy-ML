import common.vector as vec
from model.cost import CostFunction, CrossEntropy
from model.layer import Layer

import numpy as np
import numpy.typing as np_type

import pickle
from pathlib import Path


class Network:
    def __init__(self, hidden_layers: list[Layer], cost: CostFunction=CrossEntropy()):
        self._hidden_layers = hidden_layers
        self.num_layers = len(hidden_layers) + 1
        self.cost = cost

        layer_info = ""
        total_params = 0
        for layer in self.hidden_layers:
            layer_info += f"{str(layer)}\n"
            total_params += layer.num_params
        print(f"\nNetwork layers ({total_params} parameters):\n{layer_info}")

        if not _has_valid_connections(self.hidden_layers):
            raise ValueError("network layers does not have valid connections")

    def feedforward(self, x, is_training):
        """
        @param x The input vector.
        @param is_training `True` if the intent is to train the network, `False` for inference.
        @return The output vector.
        """
        # Freeze all layers if not for training
        if not is_training:
            layer_trainabilities = [layer.is_trainable for layer in self.hidden_layers]
            for layer in self.hidden_layers:
                layer.freeze()

        a = x
        for layer in self.hidden_layers:
            a = layer.feedforward(a)
        
        # Unfreeze originally trainable layers
        if not is_training:
            for layer, was_trainable in zip(self.hidden_layers, layer_trainabilities):
                if was_trainable:
                    layer.unfreeze()

        return a

    def save(self, file_path):
        """
        @param file_path Path to the saved file. A suitable extension will be added automatically.
        """
        file_path = Path(file_path).with_suffix('.model')

        with open(file_path, 'wb') as f:
            pickle.dump(self.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, file_path):
        """
        @param file_path Path to the loaded file. A suitable extension will be added automatically.
        """
        file_path = Path(file_path).with_suffix('.model')

        with open(file_path, 'rb') as f:
            self.__dict__ = pickle.load(f)

    @property
    def biases(self):
        """
        @return List of biases of the hidden layers.
        """
        return [layer.bias for layer in self.hidden_layers]
    
    @property
    def weights(self):
        """
        @return List of weights of the hidden layers.
        """
        return [layer.weight for layer in self.hidden_layers]
    
    @property
    def hidden_layers(self) -> list[Layer]:
        """
        @return List of hidden layers in this network.
        """
        return self._hidden_layers
    
    @property
    def input_shape(self) -> np_type.NDArray:
        assert len(self._hidden_layers) > 0
        return self._hidden_layers[0].input_shape
    
    @property
    def output_shape(self) -> np_type.NDArray:
        assert len(self._hidden_layers) > 0
        return self._hidden_layers[-1].output_shape
    

def _has_valid_connections(layers: list[Layer]):
    """
    @return Whether the specified layers properly connect with each other.
    """
    if len(layers) <= 1:
        return True

    for curr_layer, prev_layer in zip(layers[1:], layers[:-1]):
        if not np.array_equal(curr_layer.input_shape, prev_layer.output_shape):
            return False
    return True
