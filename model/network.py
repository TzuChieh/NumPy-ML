import common.vector as vec
from model.cost import CostFunction, CrossEntropy
from model.layer import Layer

import numpy as np


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

    def performance(self, dataset):
        """
        Evaluate the effectiveness of the network.
        @param dataset A list of pairs. Each pair contains input activation (x) and output activation (y).
        @return A tuple that contains (in order): number of correct outputs, fraction of correct outputs.
        """
        results = [
            (np.argmax(self.feedforward(vec.vector_2d(x), is_training=False)), np.argmax(y))
            for x, y in dataset]
        
        num_right_answers = sum(1 for network_y, y in results if network_y == y)
        num_data = len(dataset)
        return (num_right_answers, num_right_answers / num_data)

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
