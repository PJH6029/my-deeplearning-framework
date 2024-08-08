from typing import Union, Optional

from my_framework import Layer, utils, Variable, Parameter, Function
from my_framework.types import *
import my_framework.functions as F
import my_framework.layers as L

class Model(Layer):
    def plot(self, *inputs: Variable, to_file: str = "model.png", verbose: bool = True) -> None:
        y = self.forward(*inputs)
        return utils.plot_dot_graph(y, to_file=to_file, verbose=verbose)
    
class MLP(Model):
    def __init__(self, fc_output_sizes: list[int], activation = F.sigmoid) -> None:
        super().__init__()
        self.activation = activation
        self.layers = []
        
        for i, out_size in enumerate(fc_output_sizes):
            layer = L.Linear(out_size)
            setattr(self, f"l{i}", layer)
            self.layers.append(layer)
    
    def forward(self, x: Variable) -> Variable:
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)