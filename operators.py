import torch.nn as nn
from abc import ABC, abstractmethod
import random

def create_op_pool():
  pool = {2: [Linear(), ReLU()]}
  return pool

class Operator(ABC):
  def __init__(self, input_shape_len, output_shape_len):
    self.input_shape_len = input_shape_len
    self.output_shape_len = output_shape_len

  @abstractmethod
  def compute_dims(self, input_shape):
    pass

  @abstractmethod
  def construct_op(self):
    pass

class Linear(Operator):
  def __init__(self):
    super().__init__(2, 2)

  def compute_dims(self, input_shape):
    self.in_features = input_shape[1]
    self.out_features = random.randint(10, 10000)
    return [input_shape[0], self.out_features]

  def construct_op(self):
    return nn.Linear(self.in_features, self.out_features)

class ReLU(Operator):
  def __init__(self):
    super().__init__(0, 0)
    
  def compute_dims(self, input_shape):
    return input_shape

  def construct_op(self):
    return nn.ReLU()