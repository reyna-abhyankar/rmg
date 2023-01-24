import torch.nn as nn
from abc import ABC, abstractmethod
import random

def create_op_pool():
  pool = {4: [Conv2d(), ReLU(), Flatten()], 2: [Linear(), ReLU()]}
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

class Conv2d(Operator):
  def __init__(self):
    super().__init__(4, 4)
  
  def compute_dims(self, input_shape):
    self.in_channels = input_shape[1]
    self.out_channels = random.randint(1, 512)
    height = input_shape[2]
    width = input_shape[3]
    self.kernel_size = (random.randint(1, height), random.randint(1, width))
    h_out = height - self.kernel_size[0] + 1
    w_out = width - self.kernel_size[1] + 1
    # missing: padding, stride, dilation, groups
    return [input_shape[0], self.out_channels, h_out, w_out]
  
  def construct_op(self):
    return nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size)

class Flatten(Operator):
  def __init__(self):
    super().__init__(0, 0)
  
  def compute_dims(self, input_shape):
    self.start_dim = 1
    self.end_dim = -1
    flattened_dim = 1
    for i in range(self.start_dim, self.end_dim):
      flattened_dim *= input_shape[i]
    return [input_shape[0], flattened_dim]
  
  def construct_op(self):
    return nn.Flatten(self.start_dim, self.end_dim)

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