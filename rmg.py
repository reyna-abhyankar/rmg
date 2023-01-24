import argparse
from operators import *

def main():
  parser = argparse.ArgumentParser(description='Random Model Generation')
  parser.add_argument('--depth', type=int, default=100, help='number of operators')
  parser.add_argument('--input-shape', nargs='+', help='shape of input tensor (include batch dimension)', required=True)
  args = parser.parse_args()
  tensor_shape = [int(dim) for dim in args.input_shape]
  print(tensor_shape)

  modules = []
  pool = create_op_pool()
  depth = 0
  while depth < args.depth:
    op = random.choice(pool[len(tensor_shape)])
    tensor_shape = op.compute_dims(tensor_shape)
    modules.append(op.construct_op())
    depth += 1
    print(depth)
  model = nn.Sequential(*modules)
  print(model)

if __name__ == '__main__':
  main()