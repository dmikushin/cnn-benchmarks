from __future__ import print_function
import os, json, itertools, random, argparse

DEFAULT_MODELS = ','.join([
  'models/alexnet.t7',
  'models/vgg16.t7',
  'models/vgg19.t7',
  'models/googlenet-v1.t7',
  'models/resnet-18.t7',
  'models/resnet-34.t7',
  'models/resnet-50.t7',
  'models/resnet-101.t7',
  'models/resnet-152.t7',
  'models/resnet-200.t7',
])

parser = argparse.ArgumentParser()
parser.add_argument('--gpus', default='0')
parser.add_argument('--models', default=DEFAULT_MODELS)
parser.add_argument('--batch_sizes', default='16')
parser.add_argument('--use_cudnns', default='0,1')
parser.add_argument('--output_dir', default='outputs')


def main(args):
  factors = {
    'gpu': [int(g) for g in args.gpus.split(',')],
    'model_t7': args.models.split(','),
    'batch_size': [int(b) for b in args.batch_sizes.split(',')],
    'use_cudnn': [int(c) for c in args.use_cudnns.split(',')],
  } 

  base_command = 'torch/install/bin/th cnn_benchmark.lua'

  try: 
    os.makedirs(args.output_dir)
  except OSError:
    if not os.path.isdir(args.output_dir):
      raise

  output_jsons = set()
  for vals in itertools.product(*factors.values()):
    lua_args = dict(zip(factors.keys(), vals))
    while True:
      lua_args['output_json'] = '/%d.json' % random.randint(1, 100000)
      rand_id = random.randint(1, 1000000)
      lua_args['output_json'] = os.path.join(args.output_dir, '%d.json' % rand_id)
      if lua_args['output_json'] not in output_jsons: break
    output_jsons.add(lua_args['output_json'])

    command = base_command
    for k, v in lua_args.items():
      command = '%s -%s %s' % (command, k, v)

    print(command)
    os.system(command)


if __name__ == '__main__':
  args = parser.parse_args()
  main(args)

