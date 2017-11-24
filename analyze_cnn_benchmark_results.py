import json, os, argparse, itertools, math
from collections import defaultdict


parser = argparse.ArgumentParser()
parser.add_argument('--results_dir', default='outputs')
parser.add_argument('--include_std', default=False, action='store_true')
args = parser.parse_args()

# Maps the cuDNN version reported by torch.cudnn to a more friendly string
def cudnn_name(version):
  if version==None or version=='none':
      return 'None'
  if isinstance(version, str):
      return version
  # 5105 -> '5.1.05'
  minor = version % 100
  mid = version / 100 % 10
  major = version / 1000
  return '%d.%d.%02d' % (major, mid, minor)

# Maps the GPU name reported by the driver to a more friendly string
gpu_name_map = {
  'Tesla V100-SXM2-16GB': 'Tesla V100',
  'Tesla P100-SXM2-16GB': 'Tesla P100',
  'TITAN X': 'Pascal Titan X',
  'GeForce GTX TITAN X': 'Maxwell Titan X',
  'GeForce GTX 1080': 'GTX 1080',
  'GeForce GTX 1080 Ti': 'GTX 1080 Ti',

  'c4.4xlarge': 'CPU: Dual Xeon E5-2666 v3',
  'cpu': 'CPU: Dual Xeon E5-2630 v3',
}

# List defines order in which models will be printed
# Matches order in README.md
model_names_sorted = [
  'alexnet',
  'googlenet-v1',
  'vgg16',
  'vgg19',
  'resnet-18',
  'resnet-34',
  'resnet-50',
  'resnet-101',
  'resnet-152',
  'resnet-200'
]

def main(args):
  # Load all the results
  results = []
  for dirpath, dirnames, fns in os.walk(args.results_dir):
    for fn in fns:
      if not fn.endswith('.json'): continue
      with open(os.path.join(dirpath, fn), 'r') as f:
        results.append(json.load(f))

  all_values = defaultdict(set)
  keyed_results = {}
  
  for result in results:
    gpu_name = result['gpu_name']
    cudnn_version = result['cudnn_version']
    model = result['opt']['model_t7']
    
    batch_size = result['opt']['batch_size']
    im_width = result['opt']['image_width']
    im_height = result['opt']['image_height']
    input_size = '%d x 3 x %d x %d' % (batch_size, im_height, im_width)
    
    model = os.path.splitext(os.path.basename(model))[0]
    keyed_results[(gpu_name, cudnn_version, model)] = result
    
    all_values['gpu_name'].add(gpu_name)
    all_values['cudnn_version'].add(cudnn_version)
    all_values['model'].add(model)
    all_values['input_size'].add(input_size)
  
  for k, vs in all_values.iteritems():
    print k
    for v in vs:
      print '  %s' % v
  
  markdown_tables = []

  models = all_values['model']
  # Sort loaded models to match README.md
  sorted_models = [x for x in model_names_sorted if x in models]
  # Append any remaining models that are not yet mentioned in README.md at the end
  sorted_models += [x for x in models if x not in model_names_sorted]
  
  for model in sorted_models:
    for input_size in all_values['input_size']:
      table_header = '|GPU|cuDNN|Forward (ms)|Backward (ms)|Total (ms)|'
      table_header2 = '|---|---|---:|---:|---:|'
      table_lines = {}
      for gpu_name in all_values['gpu_name']:
        for cudnn_version in all_values['cudnn_version']:
          k = (gpu_name, cudnn_version, model)
          if k not in keyed_results: continue
          result = keyed_results[k]

          cudnn_str = cudnn_name(cudnn_version)
          gpu_str = gpu_name_map.get(gpu_name, gpu_name)

          f_mean = mean(result['forward_times']) * 1000
          f_std = std(result['forward_times']) * 1000
          b_mean = mean(result['backward_times']) * 1000
          b_std = std(result['backward_times']) * 1000
          t_mean = mean(result['total_times']) * 1000
          t_std = std(result['total_times']) * 1000

          if args.include_std:
            f_str = '%7.2f += %4.2f' % (f_mean, f_std)
            b_str = '%7.2f += %4.2f' % (b_mean, b_std)
            t_str = '%7.2f += %4.2f' % (t_mean, t_std)
          else:
            f_str = '%7.2f' % f_mean
            b_str = '%7.2f' % b_mean
            t_str = '%7.2f' % t_mean
          table_lines[t_mean] = '|%-25s|%-7s|%s|%s|%s|' % (
                gpu_str, cudnn_str, f_str, b_str, t_str)

      table_lines = [table_lines[k] for k in sorted(table_lines)]
      table_lines = [table_header, table_header2] + table_lines
      model_batch_str = '%s (input %s)' % (model, input_size)
      markdown_tables.append((model_batch_str, table_lines))

  for model, table_lines in markdown_tables:
    print model
    for line in table_lines:
      print line
    print


def mean(xs):
  return float(sum(xs)) / len(xs)


def std(xs):
  m = mean(xs)
  diffs = [x - m for x in xs]
  var = sum(d ** 2.0 for d in diffs) / (len(xs) - 1)
  return math.sqrt(var)
  
        
if __name__ == '__main__':
  args = parser.parse_args()
  main(args)

