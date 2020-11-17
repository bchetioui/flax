# Copyright 2020 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3

# Copyright 2020 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for Jax2Tf regression and integration testing."""

import atexit
import contextlib
import logging
import time
from absl.testing import absltest
from collections import defaultdict
from typing import Callable, Dict, List, Tuple

import numpy as np

import jax
import jax.profiler
from jax import numpy as jnp
from jax import test_util as jtu
from jax.experimental import jax2tf
import tensorflow as tf

# Profiler
server = jax.profiler.start_server(9999)

benchmarks: List[Tuple[str, Dict[str, float]]]
benchmarks = []

def log_benchmarks():
  def _pad_cell(cell_content, length, padchar=' '):
    assert len(cell_content) <= length, str((cell_content, length))
    return cell_content + (length - len(cell_content)) * padchar

  precision = 6
  modes = ('JAX', 'TF eager', 'TF graph', 'TF compiled')

  lines = [['Example'] + list(modes)]
  pad_len = {k: max(len(k), precision) for k in lines[0]}
  for example, benchmark in sorted(benchmarks):
    benchmark_line = [example]
    pad_len['Example'] = max(pad_len['Example'], len(example))
    for mode in modes:
      #pad_len[mode] = max(len(str(benchmark[mode])), pad_len[mode])
      benchmark_line.append(str(benchmark[mode])[:precision])
    lines.append(benchmark_line)

  columns = lines[0][:]
  for line in lines:
    for i in range(len(columns)):
      line[i] = _pad_cell(line[i], pad_len[columns[i]])

  table = '\n'.join(list(map(lambda line: ' | '.join(line), lines)))
  print(table)
  return table

atexit.register(log_benchmarks)

class JaxToTfTestCase(absltest.TestCase):
  """Base class for JaxToTf tests."""

  def setUp(self):
    super().setUp()
    # Ensure that all TF ops are created on the proper device (TPU, GPU or CPU)
    # TODO(necula): why doesn't TF do this automatically?
    tf_preferred_devices = (
        tf.config.list_logical_devices("TPU") +
        tf.config.list_logical_devices("GPU") +
        tf.config.list_logical_devices())
    self.tf_default_device = tf_preferred_devices[0]
    logging.info("Running jax2tf converted code on %s.", self.tf_default_device)
    if jtu.device_under_test() != "gpu":
      # TODO(necula): Change the build flags to ensure the GPU is seen by TF
      # It seems that we need --config=cuda build flag for this to work?
      self.assertEqual(jtu.device_under_test().upper(),
                       self.tf_default_device.device_type)

    with contextlib.ExitStack() as stack:
      stack.enter_context(tf.device(self.tf_default_device))
      self.addCleanup(stack.pop_all().close)

  def ConvertAndBenchmark(self, func_jax: Callable, *args,
                          enable_xla: bool = True,
                          name: str):
    def benchmark(func: Callable, *args):
      input(f'Launch execution of {name}?')
      start = time.time()
      func(*args)
      end = time.time()
      return end - start

    # JITTING
    _ = jax.jit(func_jax)(*args)[0].block_until_ready()

    results = dict()
    results["JAX"] = benchmark(lambda *args: jax.jit(func_jax)(*args)[0].block_until_ready(), *args)
    
    func_tf = jax2tf.convert(jax.jit(func_jax), enable_xla=enable_xla)

    def convert_if_bfloat16(v):
      if hasattr(v, "dtype"):
        return tf.convert_to_tensor(np.array(v, jnp.float32) if
                                      v.dtype == jnp.bfloat16 else v,
                                    jax2tf.jax2tf.to_tf_dtype(v.dtype))
      return v

    tf_args = tf.nest.map_structure(convert_if_bfloat16, args)

    def make_input_signature(*tf_args) -> List[tf.TensorSpec]:
      # tf_args can be PyTrees
      def make_one_arg_signature(tf_arg):
        return tf.TensorSpec(np.shape(tf_arg), tf_arg.dtype)
      return tf.nest.map_structure(make_one_arg_signature, list(tf_args))

    def build_tf_func(mode, tf_args=tf_args):
      if mode == "eager":
        return func_tf
      elif mode == "graph":
        print(make_input_signature(*tf_args))
        return (tf.function(
          func_tf, autograph=False,
          input_signature=make_input_signature(*tf_args))
          .get_concrete_function(*tf_args))
      elif mode == "compiled":
        # Adding an explicit input_signature prevents TF from constant-folding
        # the computation eagerly before compilation
        return (tf.function(
          func_tf, autograph=False,
          experimental_compile=True,
          input_signature=make_input_signature(*tf_args))
          .get_concrete_function(*tf_args))
      else:
        assert False

    for mode in ("eager", "graph", "compiled"):
      func_tf = build_tf_func(mode)
      _ = func_tf(*tf_args)
      results[f"TF {mode}"] = benchmark(func_tf, *tf_args)

    print(f'Results are: {results}')

    benchmarks.append((name, results))
