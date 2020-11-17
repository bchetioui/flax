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
"""Jax2Tf tests for flax.examples.mnist."""

import functools

from absl.testing import absltest
from importlib import reload
import mnist_lib
reload(mnist_lib) # hack

from flax.testing import jax2tf_test_util

import jax
from jax import random
from jax import test_util as jtu
from jax.experimental import jax2tf

import numpy as np

import tensorflow_datasets as tfds

BATCH_SIZE = 32
DEFAULT_ATOL = 1e-6


def _single_train_step(train_ds):
  params = mnist_lib.get_initial_params(random.PRNGKey(0))
  optimizer = mnist_lib.create_optimizer(params, 0.1, 0.9)
  # Run single train step.
  optimizer, train_metrics = mnist_lib.train_step(
      optimizer=optimizer,
      batch={k: v[:BATCH_SIZE] for k, v in train_ds.items()})
  return train_metrics['loss'], train_metrics['accuracy']


def _eval(test_ds):
  params = mnist_lib.get_initial_params(random.PRNGKey(0))
  return mnist_lib.eval_step(params, test_ds)


class Jax2TfTest(jax2tf_test_util.JaxToTfTestCase):
  """Tests that compare the results of model w/ and w/o using jax2tf."""

  def setUp(self):
    super().setUp()
    # Load mock data so that dataset is not downloaded over the network.
    self._train_ds, self._test_ds = mnist_lib.get_datasets()

  def test_single_train_step(self):
    np.testing.assert_allclose(
        _single_train_step(self._train_ds),
        jax2tf.convert(_single_train_step)(self._train_ds),
        atol=DEFAULT_ATOL)

  def test_eval(self):
    assert_allclose = functools.partial(
        np.testing.assert_allclose, atol=DEFAULT_ATOL)
    jax.tree_multimap(assert_allclose, _eval(self._test_ds),
                      jax2tf.convert(_eval)(self._test_ds))

  def test_perf_single_train_step(self):
    self.ConvertAndBenchmark(_single_train_step, self._train_ds, name='mnist')

if __name__ == '__main__':
  # Parse absl flags test_srcdir and test_tmpdir.
  jax.config.parse_flags_with_absl()
  absltest.main(testLoader=jtu.JaxTestLoader())
