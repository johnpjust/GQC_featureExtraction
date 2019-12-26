# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Operations for generating random numbers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_random_ops
from tensorflow.python.ops import math_ops

# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_random_ops import *
# pylint: enable=wildcard-import

from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export

# @tf_export("random.uniform", v1=["random.uniform", "random_uniform"])
# @deprecation.deprecated_endpoints("random_uniform")
def random_uniform(shape,
                   minval=0,
                   maxval=None,
                   dtype=dtypes.float32,
                   seed=None,
                   name=None):
  """Outputs random values from a uniform distribution.

  The generated values follow a uniform distribution in the range
  `[minval, maxval)`. The lower bound `minval` is included in the range, while
  the upper bound `maxval` is excluded.

  For floats, the default range is `[0, 1)`.  For ints, at least `maxval` must
  be specified explicitly.

  In the integer case, the random integers are slightly biased unless
  `maxval - minval` is an exact power of two.  The bias is small for values of
  `maxval - minval` significantly smaller than the range of the output (either
  `2**32` or `2**64`).

  Args:
    shape: A 1-D integer Tensor or Python array. The shape of the output tensor.
    minval: A 0-D Tensor or Python value of type `dtype`. The lower bound on the
      range of random values to generate.  Defaults to 0.
    maxval: A 0-D Tensor or Python value of type `dtype`. The upper bound on
      the range of random values to generate.  Defaults to 1 if `dtype` is
      floating point.
    dtype: The type of the output: `float16`, `float32`, `float64`, `int32`,
      or `int64`.
    seed: A Python integer. Used to create a random seed for the distribution.
      See `tf.compat.v1.set_random_seed`
      for behavior.
    name: A name for the operation (optional).

  Returns:
    A tensor of the specified shape filled with random uniform values.

  Raises:
    ValueError: If `dtype` is integral and `maxval` is not specified.
  """
  dtype = dtypes.as_dtype(dtype)
  if dtype not in (dtypes.float16, dtypes.bfloat16, dtypes.float32,
                   dtypes.float64, dtypes.int32, dtypes.int64):
    raise ValueError("Invalid dtype %r" % dtype)
  if maxval is None:
    if dtype.is_integer:
      raise ValueError("Must specify maxval for integer dtype %r" % dtype)
    maxval = 1
  with ops.name_scope(name, "random_uniform", [shape, minval, maxval]) as name:
    shape = tensor_util.shape_tensor(shape)
    minval = ops.convert_to_tensor(minval, dtype=dtype, name="min")
    maxval = ops.convert_to_tensor(maxval, dtype=dtype, name="max")
    seed1, seed2 = random_seed.get_seed(seed)
    if dtype.is_integer:
      result = gen_random_ops.random_uniform_int(
          shape, minval, maxval, seed=seed1, seed2=seed2, name=name)
    else:
      rnd = gen_random_ops.random_uniform(shape, dtype, seed=seed1, seed2=seed2)
      result = math_ops.add(rnd * (maxval - minval), minval, name=name)
    # TODO(b/132092188): C++ shape inference inside functional ops does not
    # cross FuncGraph boundaries since that information is only available in
    # python. So we manually get the static shape using
    # `constant_value_as_shape` which *does* cross function boundaries.
    tensor_util.maybe_set_static_shape(result, shape)
    return result


# ops.NotDifferentiable("RandomUniform")


# @tf_export("image.random_crop", v1=["image.random_crop", "random_crop"])
@deprecation.deprecated_endpoints("random_crop")
def random_crop(value, size, seed=None, name=None):
  """Randomly crops a tensor to a given size.

  Slices a shape `size` portion out of `value` at a uniformly chosen offset.
  Requires `value.shape >= size`.

  If a dimension should not be cropped, pass the full size of that dimension.
  For example, RGB images can be cropped with
  `size = [crop_height, crop_width, 3]`.

  Args:
    value: Input tensor to crop.
    size: 1-D tensor with size the rank of `value`.
    seed: Python integer. Used to create a random seed. See
      `tf.compat.v1.set_random_seed`
      for behavior.
    name: A name for this operation (optional).

  Returns:
    A cropped tensor of the same rank as `value` and shape `size`.
  """
  # TODO(shlens): Implement edge case to guarantee output size dimensions.
  # If size > value.shape, zero pad the result so that it always has shape
  # exactly size.
  with ops.name_scope(name, "random_crop", [value, size]) as name:
    value = ops.convert_to_tensor(value, name="value")
    size = ops.convert_to_tensor(size, dtype=dtypes.int32, name="size")
    shape = array_ops.shape(value)
    check = control_flow_ops.Assert(
        math_ops.reduce_all(shape >= size),
        ["Need value.shape >= size, got ", shape, size],
        summarize=1000)
    shape = control_flow_ops.with_dependencies([check], shape)
    limit = shape - size + 1
    offset = random_uniform(
        array_ops.shape(shape),
        dtype=size.dtype,
        maxval=size.dtype.max,
        seed=seed) % limit
    return array_ops.slice(value, offset, size, name=name), offset, size

