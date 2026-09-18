from typing import Any

import jax.numpy as jnp
from jax import Device, NamedSharding
from jax.typing import DTypeLike

from lynqx._src.array import NamedArrayImpl
from lynqx._src.axis_util import axis_shape_to_tuple
from lynqx._src.sharding import canonicalize_sharding
from lynqx._src.typing import AxisShape, NamedArray, ShardingLike


def array(
    a: Any,
    shape: AxisShape,
    dtype: DTypeLike | None = None,
    *,
    copy: bool | None = None,
    device: Device | NamedSharding | None = None,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Convert an object into a `NamedArray`

    Args:
        a: an object that is convertiable to a `NamedArray`. This includes JAX arrays, NumPy arrays, Python scalars,
            Python collections like lists and tuples, objects with a `__jax_array__` method, and objects supporting
            the Python buffer protocol.
        shape: the shape of the resulting `NamedArray`. This can be specified as a sequence of `Axis` objects, a
            single `Axis` object, a sor a mapping from axis names to sizes.
        dtype: optionally specify the data type of the resulting `NamedArray`. If None, the data type will be inferred
            from the input object.
        copy: optional boolean specifying the copy mode. If None, the default behavior of `jax.numpy.asarray` will be
            used.
        device: optional :class:`~jax.Device` or :class:`~jax.sharding.Sharding` to which the created array will be
            committed.
        out_sharding: optional :class:`~jax.P`, :class:`~lynwx.PM` or :class:`~jax.NamedSharding` specifying the
            sharding of the output array. Cannot be used together with ``device``.

    Returns:
        A `NamedArray` constructed from the input arguments.
    """

    axis_shape = axis_shape_to_tuple(shape)
    jax_sharding = canonicalize_sharding(axis_shape, out_sharding, "array")

    jax_array = jnp.asarray(a, dtype, copy=copy, device=device, out_sharding=jax_sharding)
    return NamedArrayImpl(jax_array, axis_shape)


def asnamedarray(
    a: Any,
    shape: AxisShape,
    dtype: DTypeLike | None = None,
    *,
    copy: bool | None = None,
    device: Device | NamedSharding | None = None,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Convert an object into a `NamedArray`. Alias for :func:`lynqx.named.named`.

    Args:
        a: an object that is convertiable to a `NamedArray`. This includes JAX arrays, NumPy arrays, Python scalars,
            Python collections like lists and tuples, objects with a `__jax_array__` method, and objects supporting
            the Python buffer protocol.
        shape: the shape of the resulting `NamedArray`. This can be specified as a sequence of `Axis` objects, a
            single `Axis` object, a sor a mapping from axis names to sizes.
        dtype: optionally specify the data type of the resulting `NamedArray`. If None, the data type will be inferred
            from the input object.
        copy: optional boolean specifying the copy mode. If None, the default behavior of `jax.numpy.asarray` will be
            used.
        device: optional :class:`~jax.Device` or :class:`~jax.sharding.Sharding` to which the created array will be
            committed.
        out_sharding: optional :class:`~jax.P`, :class:`~lynwx.PM` or :class:`~jax.NamedSharding` specifying the
            sharding of the output array. Cannot be used together with ``device``.

    Returns:
        A `NamedArray` constructed from the input arguments.
    """

    return array(a, shape, dtype, copy=copy, device=device, out_sharding=out_sharding)
