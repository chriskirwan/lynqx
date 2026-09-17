import jax.lax
import jax.numpy as jnp
from jax.typing import DTypeLike

from lynqx._src.axis_util import axis_index, axis_shape_to_tuple, axis_sizes, validate_unique_axes
from lynqx._src.named import constructors, util
from lynqx._src.sharding import canonicalize_sharding
from lynqx._src.typing import Axis, AxisSelector, AxisShape, DuckTypedArray, NamedArray, NamedArrayLike, ShardingLike


def zeros(
    shape: AxisShape,
    dtype: DTypeLike | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Create an array full of zeros.

    Args:
        shape: specify the shape of the created array.
        dtype: optional dtype for the created array; defaults to float32 or float64 depending on the X64 configuration.
        out_sharding: optional :class:`~lynqx.PM`, :class:`~jax.P` or :class:`~jax.NamedSharding` specifying the
            sharding of the output array.
    Returns:
        Named array of the specified shape and dtype, with the given sharding if specified.
    """

    return full(shape, 0.0, dtype, out_sharding=out_sharding)


def ones(
    shape: AxisShape,
    dtype: DTypeLike | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Create an array full of ones.

    Args:
            shape: specify the shape of the created array.
            dtype: optional dtype for the created array; defaults to float32 or float64 depending on the X64
                configuration.
            out_sharding: optional :class:`~lynqx.PM`, :class:`~jax.P` or :class:`~jax.NamedSharding` specifying the
                sharding of the output array.
        Returns:
            Named array of the specified shape and dtype, with the given device/sharding if specified.
    """

    return full(shape, 1.0, dtype, out_sharding=out_sharding)


def empty(
    shape: AxisShape,
    dtype: DTypeLike | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Create an empty array.

    Args:
        shape: specify the shape of the created array.
        dtype: optional dtype for the created array; defaults to float32 or float64 depending on the X64 configuration.
        out_sharding: optional :class:`~lynqx.PM`, :class:`~jax.P` or :class:`~jax.NamedSharding` specifying the
            sharding of the output array.
    Returns:
        Named array of the specified shape and dtype, with the given device/sharding if specified.
    """

    return full(shape, 0.0, dtype, out_sharding=out_sharding)


@validate_unique_axes(arg_names=("shape",))
def full(
    shape: AxisShape,
    fill_value: NamedArrayLike,
    dtype: DTypeLike | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Create an array filled with a specified value.

    Args:
        shape: specify the shape of the created array.
        fill_value: scalar or array with which to fill the created array.
        dtype: optional dtype for the created array; defaults to float32 or float64 depending on the X64 configuration.
        out_sharding: optional :class:`~lynqx.PM`, :class:`~jax.P` or :class:`~jax.NamedSharding` specifying the
            sharding of the output array.

    Returns:
        Named array of the specified shape and dtype, with the given device/sharding if specified.
    """

    target = axis_shape_to_tuple(shape)
    target_sizes = axis_sizes(target)

    jax_sharding = canonicalize_sharding(out_sharding, target, "full")

    if isinstance(fill_value, NamedArray):
        fill_value = fill_value.array

    jax_array = jnp.full(
        target_sizes,
        fill_value,
        dtype=dtype,
        out_sharding=jax_sharding,
    )

    return constructors.array(jax_array, target)


def zeros_like(
    a: NamedArrayLike | DuckTypedArray,
    dtype: DTypeLike | None = None,
    shape: AxisShape | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Create an array full of zeros with the same shape and dtype as an array.

    Args:
        a: NamedArray-like object with ``shape`` and ``dtype`` attributes.
        shape: optionally override the shape of the created array.
        dtype: optionally override the dtype of the created array.
        out_sharding: optional :class:`~lynqx.PM`, :class:`~jax.P` or :class:`~jax.NamedSharding` specifying the
            sharding of the output array.

    Returns:
        Array of the specified shape and dtype, on the specified device if specified.
    """

    return full_like(a, 0.0, shape=shape, dtype=dtype, out_sharding=out_sharding)


def ones_like(
    a: NamedArrayLike | DuckTypedArray,
    dtype: DTypeLike | None = None,
    shape: AxisShape | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Create an array full of ones with the same shape and dtype as an array.

    Args:
        a: NamedArray-like object with ``shape`` and ``dtype`` attributes.
        shape: optionally override the shape of the created array.
        dtype: optionally override the dtype of the created array.
        out_sharding: optional :class:`~lynqx.PM`, :class:`~jax.P` or :class:`~jax.NamedSharding` specifying the
            sharding of the output array.

    Returns:
        Array of the specified shape and dtype, on the specified device if specified.
    """
    return full_like(a, 1.0, shape=shape, dtype=dtype, out_sharding=out_sharding)


def empty_like(
    a: NamedArrayLike | DuckTypedArray,
    dtype: DTypeLike | None = None,
    shape: AxisShape | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Create an empty array with the same shape and dtype as an array.

    Args:
        a: NamedArray-like object with ``shape`` and ``dtype`` attributes.
        shape: optionally override the shape of the created array.
        dtype: optionally override the dtype of the created array.
        out_sharding: optional :class:`~lynqx.PM`, :class:`~jax.P` or :class:`~jax.NamedSharding` specifying the
            sharding of the output array.

    Returns:
        Array of the specified shape and dtype, on the specified device if specified.
    """

    return full_like(a, 0.0, shape=shape, dtype=dtype, out_sharding=out_sharding)


def full_like(
    a: NamedArrayLike | DuckTypedArray,
    fill_value: NamedArrayLike,
    dtype: DTypeLike | None = None,
    shape: AxisShape | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Create an array full of ones with the same shape and dtype as an array.

    Args:
        a: NamedArray-like object with ``shape`` and ``dtype`` attributes.
        fill_value: scalar or array with which to fill the created array.
        shape: optionally override the shape of the created array.
        dtype: optionally override the dtype of the created array.
        out_sharding: optional :class:`~lynqx.PM`, :class:`~jax.P` or :class:`~jax.NamedSharding` specifying the
            sharding of the output array.

    Returns:
        Array of the specified shape and dtype, on the specified device if specified.
    """

    named, fill_named = util.ensure_named("full_like", a, fill_value)

    fill_shape = named.axes if shape is None else shape

    return full(fill_shape, fill_value, dtype, out_sharding=out_sharding)


def iota(
    shape: AxisShape,
    axis: AxisSelector | None = None,
    dtype: DTypeLike | None = None,
    *,
    out_sharding: ShardingLike | None = None,
):
    """
    Create an array starting at zero and incrementing by one along the specified axis.

    Args:
        shape: the shape of the created array.
        axis: optional axis to increment along. Must be specified if `shape` has more than one axis.
        dtype: optional dtype for the created array.

    """

    shape = axis_shape_to_tuple(shape)
    dtype = jax.dtypes.canonicalize_dtype(dtype)

    if len(shape) == 1:
        # redo this check for compatible axes instead
        if axis is not None:
            raise ValueError(f"axis={axis!r} must not be provided for 1-D shape={shape!r}")
        jax_array = jax.lax.iota(dtype, shape[0].size)
        return constructors.array(jax_array, shape, out_sharding=out_sharding)

    if axis is None:
        raise ValueError()

    axis_idx = axis_index(shape, axis)

    if axis_idx is None:
        raise ValueError()

    jax_shape = axis_sizes(shape)
    jax_sharding = canonicalize_sharding(out_sharding, shape, "iota")
    jax_array = jax.lax.broadcasted_iota(dtype, jax_shape, axis_idx, out_sharding=jax_sharding)

    return constructors.array(jax_array, shape)


def arange(
    axis: Axis,
    start: NamedArrayLike | None = None,
    step: NamedArrayLike | None = None,
    dtype: DTypeLike | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """
    Create an array of evenly-spaced values.

    Args:
        axis: axis to apply the values along.
        start: optional start of the interval, inclusive
        step: optional end on the interval, exclusive.

    Returns:
        Array of evernly spaced values from `start`, separated by `axis.size` values of
        `step`
    """

    if start is None:
        start = 0
    if step is None:
        step = 1

    start, step = util.ensure_scalar("arange", start, step)

    if dtype is None:
        dtype = jax.dtypes.result_type(start, step)

    if jnp.iscomplexobj(start) or jnp.iscomplexobj(step):
        raise ValueError("`arange` does not support complex start/step")

    return iota(axis, dtype=dtype, out_sharding=out_sharding) * step + start
