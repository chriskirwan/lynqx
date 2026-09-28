from collections.abc import Callable

import jax.numpy as jnp
from jax.typing import DTypeLike

from lynqx._src.axis_util import axis_indices, axis_selection_to_tuple, remove_axes
from lynqx._src.named import constructors, operations, util
from lynqx._src.typing import AxisLike, AxisSelection, NamedArray, NamedArrayLike


py_any = any


def wrap_axiswise_call(fn: Callable, single_axis_only: bool = False) -> Callable:
    """Wrap an axis-wise JAX function so it accepts named axes.

    Args:
        fn: JAX function to wrap.
        single_axis_only: Whether the function accepts only one axis.

    Returns:
        A callable that preserves named-array axes.
    """

    def wrapped_axiswise_fn(
        a: NamedArrayLike, axis: AxisSelection = (), where: NamedArrayLike | None = None, **kwargs
    ) -> NamedArray:
        named = util.ensure_named(f"{getattr(fn, '__name__', 'function')}", a)

        if where is not None:
            where = util.ensure_named(f"{getattr(fn, '__name__', 'function')}", where)
            where = operations.broadcast_to(where, named.axes)
            kwargs["where"] = where.array

        if axis is None:
            result = fn(named.array, axis=None, **kwargs)
            return constructors.array(result, named.axes)
        else:
            axis_tuple = axis_selection_to_tuple(axis)
            indices = axis_indices(named.axes, axis_tuple)
            if py_any(x is None for x in indices):
                raise ValueError(f"Axis {axis} not in {named.axes}")
            if len(indices) == 1:
                return constructors.array(fn(named.array, axis=indices[0], **kwargs), named.axes)
            elif single_axis_only:
                raise ValueError(f"{getattr(fn, '__name__', 'function')} only supports a single axis")
            else:
                return constructors.array(fn(named.array, axis=indices, **kwargs), named.axes)

    return wrapped_axiswise_fn


def wrap_reduction_call(
    fn: Callable,
    single_axis_only: bool = False,
    supports_where: bool = True,
    supports_initial: bool = False,
) -> Callable:
    """
    Wraps a reduction function to support NamedArray and apply common checks.

    Args:
        fn: The reduction function (e.g., jnp.sum, jnp.mean).
        single_axis_only: If True, the wrapped function will only allow reduction
                          along a single axis for NamedArray inputs.
        supports_where: If False, a ValueError will be raised if 'where' is provided.
        supports_initial: If False, a ValueError will be raised if 'initial' is provided.

    Returns:
        A wrapped function that can be called with NamedArray or standard array inputs.
    """

    def wrapped_reduction_fn(
        a: NamedArrayLike,
        axis: AxisSelection = (),
        initial: NamedArrayLike | None = None,
        where: NamedArrayLike | None = None,
        **kwargs,
    ):
        if where is not None:
            where = util.ensure_named(f"{getattr(fn, '__name__', 'function')}", where)

        named = util.ensure_named(f"{fn}", a)
        return _reduce_one_leaf(
            fn, named, axis, initial, where, single_axis_only, supports_where, supports_initial, **kwargs
        )

    return wrapped_reduction_fn


def _reduce_one_leaf(
    fn: Callable,
    a: NamedArray,
    axis: AxisSelection,
    initial: NamedArrayLike | None,
    where: NamedArrayLike | None,
    single_axis_only: bool,
    supports_where: bool,
    supports_initial: bool,
    **kwargs,
):
    """
    Helper function to perform reduction on a single leaf (NamedArray or Array).
    """
    if where is not None and not supports_where:
        raise ValueError(f"where is not supported by {getattr(fn, '__name__', 'function')}")
    if initial is not None and not supports_initial:
        raise ValueError(f"initial is not supported by {getattr(fn, '__name__', 'function')}")

    # These checks apply before any array-specific logic
    if kwargs.get("out", None) is not None:
        raise ValueError("out is not supported for NamedArray")
    keepdims = kwargs.get("keepdims", False)

    if where is not None:
        where = operations.broadcast_to(where, a.axes)
        kwargs["where"] = where.array

    if initial is not None:
        kwargs["initial"] = initial

    if axis is None:
        result = fn(a.array, axis=None, **kwargs)
        if keepdims:
            out_axes = tuple(ax.resize(size=1) for ax in a.axes)
        else:
            out_axes = ()

        return constructors.array(result, out_axes)
    else:
        axis_tuple = axis_selection_to_tuple(axis)
        if single_axis_only and len(axis_tuple) > 1:
            raise ValueError(f"{getattr(fn, '__name__', 'function')} only supports a single axis")

        indices = axis_indices(a.axes, axis_tuple)
        if indices is None or py_any(x is None for x in indices):
            raise ValueError(f"axis {axis} is not in {a.axes}")

        new_axes = remove_axes(a.axes, axis)

        if keepdims:
            indices_set = set(indices)
            new_axes = tuple(ax.resize(size=1) if i in indices_set else ax for i, ax in enumerate(a.axes))
        else:
            new_axes = remove_axes(a.axes, axis)

        if single_axis_only:
            result = fn(a.array, axis=indices[0], **kwargs)
        else:
            result = fn(a.array, axis=indices, **kwargs)
        return constructors.array(result, new_axes)


def any(
    array: NamedArrayLike,
    axis: AxisSelection | None = None,
    keepdims: bool = False,
    *,
    where: NamedArrayLike | None = None,
) -> NamedArray:
    """Test whether any element is true along the selected axes.

    Args:
        array: Input array.
        axis: Axis or axes to reduce, or ``None`` for all axes.
        keepdims: Whether to retain reduced axes with size one.
        where: Optional boolean mask selecting elements to include.

    Returns:
        A boolean named array containing the result.
    """
    return wrap_reduction_call(jnp.any)(array, axis, keepdims=keepdims, where=where)


def all(
    array: NamedArrayLike,
    axis: AxisSelection | None = None,
    keepdims: bool = False,
    *,
    where: NamedArrayLike | None = None,
) -> NamedArray:
    """Test whether all elements are true along the selected axes.

    Args:
        array: Input array.
        axis: Axis or axes to reduce, or ``None`` for all axes.
        keepdims: Whether to retain reduced axes with size one.
        where: Optional boolean mask selecting elements to include.

    Returns:
        A boolean named array containing the result.
    """
    return wrap_reduction_call(jnp.all)(array, axis, keepdims=keepdims, where=where)


# TODO: restrict typehints on `initial` to Scalar (Scalar NamedArray, Scalar Array, Python Scalars)
def sum(
    array: NamedArrayLike,
    axis: AxisSelection | None = None,
    dtype: DTypeLike | None = None,
    keepdims: bool = False,
    initial: NamedArrayLike | None = None,
    where: NamedArrayLike | None = None,
    promote_integers: bool = True,
) -> NamedArray:
    """Sum array elements over the selected axes.

    Args:
        array: Input array.
        axis: Axis or axes to reduce, or ``None`` for all axes.
        dtype: Dtype of the result.
        keepdims: Whether to retain reduced axes with size one.
        initial: Value added to the reduction.
        where: Optional boolean mask selecting elements to include.
        promote_integers: Whether to promote integer inputs.

    Returns:
        A named array containing the sum.
    """
    return wrap_reduction_call(jnp.sum, supports_initial=True)(
        array, axis, dtype=dtype, initial=initial, keepdims=keepdims, where=where, promote_integers=promote_integers
    )


# TODO: restrict typehints on `initial` to Scalar (Scalar NamedArray, Scalar Array, Python Scalars)
def prod(
    array: NamedArrayLike,
    axis: AxisSelection | None = None,
    dtype: DTypeLike | None = None,
    keepdims: bool = False,
    initial: NamedArrayLike | None = None,
    where: NamedArrayLike | None = None,
    promote_integers: bool = True,
) -> NamedArray:
    """Multiply array elements over the selected axes.

    Args:
        array: Input array.
        axis: Axis or axes to reduce, or ``None`` for all axes.
        dtype: Dtype of the result.
        keepdims: Whether to retain reduced axes with size one.
        initial: Value multiplied into the reduction.
        where: Optional boolean mask selecting elements to include.
        promote_integers: Whether to promote integer inputs.

    Returns:
        A named array containing the product.
    """
    return wrap_reduction_call(jnp.prod, supports_initial=True)(
        array, axis, initial=initial, keepdims=keepdims, where=where, dtype=dtype, promote_integers=promote_integers
    )


def mean(
    array: NamedArrayLike,
    axis: AxisSelection | None = None,
    dtype: DTypeLike | None = None,
    keepdims: bool = False,
    *,
    where: NamedArrayLike | None = None,
) -> NamedArray:
    """Compute the arithmetic mean over the selected axes.

    Args:
        array: Input array.
        axis: Axis or axes to reduce, or ``None`` for all axes.
        dtype: Dtype of the result.
        keepdims: Whether to retain reduced axes with size one.
        where: Optional boolean mask selecting elements to include.

    Returns:
        A named array containing the mean.
    """
    return wrap_reduction_call(jnp.mean)(array, axis, keepdims=keepdims, where=where, dtype=dtype)


# TODO: support `mean` parameter
def std(
    array: NamedArrayLike,
    axis: AxisSelection | None = None,
    dtype: DTypeLike | None = None,
    ddof: int = 0,
    keepdims: bool = False,
    *,
    where: NamedArrayLike | None = None,
    correction: int | float | None = None,
) -> NamedArray:
    """Compute the standard deviation over the selected axes.

    Args:
        array: Input array.
        axis: Axis or axes to reduce, or ``None`` for all axes.
        dtype: Dtype of the result.
        ddof: Relative number of degrees of freedom.
        keepdims: Whether to retain reduced axes with size one.
        where: Optional boolean mask selecting elements to include.
        correction: Alternative degrees-of-freedom correction.

    Returns:
        A named array containing the standard deviation.
    """
    return wrap_reduction_call(jnp.std)(
        array, axis, ddof=ddof, keepdims=keepdims, correction=correction, where=where, dtype=dtype
    )


def var(
    array: NamedArrayLike,
    axis: AxisSelection | None = None,
    dtype: DTypeLike | None = None,
    ddof: int = 0,
    keepdims: bool = False,
    *,
    where: NamedArrayLike | None = None,
    correction: int | float | None = None,
) -> NamedArray:
    """Compute the variance over the selected axes.

    Args:
        array: Input array.
        axis: Axis or axes to reduce, or ``None`` for all axes.
        dtype: Dtype of the result.
        ddof: Relative number of degrees of freedom.
        keepdims: Whether to retain reduced axes with size one.
        where: Optional boolean mask selecting elements to include.
        correction: Alternative degrees-of-freedom correction.

    Returns:
        A named array containing the variance.
    """
    return wrap_reduction_call(jnp.var)(
        array, axis, ddof=ddof, keepdims=keepdims, correction=correction, where=where, dtype=dtype
    )


# TODO: restrict typehints on `initial` to Scalar (Scalar NamedArray, Scalar Array, Python Scalars)
def min(
    array: NamedArrayLike,
    axis: AxisSelection | None = None,
    keepdims: bool = False,
    initial: NamedArrayLike | None = None,
    where: NamedArrayLike | None = None,
) -> NamedArray:
    """Compute the minimum over the selected axes.

    Args:
        array: Input array.
        axis: Axis or axes to reduce, or ``None`` for all axes.
        keepdims: Whether to retain reduced axes with size one.
        initial: Value used to initialize the reduction.
        where: Optional boolean mask selecting elements to include.

    Returns:
        A named array containing the minimum.
    """
    return wrap_reduction_call(jnp.min, supports_initial=True)(
        array, axis, keepdims=keepdims, initial=initial, where=where
    )


# TODO: restrict typehints on `initial` to Scalar (Scalar NamedArray, Scalar Array, Python Scalars)
def max(
    array: NamedArrayLike,
    axis: AxisSelection | None = None,
    keepdims: bool = False,
    initial: NamedArrayLike | None = None,
    where: NamedArrayLike | None = None,
) -> NamedArray:
    """Compute the maximum over the selected axes.

    Args:
        array: Input array.
        axis: Axis or axes to reduce, or ``None`` for all axes.
        keepdims: Whether to retain reduced axes with size one.
        initial: Value used to initialize the reduction.
        where: Optional boolean mask selecting elements to include.

    Returns:
        A named array containing the maximum.
    """
    return wrap_reduction_call(jnp.max, supports_initial=True)(
        array, axis, initial=initial, keepdims=keepdims, where=where
    )


# TODO: restrict typehints on `initial` to Scalar (Scalar NamedArray, Scalar Array, Python Scalars)
def amin(
    array: NamedArrayLike,
    axis: AxisSelection | None = None,
    initial: NamedArrayLike | None = None,
    keepdims: bool = False,
    where: NamedArrayLike | None = None,
) -> NamedArray:
    """Compute the minimum over the selected axes; alias of :func:`min`.

    Args:
        array: Input array.
        axis: Axis or axes to reduce, or ``None`` for all axes.
        initial: Value used to initialize the reduction.
        keepdims: Whether to retain reduced axes with size one.
        where: Optional boolean mask selecting elements to include.

    Returns:
        A named array containing the minimum.
    """
    return wrap_reduction_call(jnp.amin, supports_initial=True)(
        array, axis, initial=initial, keepdims=keepdims, where=where
    )


# TODO: restrict typehints on `initial` to Scalar (Scalar NamedArray, Scalar Array, Python Scalars)
def amax(
    array: NamedArrayLike,
    axis: AxisSelection | None = None,
    keepdims: bool = False,
    initial: NamedArrayLike | None = None,
    where: NamedArrayLike | None = None,
) -> NamedArray:
    """Compute the maximum over the selected axes; alias of :func:`max`.

    Args:
        array: Input array.
        axis: Axis or axes to reduce, or ``None`` for all axes.
        initial: Value used to initialize the reduction.
        keepdims: Whether to retain reduced axes with size one.
        where: Optional boolean mask selecting elements to include.

    Returns:
        A named array containing the maximum.
    """
    return wrap_reduction_call(jnp.amax, supports_initial=True)(
        array, axis, initial=initial, keepdims=keepdims, where=where
    )


def argmin(
    array: NamedArrayLike,
    axis: AxisSelection | None = None,
    keepdims: bool = False,
) -> NamedArray:
    """Return indices of minimum values over the selected axes.

    Args:
        array: Input array.
        axis: Axis or axes to reduce, or ``None`` for all axes.
        keepdims: Whether to retain reduced axes with size one.

    Returns:
        A named integer array containing the minimum indices.
    """
    return wrap_reduction_call(jnp.argmin, single_axis_only=True, supports_where=False)(array, axis, keepdims=keepdims)


def argmax(
    array: NamedArrayLike,
    axis: AxisSelection | None = None,
    keepdims: bool = False,
) -> NamedArray:
    """Return indices of maximum values over the selected axes.

    Args:
        array: Input array.
        axis: Axis or axes to reduce, or ``None`` for all axes.
        keepdims: Whether to retain reduced axes with size one.

    Returns:
        A named integer array containing the maximum indices.
    """
    return wrap_reduction_call(jnp.argmax, single_axis_only=True, supports_where=False)(array, axis, keepdims=keepdims)


def ptp(
    array: NamedArrayLike,
    axis: AxisSelection | None = None,
    keepdims: bool = False,
) -> NamedArray:
    """Compute the peak-to-peak range over the selected axes.

    Args:
        array: Input array.
        axis: Axis or axes to reduce, or ``None`` for all axes.
        keepdims: Whether to retain reduced axes with size one.

    Returns:
        A named array containing maximum minus minimum.
    """
    return wrap_reduction_call(jnp.ptp, supports_where=False)(array, axis, keepdims=keepdims)


def nanargmax(array: NamedArrayLike, axis: AxisLike | None = None, *, keepdims: bool = False) -> NamedArray:
    """Return indices of maximum values while ignoring NaNs.

    Args:
        array: Input array.
        axis: Axis along which to reduce, or ``None`` for all axes.
        keepdims: Whether to retain the reduced axis with size one.

    Returns:
        A named integer array containing the maximum indices.
    """
    return wrap_reduction_call(jnp.nanargmax, single_axis_only=True, supports_where=False)(
        array, axis, keepdims=keepdims
    )


def nanargmin(array: NamedArrayLike, axis: AxisLike | None = None, *, keepdims: bool = False) -> NamedArray:
    """Return indices of minimum values while ignoring NaNs.

    Args:
        array: Input array.
        axis: Axis along which to reduce, or ``None`` for all axes.
        keepdims: Whether to retain the reduced axis with size one.

    Returns:
        A named integer array containing the minimum indices.
    """
    return wrap_reduction_call(jnp.nanargmin, single_axis_only=True, supports_where=False)(
        array, axis, keepdims=keepdims
    )


def nanmax(
    array: NamedArrayLike,
    axis: AxisLike | None = None,
    keepdims: bool = False,
    initial: NamedArrayLike | None = None,
    where: NamedArrayLike | None = None,
) -> NamedArray:
    """Compute the maximum over the selected axes while ignoring NaNs.

    Args:
        array: Input array.
        axis: Axis or axes to reduce, or ``None`` for all axes.
        keepdims: Whether to retain reduced axes with size one.
        initial: Value used to initialize the reduction.
        where: Optional boolean mask selecting elements to include.

    Returns:
        A named array containing the maximum.
    """
    return wrap_reduction_call(jnp.nanmax, supports_initial=True, supports_where=True)(
        array, axis, initial=initial, keepdims=keepdims, where=where
    )


def nanmin(
    array: NamedArrayLike,
    axis: AxisLike | None = None,
    keepdims: bool = False,
    initial: NamedArrayLike | None = None,
    where: NamedArrayLike | None = None,
) -> NamedArray:
    """Compute the minimum over the selected axes while ignoring NaNs.

    Args:
        array: Input array.
        axis: Axis or axes to reduce, or ``None`` for all axes.
        keepdims: Whether to retain reduced axes with size one.
        initial: Value used to initialize the reduction.
        where: Optional boolean mask selecting elements to include.

    Returns:
        A named array containing the minimum.
    """
    return wrap_reduction_call(jnp.nanmin, supports_initial=True, supports_where=True)(
        array, axis, initial=initial, keepdims=keepdims, where=where
    )


def nanprod(
    array: NamedArrayLike,
    axis: AxisLike | None = None,
    dtype: DTypeLike | None = None,
    keepdims: bool = False,
    where: NamedArrayLike | None = None,
) -> NamedArray:
    """Compute the product over the selected axes while ignoring NaNs.

    Args:
        array: Input array.
        axis: Axis or axes to reduce, or ``None`` for all axes.
        dtype: Dtype of the result.
        keepdims: Whether to retain reduced axes with size one.
        where: Optional boolean mask selecting elements to include.

    Returns:
        A named array containing the product.
    """
    return wrap_reduction_call(jnp.nanprod)(array, axis, dtype=dtype, keepdims=keepdims, where=where)


def nanstd(
    array: NamedArrayLike,
    axis: AxisLike | None = None,
    dtype: DTypeLike | None = None,
    keepdims: bool = False,
    where: NamedArrayLike | None = None,
) -> NamedArray:
    """Compute the standard deviation over the selected axes while ignoring NaNs.

    Args:
        array: Input array.
        axis: Axis or axes to reduce, or ``None`` for all axes.
        dtype: Dtype of the result.
        keepdims: Whether to retain reduced axes with size one.
        where: Optional boolean mask selecting elements to include.

    Returns:
        A named array containing the standard deviation.
    """
    return wrap_reduction_call(jnp.nanstd)(array, axis, dtype=dtype, keepdims=keepdims, where=where)


def nansum(
    array: NamedArrayLike,
    axis: AxisLike | None = None,
    dtype: DTypeLike | None = None,
    keepdims: bool = False,
    where: NamedArrayLike | None = None,
) -> NamedArray:
    """Compute the sum over the selected axes while ignoring NaNs.

    Args:
        array: Input array.
        axis: Axis or axes to reduce, or ``None`` for all axes.
        dtype: Dtype of the result.
        keepdims: Whether to retain reduced axes with size one.
        where: Optional boolean mask selecting elements to include.

    Returns:
        A named array containing the sum.
    """
    return wrap_reduction_call(jnp.nansum)(array, axis, dtype=dtype, keepdims=keepdims, where=where)


def nanvar(
    array: NamedArrayLike,
    axis: AxisLike | None = None,
    dtype: DTypeLike | None = None,
    ddof: int = 0,
    keepdims: bool = False,
    where: NamedArrayLike | None = None,
) -> NamedArray:
    """Compute the variance over the selected axes while ignoring NaNs.

    Args:
        array: Input array.
        axis: Axis or axes to reduce, or ``None`` for all axes.
        dtype: Dtype of the result.
        ddof: Relative number of degrees of freedom.
        keepdims: Whether to retain reduced axes with size one.
        where: Optional boolean mask selecting elements to include.

    Returns:
        A named array containing the variance.
    """
    return wrap_reduction_call(jnp.nanvar)(array, axis, dtype=dtype, ddof=ddof, keepdims=keepdims, where=where)


def cumsum(array: NamedArrayLike, axis: AxisLike, dtype: DTypeLike | None = None) -> NamedArray:
    """Return the cumulative sum along an axis.

    Args:
        array: Input array.
        axis: Axis along which to compute the cumulative sum.
        dtype: Dtype of the result.

    Returns:
        A named array containing the cumulative sums.
    """
    return wrap_axiswise_call(jnp.cumsum, single_axis_only=True)(array, axis, dtype=dtype)


def cumprod(array: NamedArrayLike, axis: AxisLike, dtype: DTypeLike | None = None) -> NamedArray:
    """Return the cumulative product along an axis.

    Args:
        array: Input array.
        axis: Axis along which to compute the cumulative product.
        dtype: Dtype of the result.

    Returns:
        A named array containing the cumulative products.
    """
    return wrap_axiswise_call(jnp.cumprod, single_axis_only=True)(array, axis, dtype=dtype)


def nancumsum(array: NamedArrayLike, axis: AxisLike, dtype: DTypeLike | None = None) -> NamedArray:
    """Return the cumulative sum along an axis, treating NaNs as zero.

    Args:
        array: Input array.
        axis: Axis along which to compute the cumulative sum.
        dtype: Dtype of the result.

    Returns:
        A named array containing the cumulative sums.
    """
    return wrap_axiswise_call(jnp.nancumsum, single_axis_only=True)(array, axis, dtype=dtype)


def nancumprod(array: NamedArrayLike, axis: AxisLike, dtype: DTypeLike | None = None) -> NamedArray:
    """Return the cumulative product along an axis, treating NaNs as one.

    Args:
        array: Input array.
        axis: Axis along which to compute the cumulative product.
        dtype: Dtype of the result.

    Returns:
        A named array containing the cumulative products.
    """
    return wrap_axiswise_call(jnp.nancumprod, single_axis_only=True)(array, axis, dtype=dtype)


def sort(
    array: NamedArrayLike,
    axis: AxisSelection | None = None,
    *,
    kind: str | None = None,
    order: str | None = None,
) -> NamedArray:
    """Sort an array along an axis.

    Args:
        array: Input array.
        axis: Axis along which to sort, or ``None`` for the flattened array.
        kind: Sorting algorithm to use.
        order: Field name for structured arrays.

    Returns:
        A named array containing the sorted values.
    """
    return wrap_axiswise_call(jnp.sort, single_axis_only=True)(array, axis, kind=kind, order=order)


def argsort(
    array: NamedArrayLike,
    axis: AxisSelection | None = None,
    *,
    kind: str | None = None,
    order: str | None = None,
    dtype: DTypeLike | None = None,
) -> NamedArray:
    """Return indices that would sort an array along an axis.

    Args:
        array: Input array.
        axis: Axis along which to sort, or ``None`` for the flattened array.
        kind: Sorting algorithm to use.
        order: Field name for structured arrays.
        dtype: Dtype of the returned indices.

    Returns:
        A named integer array containing the sorting indices.
    """
    return wrap_axiswise_call(jnp.argsort, single_axis_only=True)(array, axis, kind=kind, order=order, dtype=dtype)


def partition(
    a: NamedArrayLike,
    kth: int,
    axis: AxisSelection,
) -> NamedArray:
    """Partition an array around ``kth`` along an axis.

    Args:
        a: Input array.
        kth: Index of the element around which to partition.
        axis: Axis along which to partition.

    Returns:
        A named array containing the partitioned values.
    """
    return wrap_axiswise_call(jnp.partition, single_axis_only=True)(a, axis, kth=kth)


def argpartition(
    a: NamedArrayLike,
    kth: int,
    axis: AxisSelection,
) -> NamedArray:
    """Return indices that partition an array around ``kth`` along an axis.

    Args:
        a: Input array.
        kth: Index of the element around which to partition.
        axis: Axis along which to partition.

    Returns:
        A named integer array containing the partitioning indices.
    """
    return wrap_axiswise_call(jnp.argpartition, single_axis_only=True)(a, axis, kth=kth)
