import operator
from collections.abc import Callable, Mapping, Sequence
from functools import reduce
from math import prod
from typing import Any, Literal, overload

import equinox as eqx
import jax.lax
import jax.numpy as jnp
from jax import Device
from jax.typing import DTypeLike

from lynqx._src.axis import AxisLike
from lynqx._src.axis_util import (
    axis_index,
    axis_indices,
    axis_names,
    axis_shape_to_tuple,
    axis_sizes,
    match_axes,
    remove_axes,
    resolve_axes,
    validate_unique_axes,
)
from lynqx._src.named import constructors, creation, reductions, ufuncs, util
from lynqx._src.sharding import canonicalize_sharding
from lynqx._src.typing import Axis, AxisSelection, AxisSelector, AxisShape, NamedArray, NamedArrayLike, ShardingLike


# utility functions


def iscomplex(a: NamedArrayLike) -> NamedArray:
    """Return a boolean array showing where the input is complex.

    Args:
        x: The input array or scalar.

    Returns:
        A zero-dimensional boolean ``NamedArray`` indicating whether ``a`` is complex.
    """

    a = util.ensure_named("iscomplex", a)
    return constructors.array(jnp.iscomplex(a.array), a.axes)


def iscomplexobj(x: Any) -> bool:
    """Check whether the input is a complex number or contains complex elements.

    The decision is based on the input's dtype rather than its values, so arrays with zero imaginary parts are still
    considered complex.

    Args:
        x: The input to test.

    Returns:
        ``True`` if ``x`` has a complex dtype, otherwise ``False``.
    """

    return jnp.iscomplexobj(x)


# Broadcasting support


@validate_unique_axes(arg_names=("shape",))
def broadcast_to(
    a: NamedArrayLike,
    shape: AxisShape,
    *,
    out_sharding: ShardingLike | None = None,
):
    """Broadcast an array to a specified shape.

    Lynqx uses extended broadcasting semantics that allow broadcasting to a shape with different axes, as long as the
    existing axes are preserved and any new axes are added with size 1. This is similar to NumPy's broadcasting rules,
    but with the added constraint that named axes must match.

    Args:
        a: The array to broadcast.
        shape: The target shape to broadcast to.
        out_sharding: Optional sharding specification for the output array. If not specified, it will be determined
            automatically by the compiler.

    Returns:
        A copy of ``a`` broadcasted to the specified shape.
    """
    named = util.ensure_named("broadcast_to", a)
    target = axis_shape_to_tuple(shape)

    matched = match_axes(
        named.axes,
        target,
        allow_positional_fallback=True,
    )

    if matched.unmatched_source:
        missing = [named.axes[i] for i in matched.unmatched_source]
        raise ValueError(f"broadcast_to cannot drop axes {missing}; every existing axis must appear in `shape`")

    # Validate that every matched axis is broadcast-compatible.
    for m in matched.matches:
        src_ax = named.axes[m.source]
        tgt_ax = target[m.target]

        if src_ax.size != tgt_ax.size and src_ax.size != 1:
            raise ValueError(f"Cannot broadcast axis {src_ax.name!r} of size {src_ax.size} to {tgt_ax.size}")

    target_sizes = axis_sizes(target)
    jax_sharding = canonicalize_sharding(
        target,
        out_sharding,
        "broadcast_to",
    )

    if named.array.ndim == 0:
        jax_array = jnp.broadcast_to(named.array, target_sizes, out_sharding=jax_sharding)
        return constructors.array(jax_array, target)

    broadcast_dimensions = tuple(m.target for m in matched.matches)

    # If source axes are not already in target order, transpose first.
    source_order = tuple(m.source for m in sorted(matched.matches, key=lambda m: m.target))
    identity = tuple(range(named.array.ndim))

    if source_order != identity:
        array = jnp.transpose(named.array, source_order)

        # After transposing, matches are in target order.
        broadcast_dimensions = tuple(m.target for m in sorted(matched.matches, key=lambda m: m.target))
    else:
        array = named.array

    jax_array = jax.lax.broadcast_in_dim(
        array,
        target_sizes,
        broadcast_dimensions=broadcast_dimensions,
    )

    return constructors.array(jax_array, target, out_sharding=out_sharding)


def broadcast_arrays_and_shape(
    *arrays: NamedArrayLike, out_sharding: ShardingLike | None = None
) -> tuple[tuple[NamedArray, ...], tuple[Axis, ...]]:
    """Broadcast multiple arrays to a common shape and return both.

    Args:
        *arrays: The arrays to broadcast.
        out_sharding: Optional sharding specification for the output arrays.

    Returns:
        A tuple ``(broadcasted_arrays, common_shape)`` where ``broadcasted_arrays`` is a tuple of ``NamedArray``
        sharing ``common_shape``.
    """

    named_arrays = util.ensure_named_tuple("broadcast_arrays", arrays)
    if not named_arrays:
        return (), ()

    target_shape = named_arrays[0].axes
    for arr in named_arrays[1:]:
        _, _, target_shape = util.align_shapes_for_broadcast(target_shape, arr.axes)

    broadcasted_arrays = tuple(broadcast_to(arr, target_shape, out_sharding=out_sharding) for arr in arrays)
    return broadcasted_arrays, target_shape


def broadcast_arrays(*arrays: NamedArrayLike, out_sharding: ShardingLike | None = None):
    """Broadcast multiple NamedArrays to a common shape.

    Args:
        *arrays: The arrays to broadcast.
        out_sharding: Optional sharding specification for the output arrays.

    Returns:
        A tuple of broadcasted ``NamedArrays``.
    """
    return broadcast_arrays_and_shape(*arrays, out_sharding=out_sharding)[0]


def broadcast_shapes(*shapes: AxisShape) -> tuple[Axis, ...]:
    """Broadcast multiple shapes to a common shape.

    Args:
        *shapes: The shapes to broadcast.

    Returns:
        The common broadcasted shape.
    """
    if not shapes:
        return ()

    target_shape = axis_shape_to_tuple(shapes[0])
    for shape in shapes[1:]:
        _, _, target_shape = util.align_shapes_for_broadcast(target_shape, axis_shape_to_tuple(shape))

    return target_shape


# Additional creation operations


@overload
def linspace(
    start: NamedArrayLike,
    stop: NamedArrayLike,
    axis: AxisLike,
    endpoint: bool = True,
    retstep: Literal[False] = False,
    dtype: DTypeLike | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray: ...


@overload
def linspace(
    start: NamedArrayLike,
    stop: NamedArrayLike,
    axis: AxisLike,
    endpoint: bool = True,
    *,
    retstep: Literal[True],
    dtype: DTypeLike | None = None,
    out_sharding: ShardingLike | None = None,
) -> tuple[NamedArray, NamedArray]: ...


@overload
def linspace(
    start: NamedArrayLike,
    stop: NamedArrayLike,
    axis: AxisLike,
    endpoint: bool = True,
    retstep: bool = False,
    dtype: DTypeLike | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray | tuple[NamedArray, NamedArray]: ...


def linspace(
    start: NamedArrayLike,
    stop: NamedArrayLike,
    axis: AxisLike,
    endpoint: bool = True,
    retstep: bool = False,
    dtype: DTypeLike | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray | tuple[NamedArray, NamedArray]:
    """Return evenly spaced values over a specified interval along a new axis.

    Values are generated over ``[start, stop]`` (inclusive when ``endpoint`` is ``True``) and returned along the new
    ``axis``, with all remaining axes coming from broadcasting ``start`` and ``stop``.

    Args:
        start: The starting value(s).
        stop: The ending value(s).
        axis: The new axis along which the samples are generated. A string is treated as an unnamed axis of size
            50 (matching NumPy's default); an `Axis` supplies both size and optional name.
        endpoint: If ``True`` (default), ``stop`` is included.
        retstep: If ``True``, also return the step size between samples.
        dtype: Optional dtype of the output.
        out_sharding: Optional sharding specification for the output array.

    Returns:
        A ``NamedArray`` of samples, or a tuple ``(samples, step)`` when ``retstep=True``.
    """

    if isinstance(axis, str):
        axis = Axis(50, axis)  # numpy default

    (start, stop), common_axes = broadcast_arrays_and_shape(start, stop)
    if axis.name in axis_names(common_axes):
        raise ValueError(f"Axis {axis} already exists in one of the input arrays.")

    jax_array = jnp.linspace(start.array, stop.array, axis.size, endpoint=endpoint, retstep=retstep, dtype=dtype)

    out_axes = (axis,) + common_axes
    step = None
    if retstep:
        jax_array, jax_array_step = jax_array
        step = constructors.array(jax_array_step, common_axes)

    result = constructors.array(jax_array, out_axes, out_sharding=out_sharding)

    return (result, step) if step is not None else result


def logspace(
    start: NamedArrayLike,
    stop: NamedArrayLike,
    axis: AxisLike,
    endpoint: bool = True,
    base: NamedArrayLike = 10.0,
    dtype: DTypeLike | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Return numbers spaced evenly on a log scale along a new axis.

    Args:
        start: ``base ** start`` is the starting value.
        stop: ``base ** stop`` is the final value.
        axis: The new axis along which the samples are generated. A string is treated as an unnamed axis of size 50;
            an `Axis` supplies size and optional name.
        endpoint: If ``True`` (default), ``stop`` is included.
        base: The base of the log space. Defaults to 10.
        dtype: Optional dtype of the output.
        out_sharding: Optional sharding specification for the output array.

    Returns:
        A ``NamedArray`` of samples on a log scale.
    """
    if isinstance(axis, str):
        axis = Axis(50, axis)  # numpy default

    (start, stop, base), common_axes = broadcast_arrays_and_shape(start, stop, base)
    if axis.name in axis_names(common_axes):
        raise ValueError(f"Axis {axis} already exists in one of the input arrays")

    jax_array = jnp.logspace(start.array, stop.array, axis.size, endpoint=endpoint, base=base.array, dtype=dtype)

    out_axes = (axis,) + common_axes
    return constructors.array(jax_array, out_axes, out_sharding=out_sharding)


def geomspace(
    start: NamedArrayLike,
    stop: NamedArrayLike,
    axis: AxisLike,
    endpoint: bool = True,
    dtype: DTypeLike | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Return numbers spaced evenly on a geometric progression along a new axis.

    Args:
        start: The starting value(s) of the sequence.
        stop: The final value(s) of the sequence.
        axis: The new axis along which the samples are generated. A string is treated as an unnamed axis of size 50;
            an `Axis` supplies size and optional name.
        endpoint: If ``True`` (default), ``stop`` is included.
        dtype: Optional dtype of the output.
        out_sharding: Optional sharding specification for the output array.

    Returns:
        A ``NamedArray`` of samples on a geometric progression.
    """

    if isinstance(axis, str):
        axis = Axis(50, axis)  # numpy default

    (start, stop), common_axes = broadcast_arrays_and_shape(start, stop)
    if axis.name in axis_names(common_axes):
        raise ValueError(f"Axis {axis} already exists in one of the input arrays")

    jax_array = jnp.geomspace(start.array, stop.array, axis.size, endpoint=endpoint, dtype=dtype)

    out_axes = (axis,) + common_axes
    return constructors.array(jax_array, out_axes, out_sharding=out_sharding)


def delta(
    shape: AxisShape,
    axes: AxisSelection | None = None,
    dtype: DTypeLike | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Return a Kronecker delta over the selected axes of the given shape.

    The result equals 1 where all selected axes share the same index and 0 elsewhere.

    Args:
        shape: The output shape.
        axes: The axes to compare for equality. Defaults to all axes.
        dtype: Optional dtype of the output. Defaults to ``float32``.
        out_sharding: Optional sharding specification for the output array.

    Returns:
        A ``NamedArray`` with the given shape.
    """
    shape = axis_shape_to_tuple(shape)
    indices = axis_indices(shape, axes)

    if len(indices) < 2:
        raise ValueError(f"delta requires at least 2 axes to compare, got {len(indices)}")
    if len(set(indices)) != len(indices):
        raise ValueError(f"delta requires distinct axes, got {axes!r} resolving to duplicate positions {indices}")

    indices = tuple(sorted(indices))  # normalize to shape order; equality is symmetric, so this is free
    spec = tuple(shape[i] for i in indices)

    iotas = [creation.iota(spec, i, dtype=jnp.uint32) for i in range(len(spec))]
    eyes = [i1 == i2 for i1, i2 in zip(iotas[:-1], iotas[1:])]
    result = reduce(operator.and_, eyes)

    new_dtype = jnp.float32 if dtype is None else dtype
    casted_result = eqx.tree_at(lambda m: m.array, result, replace_fn=lambda x: jnp.astype(x, new_dtype))
    return broadcast_to(casted_result, shape, out_sharding=out_sharding)


def identity(
    shape: AxisShape,
    axes: AxisSelection = (-1, -2),
    dtype: DTypeLike | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Return a square identity matrix over two selected axes of the given shape.

    Args:
        shape: The output shape.
        axes: The two axes forming the matrix. Defaults to the last two axes.
        dtype: Optional dtype of the output.
        out_sharding: Optional sharding specification for the output array.

    Returns:
        A ``NamedArray`` with the given shape whose two selected axes form an identity matrix.
    """

    resolved = axis_shape_to_tuple(shape)
    indices = axis_indices(resolved, axes)

    if len(indices) != 2:
        raise ValueError(f"identity requires exactly 2 axes, got {len(indices)} from {axes!r}")

    a, b = resolved[indices[0]], resolved[indices[1]]
    if a.size != b.size:
        raise ValueError(
            f"identity requires square axes, got sizes {a.size} and {b.size}; "
            "use delta(...) directly for a non-square diagonal tensor"
        )

    return delta(resolved, axes, dtype=dtype, out_sharding=out_sharding)


# shape/axis-wise operations


def rename(a: NamedArrayLike, axis: AxisSelector, name: str | None) -> NamedArray:
    """Rename an axis of an array.

    Args:
        a: The NamedArray to rename.
        axis: The axis to rename.
        name: The new name for the axis. If None, the axis will be unnamed.

    Returns:
        A ``NamedArray`` with the specified axis renamed.
    """
    named = util.ensure_named("rename", a)
    idx = axis_index(named.axes, axis)
    if idx is None:
        raise ValueError(f"rename: axis {axis!r} not found in {named.axes}")

    new_axes = list(named.axes)
    new_axes[idx] = Axis(new_axes[idx].size, name)
    return constructors.array(named.array, new_axes)


def reshape(a: NamedArrayLike, shape: AxisShape, *, out_sharding: ShardingLike | None = None):
    """Reshape an array to a new shape.

    The total number of elements must be preserved, and the ordering of axes follows the flattened (row-major)
    ordering of the input.

    Args:
        a: The ``NamedArray`` to reshape.
        shape: The target shape.
        out_sharding: Optional sharding specification for the output array.

    Returns:
        A ``NamedArray`` with the requested shape.
    """
    a = util.ensure_named("reshape", a)

    shape_tuple = axis_shape_to_tuple(shape)
    shape_sizes = axis_sizes(shape_tuple)
    if prod(shape_sizes) != a.size:
        raise ValueError()

    jax_sharding = canonicalize_sharding(shape_tuple, out_sharding, "reshape")
    jax_array = jnp.reshape(a.array, shape_sizes, out_sharding=jax_sharding)

    return constructors.array(jax_array, shape_tuple)


def diagonal(
    a: NamedArrayLike, axis1: AxisSelector, axis2: AxisSelector, diagonal: AxisLike | None = None, offset: int = 0
) -> NamedArray:
    """Extract a diagonal from an array.

    Args:
        a: The input array.
        axis1: The first axis forming the diagonal.
        axis2: The second axis forming the diagonal.
        diagonal: The new axis for the extracted diagonal. A string creates an
            unnamed diagonal axis of the appropriate size.
        offset: Offset of the diagonal from the main diagonal. Defaults to 0.

    Returns:
        A ``NamedArray`` with the diagonal extracted along a new axis.
    """
    a = util.ensure_named("diagonal", a)

    batch_axes = remove_axes(a.axes, (axis1, axis2))
    idx0, idx1 = axis_indices(a.axes, (axis1, axis2))

    diag_axis_size = min(a.axes[idx0].size, a.axes[idx1].size) - abs(offset)
    diag_axis = util.resolve_new_axis(diagonal, diag_axis_size, "diagonal", "diagonal")

    out_shape = batch_axes + (diag_axis,)
    jax_array = jnp.diagonal(a.array, offset, idx0, idx1)

    return constructors.array(jax_array, out_shape)


def _tri_mask(array: NamedArray, axis1: AxisSelector, axis2: AxisSelector, k: int, *, upper: bool) -> NamedArray:
    idx0, idx1 = axis_indices(array.axes, (axis1, axis2))
    if idx0 == idx1:
        raise ValueError(f"axis1 and axis2 must be distinct, both resolved to position {idx0}")

    # Keep the two axes in *array* order so broadcast_to needs no transpose.
    spec = resolve_axes(array.axes, (idx0, idx1))

    # axis1 plays "row", axis2 plays "column", wherever they sit in spec.
    row_pos, col_pos = (0, 1) if idx0 < idx1 else (1, 0)
    row = creation.iota(spec, row_pos, dtype=jnp.int32)
    col = creation.iota(spec, col_pos, dtype=jnp.int32)

    diff = row - col
    mask = (diff >= k) if upper else (diff <= k)
    return broadcast_to(mask, array.axes)


def triu(a: NamedArrayLike, axis1: AxisSelector, axis2: AxisSelector, k: int = 0) -> NamedArray:
    """Return the upper triangle of an array.

    Args:
        a: The input array.
        axis1: The first axis of the matrix.
        axis2: The second axis of the matrix.
        k: Diagonal above which to keep elements. Defaults to 0.

    Returns:
        A ```NamedArray``` with elements below the ``k``-th diagonal zeroed out.
    """

    named = util.ensure_named("triu", a)
    mask = _tri_mask(named, axis1, axis2, k, upper=True)
    return constructors.array(jnp.where(mask.array, named.array, 0), named.axes)


def tril(a: NamedArrayLike, axis1: AxisSelector, axis2: AxisSelector, k: int = 0) -> NamedArray:
    """Return the lower triangle of an array.

    Args:
        a: The input array.
        axis1: The first axis of the matrix.
        axis2: The second axis of the matrix.
        k: Diagonal below which to keep elements. Defaults to 0.

    Returns:
        A ```NamedArray``` with elements above the ``k``-th diagonal zeroed out.
    """
    named = util.ensure_named("tril", a)
    mask = _tri_mask(named, axis1, axis2, k, upper=False)
    return constructors.array(jnp.where(mask.array, named.array, 0), named.axes)


def tri(shape: AxisShape, axes: AxisSelection = (-1, -2), k: int = 0, dtype: DTypeLike | None = None) -> NamedArray:
    """Return an array with ones at and below the ``k``-th diagonal and zeros elsewhere.

    Args:
        shape: The output shape.
        axes: The two axes forming the matrix. Defaults to the last two axes.
        k: The diagonal offset. Defaults to 0.
        dtype: Optional dtype of the output.

    Returns:
        A `NamedArray` with the given shape.
    """
    resolved = axis_shape_to_tuple(shape)
    indices = axis_indices(resolved, axes)

    a, b = indices

    diag = creation.ones(shape, dtype=dtype)
    mask = _tri_mask(diag, a, b, k, upper=False)
    return mask * diag


def transpose(a: NamedArrayLike, permutation: AxisSelection | None = None):
    """Transpose a NamedArray according to the given axis permutation.

    Args:
        a: The NamedArray to transpose.
        permutation: A sequence of axis names or indices specifying the new order of axes.
                     If None, reverses the order of axes.

    Returns:
        A ```NamedArray``` with axes transposed according to the specified permutation.
    """
    named = util.ensure_named("transpose", a)

    if permutation is None:
        permutes_axes = tuple(reversed(named.axes))
    else:
        permutes_axes = resolve_axes(named.axes, permutation)

    matched = match_axes(named.axes, permutes_axes, allow_positional_fallback=True)
    if matched.unmatched_source or matched.unmatched_target:
        raise ValueError("Invalid permutation: all axes must be accounted for in the permutation.")

    tgt_to_src = {m.target: m.source for m in matched.matches}
    axis_permutation = tuple(tgt_to_src[i] for i in range(len(permutes_axes)))

    jax_array = jnp.transpose(named.array, axes=axis_permutation)
    return constructors.array(jax_array, permutes_axes)


def swapaxes(a: NamedArrayLike, axis1: AxisSelector, axis2: AxisSelector):
    """Swap two axes of a NamedArray.

    Args:
        a: The NamedArray to swap axes.
        axis1: The first axis to swap.
        axis2: The second axis to swap.

    Returns:
        A new NamedArray with the specified axes swapped.
    """
    named = util.ensure_named("swapaxes", a)
    idx0, idx1 = axis_indices(named.axes, (axis1, axis2))

    if idx0 == idx1:
        raise ValueError(f"swapaxes: axis1 and axis2 must be distinct, both resolved to position {idx0}")

    new_axes = list(named.axes)
    new_axes[idx0], new_axes[idx1] = new_axes[idx1], new_axes[idx0]

    jax_array = jnp.swapaxes(named.array, idx0, idx1)
    return constructors.array(jax_array, tuple(new_axes))


def roll(a: NamedArrayLike, shift: NamedArrayLike | int | Sequence[int], axis: AxisSelection):
    """
    Roll a NamedArray along specified axes.

    Args:
        a: The NamedArray to roll.
        shift: The number of positions by which elements are shifted. Can be an integer or a sequence of integers. If
            a sequence, its length must match the number of axes specified.
        axis: The axes along which to roll. Can be a single axis or a sequence of axes.

    Returns:
        A new NamedArray with the same shape as `a`, but with elements rolled along the specified axes.
    """

    named = util.ensure_named("roll", a)
    resolved_axes = resolve_axes(named.axes, axis)

    if isinstance(shift, int):
        shifts = (shift,) * len(resolved_axes)
    elif isinstance(shift, Sequence):
        if len(shift) != len(resolved_axes):
            raise ValueError(f"roll: shift length {len(shift)} does not match number of axes {len(resolved_axes)}")
        shifts = tuple(shift)
    else:
        raise TypeError(f"roll: shift must be int or sequence of ints, got {type(shift).__name__}")

    jax_array = jnp.roll(named.array, shift=shifts, axis=axis_indices(named.axes, resolved_axes))
    return constructors.array(jax_array, named.axes)


PadValue = int | tuple[int, int]
PadMapping = Mapping[AxisSelector, PadValue]
PadSpec = PadValue | PadMapping


def _resolve_pad_width(val: PadValue) -> tuple[int, int]:
    if isinstance(val, int):
        if val < 0:
            raise ValueError(f"Pad width must be non-negative, got {val}")
        return (val, val)
    else:
        lo, hi = val
        if lo < 0 or hi < 0:
            raise ValueError(f"Pad width must be non-negative, got ({lo}, {hi})")
        return val


def _normalize_pad_width(axes: tuple[Axis, ...], pad_width: PadSpec) -> tuple[tuple[int, int], ...]:
    if not isinstance(pad_width, Mapping):
        if isinstance(pad_width, tuple) and len(pad_width) != 2:
            raise ValueError(
                f"pad_width tuple must be (before, after), got {pad_width!r}; use a mapping for per-axis widths"
            )
        return (_resolve_pad_width(pad_width),) * len(axes)

    widths: list[tuple[int, int] | None] = [None] * len(axes)
    for key, val in pad_width.items():
        idx = axis_index(axes, key)
        if idx is None:  # drop this check if axis_index raises on failure
            raise ValueError(f"pad_width key {key!r} does not match any axis in {tuple(a.name for a in axes)}")
        if widths[idx] is not None:
            raise ValueError(f"axis {axes[idx].name!r} appears more than once in pad_width")
        widths[idx] = _resolve_pad_width(val)
    return tuple(w if w is not None else (0, 0) for w in widths)


def _pad_callable(named, widths, fn, kwargs):
    out = named
    for ax, (lo, hi) in zip(named.axes, widths):
        if lo == 0 and hi == 0:
            continue
        out = util.ensure_named("pad", fn(out, ax, (lo, hi), **kwargs))
        got = resolve_axes(out.axes, ax)[0]
        if got.size != ax.size + lo + hi:
            raise ValueError(f"pad callable returned size {got} for axis {ax.name!r}, expected {ax.size + lo + hi}")
    return out


def pad(
    array: NamedArrayLike, pad_width: PadSpec, mode: str | Callable[..., Any] = "constant", **kwargs
) -> NamedArray:
    """Pad an array along specified axes.

    Args:
        array: The array to pad.
        pad_width: A mapping from axes to pad widths, or a single pad width to apply to all axes. Each pad width can
        be an int (same padding before and after) or a tuple of two ints (padding before, padding after).
        mode: The padding mode. Can be a string (e.g. ``'constant'``, ``'edge'``, ``'reflect'``) or a callable taking
            a `NamedArray` and returning a padded `NamedArray`.
        **kwargs: Extra keyword arguments forwarded to the padding function.

    Returns:
        A new `NamedArray` padded according to the specified widths and mode.
    """
    named = util.ensure_named("pad", array)
    widths = _normalize_pad_width(named.axes, pad_width)
    out_axes = tuple(Axis(ax.size + lo + hi, ax.name) for ax, (lo, hi) in zip(named.axes, widths))

    if callable(mode):
        return _pad_callable(named, widths, mode, kwargs)

    jax_array = jnp.pad(named.array, widths, mode=mode, **kwargs)
    return constructors.array(jax_array, out_axes)


def ravel(a: NamedArrayLike, name: AxisLike | None = None) -> NamedArray:
    """Flatten a NamedArray into a rank-1 array while preserving its axes.

    Args:
        a: The array to flatten.
        name: Optional name for the new flattened axis. If None, the axis will be unnamed.

    Returns:
        A new NamedArray that is a flattened version of the input, with a single axis.
    """
    named = util.ensure_named("ravel", a)

    if isinstance(name, Axis):
        if name.size != named.size:
            raise ValueError(
                f"ravel: provided axis {name!r} has size {name.size}, but flattened array has size {named.size}"
            )
        flat_axis = name
    else:
        flat_axis = Axis(named.size, name)
    jax_array = jnp.ravel(named.array)
    return constructors.array(jax_array, (flat_axis,))


def _normalize_insert_index(index: int, ndim: int, fn_name: str) -> int:
    """Normalise the insertion position of a new axis to ``[0, ndim]``.

    `ndim` is the rank of the *inputs*, so the output has rank `ndim + 1` and valid values are
    `-(ndim + 1) <= index <= ndim`; `-1` appends.
    """
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError(f"{fn_name}: `index` must be an int, got {type(index).__name__}")
    if not -(ndim + 1) <= index <= ndim:
        raise ValueError(
            f"{fn_name}: `index` {index} is out of range for inserting into {ndim} existing axes; "
            f"expected {-(ndim + 1)} <= index <= {ndim}"
        )
    return index + ndim + 1 if index < 0 else index


def _align_permutation(ref: Sequence[Axis], axes: Sequence[Axis], fn_name: str, position: int) -> tuple[int, ...]:
    """Find ``perm`` such that ``axes[perm[i]]`` corresponds to ``ref[i]``.

    Named axes correspond by name. Anonymous axes correspond by order of appearance, which never needs to compare
    sizes, so repeated anonymous axes of the same size are not ambiguous and anonymous axes of different sizes can
    still be paired up.
    """
    if len(ref) != len(axes):
        raise ValueError(
            f"{fn_name}: array at position {position} has {len(axes)} axes but the first array has {len(ref)}; "
            f"all arrays must have the same axes"
        )

    name_to_pos = {ax.name: i for i, ax in enumerate(axes) if ax.name is not None}
    anonymous = iter(i for i, ax in enumerate(axes) if ax.name is None)

    perm: list[int] = []
    for ax in ref:
        if ax.name is None:
            nxt = next(anonymous, None)
            if nxt is None:
                raise ValueError(
                    f"{fn_name}: array at position {position} has fewer anonymous axes than the first array"
                )
            perm.append(nxt)
        elif ax.name in name_to_pos:
            perm.append(name_to_pos[ax.name])
        else:
            raise ValueError(
                f"{fn_name}: array at position {position} has no axis named {ax.name!r}; its axes are "
                f"{tuple(a.label for a in axes)}"
            )
    return tuple(perm)


def _canonicalize_arrays(fn_name: str, arrays: Sequence[NamedArray], *, free: int | None = None):
    """Permute every array's axes into the order of the first array and check sizes."""
    ref = arrays[0].axes
    identity = tuple(range(len(ref)))

    jax_arrays = []
    for position, arr in enumerate(arrays):
        perm = _align_permutation(ref, arr.axes, fn_name, position)
        for i, p in enumerate(perm):
            if i != free and arr.axes[p].size != ref[i].size:
                raise ValueError(
                    f"{fn_name}: axis {ref[i].label!r} has size {arr.axes[p].size} in the array at position "
                    f"{position} but size {ref[i].size} in the first array"
                )
        jax_arrays.append(arr.array if perm == identity else jnp.transpose(arr.array, perm))

    return ref, jax_arrays


def stack(
    arrays: Sequence[NamedArrayLike],
    axis: AxisLike,
    dtype: DTypeLike | None = None,
    *,
    index: int = 0,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Join arrays along a new axis.

    All arrays must have the same axes. Named axes are matched by name (so they may appear in different orders);
    anonymous axes are matched by order of appearance. The result uses the axis order of the first array.

    Args:
        arrays: The arrays to stack.
        axis: The new axis. A string creates a new axis of that name with size ``len(arrays)``; an `Axis` must have
            size ``len(arrays)``.
        dtype: Optional dtype of the result.
        index: Position at which the new axis is inserted into the result. Negative values count from the end of
            the *result*, so ``-1`` appends the new axis last. Defaults to ``0``.
        out_sharding: Optional sharding specification for the output array.

    Returns:
        A ``NamedArray`` with one more axis than the inputs.
    """
    named = util.ensure_named_tuple("stack", tuple(arrays))
    if not named:
        raise ValueError("stack requires at least one array")

    if isinstance(axis, str):
        axis = Axis(len(named), axis)
    elif axis.size != len(named):
        raise ValueError(f"stack: new axis {axis!r} has size {axis.size} but {len(named)} arrays were provided")

    ref, jax_arrays = _canonicalize_arrays("stack", named)

    if axis.name is not None and axis.name in axis_names(ref):
        raise ValueError(f"stack: new axis {axis.name!r} already exists in the input arrays")

    position = _normalize_insert_index(index, len(ref), "stack")
    jax_array = jnp.stack(jax_arrays, axis=position, dtype=dtype)

    out_axes = ref[:position] + (axis,) + ref[position:]
    return constructors.array(jax_array, out_axes, out_sharding=out_sharding)


def unstack(a: NamedArrayLike, axis: AxisSelector = 0) -> tuple[NamedArray, ...]:
    """Split an array into a tuple of arrays along an axis, removing that axis.

    This is the inverse of `stack`.

    Args:
        a: The array to split.
        axis: The axis to split along. Defaults to the first axis.

    Returns:
        A tuple of ``a.axes[axis].size`` arrays, each with the selected axis removed.
    """
    named = util.ensure_named("unstack", a)

    idx = axis_index(named.axes, axis)
    if idx is None:
        raise ValueError(f"unstack: axis {axis!r} not found in {named.axes}")

    out_axes = named.axes[:idx] + named.axes[idx + 1 :]
    return tuple(constructors.array(x, out_axes) for x in jnp.unstack(named.array, axis=idx))


TileReps = int | Sequence[int] | Mapping[AxisSelector, int]


def tile(a: NamedArrayLike, reps: TileReps, *, out_sharding: ShardingLike | None = None) -> NamedArray:
    """Repeat an array along its existing axes.

    Unlike `numpy.tile`, this never prepends axes: ``reps`` is always interpreted per existing axis, and each
    axis of size ``n`` with repetition ``r`` becomes an axis of size ``n * r`` (names are preserved).

    Args:
        a: The array to tile.
        reps: Either a single int applied to every axis, a sequence with one entry per axis (in axis order), or
            a mapping from axis selectors to repetitions, where unlisted axes are repeated once.
        out_sharding: Optional sharding specification for the output array.

    Returns:
        The tiled array, with the same axis names and order as ``a``.
    """
    named = util.ensure_named("tile", a)
    ndim = len(named.axes)

    if isinstance(reps, Mapping):
        per_axis = [1] * ndim
        seen: set[int] = set()
        for key, r in reps.items():
            idx = axis_index(named.axes, key)
            if idx is None:
                raise ValueError(f"tile: axis {key!r} not found in {named.axes}")
            if idx in seen:
                raise ValueError(f"tile: axis {named.axes[idx].label!r} appears more than once in `reps`")
            seen.add(idx)
            per_axis[idx] = r
        rep_tuple = tuple(per_axis)
    elif isinstance(reps, int):
        rep_tuple = (reps,) * ndim
    else:
        rep_tuple = tuple(reps)
        if len(rep_tuple) != ndim:
            raise ValueError(
                f"tile: `reps` has {len(rep_tuple)} entries but the array has {ndim} axes; "
                f"use a mapping to repeat only some axes"
            )

    for r in rep_tuple:
        if isinstance(r, bool) or not isinstance(r, int) or r < 1:
            raise ValueError(f"tile: repetitions must be positive integers, got {rep_tuple}")

    if ndim == 0:
        return named

    out_axes = tuple(ax.resize(ax.size * r) for ax, r in zip(named.axes, rep_tuple))
    jax_array = jnp.tile(named.array, rep_tuple)
    return constructors.array(jax_array, out_axes, out_sharding=out_sharding)


def concatenate(
    arrays: Sequence[NamedArrayLike],
    axis: AxisSelector = 0,
    dtype: DTypeLike | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Join arrays along an existing axis.

    All arrays must have the same axes, and sizes must agree on every axis except the one being joined. Named
    axes are matched by name; anonymous axes by order of appearance. The result uses the axis order of the first
    array.

    Args:
        arrays: The arrays to concatenate.
        axis: The axis to join along. Selectors are resolved against the first array. Defaults to the first axis.
        dtype: Optional dtype of the result.
        out_sharding: Optional sharding specification for the output array.

    Returns:
        A ``NamedArray`` whose size along ``axis`` is the sum of the input sizes.
    """
    named = util.ensure_named_tuple("concatenate", tuple(arrays))
    if not named:
        raise ValueError("concatenate requires at least one array")
    if not named[0].axes:
        raise ValueError("concatenate: zero-dimensional arrays cannot be concatenated")

    idx = axis_index(named[0].axes, axis)
    if idx is None:
        raise ValueError(f"concatenate: axis {axis!r} not found in {named[0].axes}")

    ref, jax_arrays = _canonicalize_arrays("concatenate", named, free=idx)
    jax_array = jnp.concatenate(jax_arrays, axis=idx, dtype=dtype)

    out_axes = ref[:idx] + (ref[idx].resize(jax_array.shape[idx]),) + ref[idx + 1 :]
    return constructors.array(jax_array, out_axes, out_sharding=out_sharding)


def concat(
    arrays: Sequence[NamedArrayLike],
    axis: AxisSelector = 0,
    dtype: DTypeLike | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Alias of `concatenate`."""
    return concatenate(arrays, axis, dtype, out_sharding=out_sharding)


def choose(a: NamedArrayLike, choices: Sequence[NamedArrayLike], *, mode: str = "raise") -> NamedArray:
    """Construct an array by picking, element-wise, from a sequence of choice arrays.

    ``result[idx] = choices[a[idx]][idx]`` after broadcasting ``a`` and all ``choices`` to a common shape.

    Args:
        a: Integer array of indices into ``choices``.
        choices: The arrays to choose from; they must be broadcast-compatible with ``a`` and each other.
        mode: How to treat out-of-bounds indices: ``'raise'`` (default; not compatible with JAX transformations),
            ``'wrap'`` or ``'clip'``.

    Returns:
        A ``NamedArray`` with the broadcast axes of ``a`` and ``choices``.
    """
    choices = tuple(choices)
    if not choices:
        raise ValueError("choose requires at least one choice array")

    (a_b, *choices_b), common_axes = broadcast_arrays_and_shape(a, *choices)
    jax_array = jnp.choose(a_b.array, [c.array for c in choices_b], mode=mode)
    return constructors.array(jax_array, common_axes)


BlockSpec = NamedArrayLike | Sequence[NamedArrayLike]


def _block_depth(x: Any) -> int:
    if isinstance(x, (list, tuple)):
        if not x:
            raise ValueError("block: empty lists are not allowed")
        depths = {_block_depth(e) for e in x}
        if len(depths) != 1:
            raise ValueError("block: all lists must be nested to the same depth")
        return depths.pop() + 1
    return 0


def block(
    arrays: BlockSpec,
    axes: Sequence[AxisSelector] | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Assemble an array from nested lists of blocks.

    The list nesting depth ``d`` maps onto ``axes[d]`` (outermost list first): the innermost lists are
    concatenated along ``axes[-1]``, the next level along ``axes[-2]``, and so on.

    For example, a 2x2 block matrix is ``block([[A, B], [C, D]], axes=("i", "j"))``: ``A, B`` and ``C, D`` are
    joined along ``"j"`` and the two results are then joined along ``"i"``.

    Args:
        arrays: A block array, or arbitrarily nested lists (all the same depth) of blocks.
        axes: One axis selector per nesting level, outermost first. Defaults to the trailing axes in order
            (``(-depth, ..., -1)``), as in `numpy.block`. Each selector is resolved per concatenation against the
            first array in that group.
        out_sharding: Optional sharding specification for the output array.

    Returns:
        The assembled ``NamedArray``.
    """
    depth = _block_depth(arrays)
    if depth == 0:
        return util.ensure_named("block", arrays)

    if axes is None:
        axes = tuple(range(-depth, 0))
    elif len(axes) != depth:
        raise ValueError(f"block: `arrays` is nested {depth} deep but {len(axes)} axes were given")

    def build(x: Any, level: int, sharding: ShardingLike | None) -> NamedArray:
        if level == depth:
            return util.ensure_named("block", x)
        parts = [build(e, level + 1, None) for e in x]
        return concatenate(parts, axes[level], out_sharding=sharding)

    return build(arrays, 0, out_sharding)


# array value operations


def astype(
    a: NamedArrayLike, dtype: DTypeLike | None, *, copy: bool = False, device: Device | ShardingLike | None = None
) -> NamedArray:
    """Convert an array to a specified dtype.

    This is implemented via :func:`jax.lax.convert_element_type`, which may have slightly different behavior than
    :func:`numpy.astype` in some cases. In particular, the details of float-to-int and int-to-float casts are
    implementation dependent.

    Args:
        a: Input array to convert.
        dtype: Output datatype.
        copy: If ``True``, always return a copy. If ``False`` (default), only return a copy if necessary.
        device: Optionally specify the device to which the output will be committed.

    Returns:
        An array with the same shape as ``a``, containing values of the
        specified dtype.
    """
    a = util.ensure_named("astype", a)
    return eqx.tree_at(lambda x: x.array, a, replace_fn=lambda x: jnp.astype(x, dtype, copy=copy, device=device))


def allclose(
    a: NamedArrayLike,
    b: NamedArrayLike,
    rtol: NamedArrayLike = 1e-05,
    atol: NamedArrayLike = 1e-08,
    equal_nan: bool = False,
) -> NamedArray:
    r"""Check if two arrays are element-wise approximately equal within a tolerance.

    Essentially this function evaluates the following condition:

    .. math::

     |a - b| \le \mathtt{atol} + \mathtt{rtol} * |b|

    ``jnp.inf`` in ``a`` will be considered equal to ``jnp.inf`` in ``b``.

    Args:
        a: First input array to compare.
        b: Second input array to compare.
        rtol: Relative tolerance used for approximate equality. Default = 1e-05.
        atol: Absolute tolerance used for approximate equality. Default = 1e-08.
        equal_nan: Boolean. If ``True``, NaNs in ``a`` will be considered equal to NaNs in ``b``. Default is ``False``.

    Returns:
        Boolean scalar ``NamedArray` indicating whether the input arrays are element-wise approximately equal within
        the specified tolerances.
    """

    (a, b, rtol, atol), common_shape = broadcast_arrays_and_shape(a, b, rtol, atol)
    jax_array = jnp.allclose(a.array, b.array, rtol.array, atol.array, equal_nan)
    return constructors.asnamedarray(jax_array, common_shape)


def isclose(
    a: NamedArrayLike,
    b: NamedArrayLike,
    rtol: NamedArrayLike = 1e-05,
    atol: NamedArrayLike = 1e-08,
    equal_nan: bool = False,
) -> NamedArray:
    r"""Check if the elements of two arrays are approximately equal within a tolerance.

    Essentially this function evaluates the following condition:

    .. math::

     |a - b| \le \mathtt{atol} + \mathtt{rtol} * |b|

    ``jnp.inf`` in ``a`` will be considered equal to ``jnp.inf`` in ``b``.

    Args:
        a: First input array to compare.
        b: Second input array to compare.
        rtol: Relative tolerance used for approximate equality. Default = 1e-05.
        atol: Absolute tolerance used for approximate equality. Default = 1e-08.
        equal_nan: Boolean. If ``True``, NaNs in ``a`` will be considered equal to NaNs in ``b``. Default is ``False``.

    Returns:
        A `NamedArray` containing boolean values indicating whether the input arrays are element-wise approximately
        equal within the specified tolerances.
    """

    (a, b, rtol, atol), common_shape = broadcast_arrays_and_shape(a, b, rtol, atol)
    jax_array = jnp.isclose(a.array, b.array, rtol.array, atol.array, equal_nan)
    return constructors.asnamedarray(jax_array, common_shape)


def array_equal(a: NamedArrayLike, b: NamedArrayLike, equal_nan: bool = False) -> NamedArray:
    """Check if two `NamedArray`s are element-wise equal.

    Args:
        a: First input array.
        b: Second input array.
        equal_nan: If ``True``, NaNs in ``a`` are considered equal to NaNs in
            ``b``. Defaults to ``False``.

    Returns:
        A zero-dimensional boolean `NamedArray`.
    """

    a, b = util.ensure_named("array_equal", a, b)
    if set(a.axes) != set(b.axes):
        return constructors.array(False, ())
    b = transpose(b, a.axes)
    return constructors.array(jnp.array_equal(a.array, b.array, equal_nan), ())


def array_equiv(a: NamedArrayLike, b: NamedArrayLike) -> NamedArray:
    """Check if two arrays are broadcast-equivalent.

    Returns ``False`` if the inputs cannot be broadcast to a common shape.

    Args:
        a: First input array.
        b: Second input array.

    Returns:
        A zero-dimensional boolean `NamedArray`.
    """
    try:
        eq = ufuncs.equal(a, b)
    except ValueError:
        return constructors.array(False, ())
    return reductions.all(eq)


def clip(a: NamedArrayLike, *, min: NamedArrayLike | None = None, max: NamedArrayLike | None = None) -> NamedArray:
    """Clip array values to a specified range.

    Args:
        a: Array to be clipped.
        min: Optional minimum value of the clipped range; if ``None`` (default), the result will not be clipped to any
            minimum value. If specified, it should be broadcast-compatible with ``a`` and ``max``.
        max: Optional maximum value of the clipped range; if ``None`` (default), the result will not be clipped to any
        maximum value. If specified, it should be broadcast-compatible with ``a`` and ``min``.

    Returns:
        A ``NamedArray`` containing values from ``a``, with values smaller than ``min`` set to ``min``, and values
        larger than ``max`` set to ``max``. Wherever ``min`` is larger than ``max``, the value of ``max`` is returned.
    """
    a = util.ensure_named("clip", a)

    if any(iscomplexobj(t) for t in (a, min, max)):
        raise ValueError(
            "Clip received a complex value either through the input or the min/max keywords. Complex values have no "
            "ordering and cannot be clipped. Please convert to a real value or array by taking the real or imaginary "
            "components via jax.numpy.real/imag respectively."
        )

    if min is not None:
        a = ufuncs.maximum(min, a)
    if max is not None:
        a = ufuncs.minimum(max, a)

    return constructors.array(a.array, a.axes)


def round(a: NamedArrayLike, decimals: int = 0) -> NamedArray:
    """Round input evenly to the given number of decimals.

    Args:
        a: Input array or scalar.
        decimals: Number of decimal points to which the input needs to be rounded. It must be specified statically.
            Not implemented for ``decimals < 0``.

    Returns:
        A ``NamedArray`` containing the rounded values to the specified decimals with same shape and dtype as ``a``
    """

    a = util.ensure_named("round", a)
    return eqx.tree_at(lambda x: x.array, a, replace_fn=lambda x: jnp.round(x, decimals))


def around(a: NamedArrayLike, decimals: int = 0) -> NamedArray:
    """Alias of ``round``."""

    return round(a, decimals)


def compress(
    condition: NamedArrayLike,
    a: NamedArrayLike,
    axis: AxisSelector | None = None,
    *,
    size: int | None = None,
    fill_value: NamedArrayLike = 0.0,
) -> NamedArray:
    """Select the slices of ``a`` along an axis where a 1D boolean ``condition`` is true.

    Axis sizes are static in lynqx, so the number of selected slices must be known when the result is built.
    Outside of `jax.jit` it is computed from ``condition``; inside `jax.jit` pass ``size`` explicitly (extra
    slots are filled with ``fill_value``, surplus selections are dropped).

    Args:
        condition: Rank-1 boolean array. Entries beyond its length are treated as ``False``.
        a: The array to select from.
        axis: The axis of ``a`` to select along. If omitted, ``a`` must be 1D, or ``condition``'s axis must be
            named so it can be looked up by name in ``a``.
        size: The static number of selected slices.
        fill_value: Value used to pad the result when ``size`` exceeds the number of selected slices.

    Returns:
        A ``NamedArray`` like ``a`` with the selected axis resized to the number of selected slices.
    """
    cond, named, fill_value = util.ensure_named("compress", condition, a, fill_value)
    if cond.ndim != 1:
        raise ValueError(f"compress: `condition` must be 1D, got {cond.ndim} axes")

    if axis is None:
        if named.ndim == 1:
            axis = 0
        elif cond.axes[0].name is not None:
            axis = cond.axes[0].name
        else:
            raise ValueError("compress: `axis` is required unless `a` is 1D or `condition` has a named axis")

    idx = axis_index(named.axes, axis)
    if idx is None:
        raise ValueError(f"compress: axis {axis!r} not found in {named.axes}")

    cond_name, target = cond.axes[0].name, named.axes[idx]
    if cond_name is not None and target.name is not None and cond_name != target.name:
        raise ValueError(f"compress: `condition` axis {cond_name!r} does not match the selected axis {target.name!r}")

    if size is None:
        try:
            size = int(jnp.count_nonzero(cond.array))
        except jax.errors.ConcretizationTypeError as err:
            raise ValueError("compress: pass an explicit `size` when calling under `jax.jit`") from err
    if size < 1:
        raise ValueError(f"compress: result would have {size} slices along {target.label!r}; axes must be non-empty")

    jax_array = jnp.compress(cond.array, named.array, axis=idx, size=size, fill_value=fill_value.array)
    out_axes = named.axes[:idx] + (target.resize(size),) + named.axes[idx + 1 :]
    return constructors.array(jax_array, out_axes)


@overload
def where(condition: NamedArrayLike, x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray: ...


@overload
def where(
    condition: NamedArrayLike,
    x: None = None,
    y: None = None,
    /,
    *,
    size: int | Axis | None = None,
    fill_value: Any = None,
) -> tuple[NamedArray, ...]: ...


@overload
def where(
    condition: NamedArrayLike,
    x: NamedArrayLike | None = None,
    y: NamedArrayLike | None = None,
    /,
    *,
    size: int | Axis | None = None,
    fill_value: Any = None,
) -> NamedArray | tuple[NamedArray, ...]: ...


def where(
    condition: NamedArrayLike,
    x: NamedArrayLike | None = None,
    y: NamedArrayLike | None = None,
    /,
    *,
    size: int | Axis | None = None,
    fill_value: Any = None,
) -> NamedArray | tuple[NamedArray, ...]:
    """Select elements by condition, or find the indices where a condition holds.

    Three-argument form: ``where(condition, x, y)``: broadcast the three arguments and take elements of ``x``
    where ``condition`` is true and of ``y`` elsewhere. ``size`` and ``fill_value`` are not allowed here.

    One-argument form: ``where(condition)``: like `numpy.nonzero`, return one integer index array per axis of
    ``condition`` (in ``condition.axes`` order). Each index array has a single new axis enumerating the true
    elements.

    Axis sizes are static in lynqx, so the number of true elements must be known when the result is built.
    Outside of `jax.jit` it is computed from ``condition``; inside `jax.jit` pass ``size`` explicitly.

    Args:
        condition: Boolean (or truthy) array.
        x: Values where ``condition`` is true (three-argument form only).
        y: Values where ``condition`` is false (three-argument form only).
        size: One-argument form only. The static length of the index axis, as an int (anonymous axis) or an `Axis`
            (to also name it). Surplus true elements are dropped; missing slots are filled with ``fill_value``.
        fill_value: One-argument form only. Index used to pad up to ``size``: a scalar, or one scalar per axis of
            ``condition``. Defaults to 0.

    Returns:
        A ``NamedArray`` with the broadcast axes of the inputs (three-argument form), or a tuple of integer
        ``NamedArray`` index arrays (one-argument form).
    """
    if x is None and y is None:
        return _where_indices(condition, size, fill_value)

    if x is None or y is None:
        raise ValueError("where: either both or neither of `x` and `y` must be provided")
    if size is not None or fill_value is not None:
        raise ValueError("where: `size` and `fill_value` cannot be used in the three-argument form")

    condition, x, y = broadcast_arrays(condition, x, y)

    jax_array = jnp.where(condition.array, x.array, y.array)
    return constructors.array(jax_array, x.axes)


def _where_indices(condition: NamedArrayLike, size: int | Axis | None, fill_value: Any) -> tuple[NamedArray, ...]:
    cond = util.ensure_named("where", condition)
    if cond.ndim == 0:
        raise ValueError("where: the one-argument form is not defined for zero-dimensional conditions")

    if isinstance(size, Axis):
        index_axis = size
    else:
        if size is None:
            try:
                size = int(jnp.count_nonzero(cond.array))
            except jax.errors.ConcretizationTypeError as err:
                raise ValueError("where: pass an explicit `size` when calling under `jax.jit`") from err
        elif isinstance(size, bool) or not isinstance(size, int):
            raise TypeError(f"where: `size` must be an int or Axis, got {type(size).__name__}")
        if size < 1:
            raise ValueError(f"where: result would have {size} elements; axes must be non-empty")
        index_axis = Axis(size)

    indices = jnp.nonzero(cond.array, size=index_axis.size, fill_value=fill_value)
    return tuple(constructors.array(i, (index_axis,)) for i in indices)


def angle(a: NamedArrayLike, deg: bool = False) -> NamedArray:
    """Return the angle of a complex-valued number or array.

    Args:
        a: A complex number or an array of complex numbers.
        deg: If ``True``, returns the result in degrees, otherwise returns in radians. Defaults to ``False``.

    Returns:
        A ``NamedArray`` of counterclockwise angles of each element of ``a``, with the same shape as ``a`` and dtype
        ``float``.
    """

    a = util.ensure_named("angle", a)
    return eqx.tree_at(lambda x: x.array, a, replace_fn=lambda x: jnp.angle(x, deg))


def conj(a: NamedArrayLike) -> NamedArray:
    """Alias of ``conjugate``."""
    a = util.ensure_named("conj", a)
    return eqx.tree_at(lambda x: x.array, a, replace_fn=lambda x: jnp.conj(x))


def conjugate(a: NamedArrayLike) -> NamedArray:
    """Return element-wise complex-conjugate of the input.

    Args:
        a: Input array or scalar

    Returns:
        A ``NmaedArray`` containing the complex-conjugate of ``a``.
    """

    a = util.ensure_named("conjugate", a)
    return eqx.tree_at(lambda x: x.array, a, replace_fn=lambda x: jnp.conjugate(x))


# sorting


def sort(a: NamedArrayLike) -> NamedArray:
    """Return a sorted copy of an array along its last axis.

    Args:
        a: The input array.

    Returns:
        A `NamedArray` with elements sorted along the last axis.
    """

    a = util.ensure_named("sort", a)
    return eqx.tree_at(lambda x: x.array, a, replace_fn=lambda x: jnp.sort(x))


def sort_complex(a: NamedArrayLike) -> NamedArray:
    """Sort a complex array using the real part first, then the imaginary part.

    Args:
        a: The input array.

    Returns:
        A `NamedArray` with elements sorted along the last axis.
    """

    a = util.ensure_named("sort_complex", a)
    return eqx.tree_at(lambda x: x.array, a, replace_fn=lambda x: jnp.sort_complex(x))


def top_k(
    a: NamedArrayLike, k: int, axis: AxisSelector, new_axis: AxisLike | None = None
) -> tuple[NamedArray, NamedArray]:
    """Return the top ``k`` values and their indices along an axis.

    Args:
        a: The input array.
        k: The number of top elements to return.
        axis: The axis along which to find the top ``k`` elements.
        new_axis: The axis descriptor for the output. If a string, creates an
            anonymous axis of size ``k`` with that name; if ``None``, the
            existing axis is resized to ``k``.

    Returns:
        A tuple ``(values, indices)`` of `NamedArray`s sharing the same axes.
    """

    named = util.ensure_named("top_k", a)

    idx = axis_index(named.axes, axis)
    if idx is None:
        raise ValueError(f"Axis {axis} not found in {a}")

    values, indices = jax.lax.top_k(named.array, k=k, axis=idx)

    if new_axis is None:
        axis = resolve_axes(named.axes, axis)[0]
        new_axis = axis.resize(k)
    elif isinstance(new_axis, str):
        new_axis = Axis(k, new_axis)

    updated_axes = named.axes[:idx] + (new_axis,) + named.axes[idx + 1 :]
    return constructors.array(values, updated_axes), constructors.array(indices, updated_axes)


def searchsorted(
    a: NamedArrayLike,
    v: NamedArrayLike,
    side: str = "left",
    sorter: NamedArrayLike | None = None,
    *,
    method: str = "scan",
) -> NamedArray:
    """Find indices where elements should be inserted to keep ``a`` sorted.

    Args:
        a: A 1-D array of sorted values to search in.
        v: The values whose insertion indices are sought.
        side: ``'left'`` or ``'right'``. Defaults to ``'left'``.
        sorter: Optional array of indices that sorts ``a``.
        method: Search method passed to :func:`jax.numpy.searchsorted`. Defaults to ``'scan'``.

    Returns:
        A `NamedArray` with the same axes as ``v``.
    """
    if sorter is None:
        a, v = util.ensure_named("searchsorted", a, v)
    else:
        a, v, sorter = util.ensure_named("searchsorted", a, v, sorter)

    if a.ndim != 1:
        raise ValueError("`seachsorted` only supports rank-1 `a`.")

    jax_sorter = sorter.array if sorter is not None else None
    jax_array = jnp.searchsorted(a.array, v.array, side=side, sorter=jax_sorter, method=method)

    return constructors.array(jax_array, v.axes)


# windows
def bartlett(M: Axis) -> NamedArray:
    """Return the Bartlett window of length ``M.size`` on axis ``M``.

    Args:
        M: The axis of the window.

    Returns:
        A `NamedArray` of length ``M.size`` on axis ``M``.
    """
    return constructors.array(jnp.bartlett(M.size), M)


def blackman(M: Axis) -> NamedArray:
    """Return the Blackman window of length ``M.size`` on axis ``M``.

    Args:
        M: The axis of the window.

    Returns:
        A `NamedArray` of length ``M.size`` on axis ``M``.
    """
    return constructors.array(jnp.blackman(M.size), M)


def hanning(M: Axis) -> NamedArray:
    """Return the Hann window of length ``M.size`` on axis ``M``.

    Args:
        M: The axis of the window.

    Returns:
        A `NamedArray` of length ``M.size`` on axis ``M``.
    """
    return constructors.array(jnp.hanning(M.size), M)


def kaiser(M: Axis, beta: NamedArrayLike) -> NamedArray:
    """Return the Kaiser window of length ``M.size`` on axis ``M``.

    Args:
        M: The axis of the window.
        beta: The shape parameter of the window.

    Returns:
        A `NamedArray` of length ``M.size`` on axis ``M``.
    """
    beta = util.ensure_named("kaiser", beta)
    return constructors.array(jnp.kaiser(M.size, beta.array), M)
