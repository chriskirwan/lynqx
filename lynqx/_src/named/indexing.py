from collections.abc import Mapping
from dataclasses import dataclass
from types import EllipsisType
from typing import cast

import jax.lax
import jax.numpy as jnp
import numpy as np
from jax import Array

from lynqx._src.axis_util import axis_index, check_unique_axis_names
from lynqx._src.named import constructors, operations, util
from lynqx._src.sharding import canonicalize_sharding
from lynqx._src.typing import Axis, AxisIndex, AxisSelector, NamedArray, NamedArrayLike, NamedIndex, ShardingLike


StaticInt = int | np.integer
StaticScalar = np.bool_ | np.number | bool | int | float | complex


def _resolve_index(axes: tuple[Axis, ...], selector: AxisSelector, fn_name: str) -> int:
    position = axis_index(axes, selector)
    if position is None:
        raise ValueError(f"{fn_name}: axis {selector} not found in {axes}")
    return position


def _resolve_index_map(axes: tuple[Axis, ...], selectors: Mapping[AxisSelector, object], fn_name: str):
    resolved: dict[int, object] = {}
    for selector, value in selectors.items():
        position = _resolve_index(axes, selector, fn_name)
        if position in resolved:
            raise ValueError(f"{fn_name}: multiple selectors resolve to axis {axes[position]}.")
        resolved[position] = value
    return resolved


def _slice_output_size(s: slice, axis: Axis, fn_name: str) -> int:
    """Static length of `axis` after applying `s` -- and the one place a traced
    slice bound gets caught and redirected, rather than failing deeper inside
    `jnp` with a less specific error."""
    try:
        return len(range(*s.indices(axis.size)))
    except TypeError as e:
        raise TypeError(
            f"{fn_name}: slice bounds on axis {axis.label} must be static Python "
            f"ints, got {s}. A traced/dynamic start isn't expressible as a Python "
            f"`slice` -- use `dynamic_slice`/`dynamic_update_slice` instead."
        ) from e


def _expand_positional_key(
    key: tuple[AxisIndex | Array | EllipsisType, ...], ndim: int, fn_name: str
) -> tuple[AxisIndex | Array, ...]:
    ellipsis_pos: int | None = None
    concrete: list[AxisIndex | Array] = []
    for i, k in enumerate(key):
        if k is Ellipsis:
            if ellipsis_pos is not None:
                raise IndexError(f"{fn_name}: only one Ellipsis (`...`) is allowed.")
            ellipsis_pos = i
        else:
            concrete.append(k)

    if len(concrete) > ndim:
        raise IndexError(f"{fn_name}: too many indices ({len(concrete)}) for {ndim} axes.")

    if ellipsis_pos is None:
        return tuple(concrete) + (slice(None),) * (ndim - len(concrete))

    fill = ndim - len(concrete)
    return tuple(concrete[:ellipsis_pos]) + (slice(None),) * fill + tuple(concrete[ellipsis_pos:])


def _check_axis_names_agree(
    a_axes: tuple[Axis, ...], b_axes: tuple[Axis, ...], fn_name: str, *, skip: frozenset[int] = frozenset()
) -> None:
    if len(a_axes) != len(b_axes):
        raise ValueError(f"{fn_name}: axis count mismatch, {len(a_axes)} vs {len(b_axes)}.")
    for i, (a, b) in enumerate(zip(a_axes, b_axes)):
        if i in skip:
            continue
        if a.name is not None and b.name is not None and a.name != b.name:
            raise ValueError(f"{fn_name}: axis {i} names disagree ({a.name} vs {b.name})")


def _wrap_bare_array(value) -> NamedArray:
    arr = jnp.asarray(value)
    return constructors.array(arr, tuple(Axis(s) for s in arr.shape))


def _arange_axis(axis: Axis) -> NamedArray:
    return constructors.array(jnp.arange(axis.size), (axis,))


def _advanced_output_axes(
    axes: tuple[Axis, ...],
    per_axis: list[AxisIndex],
    kept_mask: list[bool],
    broadcast_axes: tuple[Axis, ...],
    fn_name: str,
) -> tuple[Axis, ...]:

    def resized(i: int) -> Axis:
        index = per_axis[i]
        if not isinstance(index, slice):
            raise TypeError(f"{fn_name}: expected a slice for retained axis, got {index}.")
        return axes[i].resize(_slice_output_size(index, axes[i], fn_name))

    consumed = [i for i, k in enumerate(kept_mask) if not k]
    first, last = consumed[0], consumed[-1]
    is_contiguous = (last - first + 1) == len(consumed)

    if is_contiguous:
        before = tuple(resized(i) for i in range(first) if kept_mask[i])
        after = tuple(resized(i) for i in range(last + 1, len(axes)) if kept_mask[i])
        return before + broadcast_axes + after
    else:
        kept = tuple(resized(i) for i in range(len(axes)) if kept_mask[i])
        return broadcast_axes + kept


def _resolve_advanced_index(
    axes: tuple[Axis, ...], per_axis: list[AxisIndex], fn_name: str
) -> tuple[tuple[AxisIndex | Array, ...], tuple[Axis, ...]]:
    kept_mask = [isinstance(v, slice) for v in per_axis]
    advanced_names = {ax.name for v in per_axis if isinstance(v, NamedArray) for ax in v.axes if ax.name is not None}
    if advanced_names:
        for i, axis in enumerate(axes):
            if kept_mask[i] and per_axis[i] == slice(None) and axis.name in advanced_names:
                per_axis[i] = _arange_axis(axis)
                kept_mask[i] = False

    advanced_entries = [v for v in per_axis if isinstance(v, NamedArray)]
    broadcasted = operations.broadcast_arrays(*advanced_entries)
    broadcast_axes = broadcasted[0].axes if broadcasted else ()

    kept_names = {axes[i].name for i, k in enumerate(kept_mask) if k and axes[i].name is not None}
    collision = kept_names & {ax.name for ax in broadcast_axes if ax.name is not None}
    if collision:
        raise ValueError(
            f"{fn_name}: axis name(s) {collision} used by both a surviving "
            f"axis and an advanced index -- rename one of them."
        )

    adv_iter = iter(broadcasted)
    positional: tuple[AxisIndex | Array, ...] = tuple(
        next(adv_iter).array if isinstance(v, NamedArray) else v for v in per_axis
    )

    output_axes = _advanced_output_axes(axes, per_axis, kept_mask, broadcast_axes, fn_name)
    check_unique_axis_names(output_axes)
    return positional, output_axes


def resolve_index(axes: tuple[Axis, ...], slices: NamedIndex, *, fn_name: str):
    """The single place index resolution happens for both `__getitem__` and
    `.at[]`.

    Returns `(positional, output_axes)`: `positional` is a plain tuple of
    `int`/`slice` values, one per axis of `axes`, ready to hand straight to
    the underlying JAX array's own `__getitem__`/`.at[]`. `output_axes` is
    what the *result* of that indexing is named -- an axis hit with an `int`
    is dropped; an axis hit with a `slice` is kept, resized to that slice's
    (static) length.
    """
    ndim = len(axes)

    if isinstance(slices, Mapping):
        per_axis: list[AxisIndex] = [slice(None)] * ndim
        seen: dict[int, AxisSelector] = {}
        for selector, value in slices.items():
            position = _resolve_index(axes, selector, fn_name)
            if position in seen:
                raise ValueError(
                    f"{fn_name}: both {seen[position]} and {selector} resolve to "
                    f"axis {axes[position]} -- give each axis only one index."
                )
            seen[position] = selector
            if not isinstance(value, AxisIndex):
                raise TypeError(
                    f"{fn_name}: index for axis {axes[position]} must be an int, slice, "
                    f"or NamedArray -- got {type(value).__name__}. Bare arrays are only "
                    "accepted in the positional-tuple index form."
                )
            per_axis[position] = value
    else:
        tup = slices if isinstance(slices, tuple) else (slices,)
        expanded = _expand_positional_key(tup, ndim, fn_name)
        per_axis = [v if isinstance(v, AxisIndex) else _wrap_bare_array(v) for v in expanded]

    if any(isinstance(v, NamedArray) for v in per_axis):
        return _resolve_advanced_index(axes, per_axis, fn_name)

    output_axes: list[Axis] = []
    for axis, value in zip(axes, per_axis):
        if isinstance(value, slice):
            output_axes.append(axis.resize(_slice_output_size(value, axis, fn_name)))
        # an `int` value consumes the axis -- nothing appended for it

    return tuple(per_axis), tuple(output_axes)


def dynamic_slice(
    array: NamedArray,
    start: Mapping[AxisSelector, NamedArrayLike],
    length: Mapping[AxisSelector, StaticInt],
):
    """NamedArray-aware `jax.lax.dynamic_slice`: a traced start position with
    a static length -- the one thing ordinary `array[...]`/`.at[]` indexing
    structurally can't express, since JAX requires static slice bounds there.

    Axes named in `start` must also appear in `length`, and vice versa. Axes
    of `array` mentioned in neither are read in full, unmodified, starting at
    index 0 -- matching ordinary slicing's "axes you don't mention are left
    alone" convention.

    Unlike ordinary slicing, an out-of-bounds start is not an error here: per
    `jax.lax.dynamic_slice`'s own contract, it's silently clamped into bounds
    instead. That's `lax`'s behavior, not overridden by this wrapper.
    """
    start_positions = _resolve_index_map(array.axes, start, "dynamic_slice")
    length_positions = _resolve_index_map(array.axes, length, "dynamic_slice")

    if start_positions.keys() != length_positions.keys():
        only_start = {array.axes[p] for p in start_positions.keys() - length_positions.keys()}
        only_length = {array.axes[p] for p in length_positions.keys() - start_positions.keys()}
        raise ValueError(
            f"dynamic_slice: `start` and `length` must name the same axes. "
            f"Only in `start`: {only_start or None}; only in `length`: {only_length or None}."
        )

    starts: list[int | Array] = [0] * len(array.axes)
    sizes: list[int] = [ax.size for ax in array.axes]
    new_axes = list(array.axes)

    for position, s in start_positions.items():
        starts[position] = util.scalar_namedarray_to_jax_scalar(s)
        size = cast(int, length_positions[position])
        sizes[position] = size
        new_axes[position] = array.axes[position].resize(size)

    sliced = jax.lax.dynamic_slice(array.array, starts, sizes)
    return constructors.array(sliced, tuple(new_axes))


def dynamic_update_slice(array: NamedArray, update: NamedArrayLike, start: Mapping[AxisSelector, NamedArrayLike]):
    """The write-side counterpart to `dynamic_slice`: `jax.lax.dynamic_update_slice`
    with a traced start. `update`'s own shape (not `start`) determines how much
    of `array` gets overwritten along every axis -- same as `lax`'s own
    primitive -- and axes not named in `start` are written beginning at index 0.
    """
    update = util.ensure_named("dynamic_update_slice", update)
    _check_axis_names_agree(array.axes, update.axes, "dynamic_update_slice")

    starts: list[int | Array] = [0] * array.array.ndim
    for position, s in _resolve_index_map(array.axes, start, "dynamic_update_slice").items():
        starts[position] = util.scalar_namedarray_to_jax_scalar(s)

    updated = jax.lax.dynamic_update_slice(array.array, update.array, starts)
    return constructors.array(updated, array.axes)


class NamedIndexUpdateHelper:
    __slots__ = ("array",)

    def __init__(self, array: NamedArray):
        self.array = array

    def __getitem__(self, index: NamedIndex) -> "NamedIndexUpdateRef":
        positional, output_axes = resolve_index(self.array.axes, index, fn_name="NamedArray.at")
        return NamedIndexUpdateRef(self.array, positional, output_axes)


@dataclass(frozen=True)
class NamedIndexUpdateRef:
    source: NamedArray
    positional: tuple[AxisIndex, ...]
    output_axes: tuple[Axis, ...]

    def get(self, *, out_sharding: ShardingLike | None = None, **kwargs) -> NamedArray:
        jax_sharding = canonicalize_sharding(out_sharding, self.output_axes, "NamedArray.at.get")
        jax_array = self.source.array.at[self.positional].get(out_sharding=jax_sharding, **kwargs)

        return constructors.array(jax_array, self.output_axes)

    def set(self, value: NamedArrayLike, **kwargs) -> NamedArray:
        return self._update("set", value, **kwargs)

    def apply(self, fn, **kwargs) -> NamedArray:
        updated = self.source.array.at[self.positional].apply(fn, **kwargs)
        return constructors.array(updated, self.source.axes)

    def add(self, value: NamedArrayLike, **kwargs) -> NamedArray:
        return self._update("add", value, **kwargs)

    def multiply(self, value: NamedArrayLike, **kwargs) -> NamedArray:
        return self._update("multiply", value, **kwargs)

    def min(self, value: NamedArrayLike, **kwargs) -> NamedArray:
        return self._update("min", value, **kwargs)

    def max(self, value: NamedArrayLike, **kwargs) -> NamedArray:
        return self._update("max", value, **kwargs)

    def _update(self, method_name: str, value: NamedArrayLike, **kwargs):
        named_value = util.ensure_named(f"NamedArray.at.{method_name}", value)
        named_value = operations.broadcast_to(named_value, self.output_axes)

        updated = getattr(self.source.array.at[self.positional], method_name)(named_value.array, **kwargs)
        return constructors.array(updated, self.source.axes)


def take(
    array: NamedArray,
    axis: AxisSelector,
    index: NamedArrayLike,
    *,
    mode: str | None = None,
    fill_value: StaticScalar | None = None,
    unique_indices: bool = False,
    indices_are_sorted: bool = False,
) -> NamedArray:
    """Gather along a single named axis -- `jnp.take` with axis names.

    `index`'s own axes replace `axis` in the output, in the position `axis`
    occupied; every other axis of `array` is untouched. Fully jit-compatible:
    `index`'s *values* may be traced, since output shape depends only on
    `index`'s (static) shape, not its contents.

    Unlike `take_along_axis`, `index` is not required to share `array`'s
    rank or shape elsewhere -- it can be any shape, including one that
    introduces axes `array` didn't have at all (an embedding-table lookup:
    `array` is `(vocab,)`, `index` is `(batch, seq)`, output is
    `(batch, seq)`). If `index` happens to carry an axis name that collides
    with one of `array`'s *other* (non-gathered) axes, that's a genuine
    ambiguity in the output and raises, via the same uniqueness check
    `axis_util.check_unique_axis_names` uses elsewhere.
    """
    position = _resolve_index(array.axes, axis, "take")
    index = util.ensure_named("take", index)

    result = jnp.take(
        array.array,
        index.array,
        axis=position,
        mode=mode,
        fill_value=fill_value,
        unique_indices=unique_indices,
        indices_are_sorted=indices_are_sorted,
    )
    new_axes = array.axes[:position] + index.axes + array.axes[position + 1 :]
    check_unique_axis_names(new_axes)
    return constructors.array(result, new_axes)


def take_along_axis(
    array: NamedArray,
    indices: NamedArrayLike,
    axis: AxisSelector,
    *,
    mode: str | None = None,
) -> NamedArray:
    """`jnp.take_along_axis` with axis names -- the natural companion to
    `reductions.argsort`/`argmax`/`argmin`/`nanargmax`/`nanargmin`: gather the
    values an index array points to, one gathered element per output
    position, rather than `take`'s "replace the axis with the index array's
    own shape" semantics. The classic pattern this exists for:
    `take_along_axis(values, reductions.argsort(values, axis="t"), axis="t")`.

    `indices` must have the same rank as `array`; every axis except `axis`
    must agree in size with `array` there (only `axis` itself may differ --
    that difference is what determines the output's size along `axis`). Axis
    *names* are cross-checked positionally the same way `dynamic_update_slice`
    checks `update` against `array`: a name mismatch at a shared position
    raises, but either side may be anonymous, and the position at `axis`
    itself is exempt from the check since resizing it is the entire point.
    """
    position = _resolve_index(array.axes, axis, "take_along_axis")
    indices = util.ensure_named("take_along_axis", indices)
    _check_axis_names_agree(array.axes, indices.axes, "take_along_axis", skip=frozenset({position}))

    result = jnp.take_along_axis(array.array, indices.array, axis=position, mode=mode)
    new_size = indices.array.shape[position]
    new_axes = array.axes[:position] + (array.axes[position].resize(new_size),) + array.axes[position + 1 :]
    return constructors.array(result, new_axes)
