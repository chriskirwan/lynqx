from collections.abc import Mapping
from dataclasses import dataclass
from types import EllipsisType

import numpy as np

from lynqx._src.axis_util import axis_index
from lynqx._src.named import constructors, operations, util
from lynqx._src.sharding import canonicalize_sharding
from lynqx._src.typing import Axis, AxisIndex, AxisSelector, NamedArray, NamedArrayLike, NamedIndex, ShardingLike


StaticInt = int | np.integer


def _resolve_index(axes: tuple[Axis, ...], selector: AxisSelector, fn_name: str) -> int:
    position = axis_index(axes, selector)
    if position is None:
        raise ValueError(f"{fn_name}: axis {selector} not found in {axes}")
    return position


def _resolve_index_map(axes: tuple[Axis, ...], selectors: Mapping[AxisSelector, AxisIndex], fn_name: str):
    resolved: dict[int, AxisIndex] = {}
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
    key: tuple[AxisIndex | EllipsisType, ...], ndim: int, fn_name: str
) -> tuple[AxisIndex, ...]:
    ellipsis_pos: int | None = None
    concrete: list[AxisIndex] = []
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
                raise TypeError(f"{fn_name}: unsupported index value {value!r} for axis {axes[position]}.")
            per_axis[position] = value
    else:
        tup = slices if isinstance(slices, tuple) else (slices,)
        per_axis = list(_expand_positional_key(tup, ndim, fn_name))

    output_axes: list[Axis] = []
    for axis, value in zip(axes, per_axis):
        if isinstance(value, slice):
            output_axes.append(axis.resize(_slice_output_size(value, axis, fn_name)))
        # an `int` value consumes the axis -- nothing appended for it

    return tuple(per_axis), tuple(output_axes)


def dynamic_slice(
    array: NamedArrayLike,
    start: Mapping[AxisSelector, NamedArrayLike],
    length: Mapping[AxisSelector, StaticInt],
):
    """NamedArray-aware `jax.lax.dynamic_slice`: a *traced* start position with
    a *static* length -- the one thing ordinary `array[...]`/`.at[]` indexing
    structurally can't express, since JAX requires static slice bounds there.

    Axes named in `start` must also appear in `length`, and vice versa. Axes
    of `array` mentioned in neither are read in full, unmodified, starting at
    index 0 -- matching ordinary slicing's "axes you don't mention are left
    alone" convention.

    Unlike ordinary slicing, an out-of-bounds start is not an error here: per
    `jax.lax.dynamic_slice`'s own contract, it's silently clamped into bounds
    instead. That's `lax`'s behavior, not overridden by this wrapper.
    """


def dynamic_update_slice(array: NamedArrayLike, update: NamedArrayLike, start: Mapping[AxisSelector, NamedArrayLike]):
    """The write-side counterpart to `dynamic_slice`: `jax.lax.dynamic_update_slice`
    with a traced start. `update`'s own shape (not `start`) determines how much
    of `array` gets overwritten along every axis -- same as `lax`'s own
    primitive -- and axes not named in `start` are written beginning at index 0.
    """


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
