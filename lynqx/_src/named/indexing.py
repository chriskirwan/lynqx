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


def resolve_index(
    axes: tuple[Axis, ...], slices: NamedIndex, *, fn_name: str
) -> tuple[tuple[AxisIndex | Array, ...], tuple[Axis, ...]]:
    """Resolve a user-supplied index against a tuple of named axes.

    This is the single place index resolution happens for both ``NamedArray.__getitem__`` and ``NamedArray.at[]``.
    Two index forms are accepted: a mapping from axis selectors (names or positional indices) to per-axis indexers,
    or a positional tuple (optionally containing a single ``...``) mirroring NumPy's indexing syntax. The function
    returns both the positional key that can be handed straight to the underlying JAX array and the axes that the
    *result* of that indexing carries.

    Args:
        axes: The axes of the array being indexed.
        slices: The user index. Either a `Mapping` from `AxisSelector` to an `int`/`slice`/`NamedArray`, or a
            positional `tuple`/scalar key.
        fn_name: Name used to prefix error messages.

    Returns:
        A tuple ``(positional, output_axes)``.
            - ``positional`` is a plain tuple of ``int``/``slice``/``Array`` values, one per axis of ``axes``, ready
            to pass to the underlying JAX array's ``__getitem__`` / ``.at[]``.
            - ``output_axes`` names the result: an axis indexed with an ``int`` is dropped, and an axis indexed with
                a ``slice`` or ``NamedArray`` is kept with its (statically-known) output size.

    Raises:
        ValueError: If a selector names an axis that does not exist, if two selectors target the same axis, or if a
            surviving axis name collides with an advanced-index axis name.
        TypeError: If a mapping value is not an ``int``, ``slice``, or `NamedArray`, or if a slice bound is not a
            static Python int.
        IndexError: If the positional key contains more than one ``...``, or supplies more indices than there are axes.
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
) -> NamedArray:
    """Slice a `NamedArray` at a traced start position with a static length.

    NamedArray-aware :func:`jax.lax.dynamic_slice`. This is the one slicing
    operation ordinary ``array[...]`` / ``array.at[...]`` structurally cannot
    express, because JAX requires static slice bounds there.

    Axes named in ``start`` must also appear in ``length``, and vice versa.
    Axes of ``array`` mentioned in neither are read in full, unmodified,
    starting at index 0 -- matching ordinary slicing's "axes you don't mention
    are left alone" convention.

    Unlike ordinary slicing, an out-of-bounds start is not an error here: per
    :func:`jax.lax.dynamic_slice`'s own contract, it is silently clamped into
    bounds instead. That is ``lax``'s behavior, not overridden by this wrapper.

    Args:
        array: The `NamedArray` to slice.
        start: A mapping from axis selectors to traced (or static) start
            positions.
        length: A mapping from the same axis selectors to static output sizes.

    Returns:
        A `NamedArray` whose named axes are resized to the corresponding ``length`` values and whose other axes are
        unchanged.

    Raises:
        ValueError: If ``start`` and ``length`` name different sets of axes.
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
    """Helper property for functionally-pure index updates on a `NamedArray`.

    The ``.at`` property provides a functionally pure equivalent of in-place
    array modification, with axis selection by name rather than position:

    ==================================  =========================================
    Alternate syntax                    Equivalent in-place expression
    ==================================  =========================================
    ``x = x.at[idx].get()``             ``x = x[idx]``
    ``x = x.at[idx].set(y)``            ``x[idx] = y``
    ``x = x.at[idx].add(y)``            ``x[idx] += y``
    ``x = x.at[idx].multiply(y)``       ``x[idx] *= y``
    ``x = x.at[idx].min(y)``            ``x[idx] = minimum(x[idx], y)``
    ``x = x.at[idx].max(y)``            ``x[idx] = maximum(x[idx], y)``
    ``x = x.at[idx].apply(fn)``         ``fn.at(x, idx)``
    ==================================  =========================================

    None of the ``x.at`` expressions modify the original ``x``; instead they
    return a modified copy of ``x``. However, inside a :func:`jax.jit`-compiled
    function, expressions like ``x = x.at[idx].set(y)`` are guaranteed to be
    applied in-place.

    Unlike NumPy in-place operations such as ``x[idx] += y``, if multiple
    indices refer to the same location, all updates are applied (NumPy would
    only apply the last update). The order in which conflicting updates are
    applied is implementation-defined and may be nondeterministic.

    The index ``idx`` may be either a mapping from axis selectors to indexers
    (as accepted by `NamedArray.__getitem__`) or a positional tuple. See
    `resolve_index` for the accepted forms.

    Examples:
        >>> import jax.numpy as jnp
        >>> import lynqx as lx
        >>> x = lxn.arange(lx.Axis(5, "i"))
        >>> x.at[{"i": 2}].get()
        f32[]  # value 2.0 if idx == 2
        >>> x.at[{"i": 2}].add(10)
        f32[x:]  # [0., 1., 12., 3., 4.]

    By default, out-of-bound indices are ignored in updates; this behavior can
    be controlled with the ``mode`` parameter on the update methods, and by
    ``mode``/``fill_value`` on ``get``.
    """

    __slots__ = ("array",)

    def __init__(self, array: NamedArray):
        self.array = array

    def __getitem__(self, index: NamedIndex) -> "NamedIndexUpdateRef":
        positional, output_axes = resolve_index(self.array.axes, index, fn_name="NamedArray.at")
        return NamedIndexUpdateRef(self.array, positional, output_axes)


@dataclass(frozen=True)
class NamedIndexUpdateRef:
    """A pending indexed update on a `NamedArray`.

    Instances are produced by ``NamedArray.at[index]`` and should not be
    constructed directly. Each method returns a *new* `NamedArray` with the
    update applied; the original array is never mutated.
    """

    source: NamedArray
    positional: tuple[AxisIndex | Array, ...]
    output_axes: tuple[Axis, ...]

    def get(self, *, out_sharding: ShardingLike | None = None, **kwargs) -> NamedArray:
        """Return the value of the indexed array: ``x[idx]``.

        The result's axes are those of ``x`` after applying the index -- axes
        consumed by an ``int`` index are dropped, and axes resized by a
        ``slice`` or advanced index are resized in the output. ``out_sharding``
        is therefore canonicalized against ``self.output_axes``.

        Args:
            out_sharding: Optional sharding specification for the output.
            **kwargs: Additional keyword arguments forwarded to
                :meth:`jax.Array.at[...].get` (e.g. ``mode``, ``fill_value``,
                ``indices_are_sorted``, ``unique_indices``, ``strategy``,
                ``wrap_negative_indices``). See :func:`jax.numpy.ndarray.at`.

        Returns:
            A `NamedArray` with the axes ``self.output_axes``.
        """
        jax_sharding = canonicalize_sharding(self.output_axes, out_sharding, "NamedArray.at.get")
        jax_array = self.source.array.at[self.positional].get(out_sharding=jax_sharding, **kwargs)
        return constructors.array(jax_array, self.output_axes)

    def set(self, value: NamedArrayLike, *, out_sharding: ShardingLike | None = None, **kwargs) -> NamedArray:
        """Return ``x`` with ``x[idx] = value`` applied: the pure form of ``x[idx] = value``.

        Args:
            value: The values to write. Broadcast against ``self.output_axes``
                (the shape of the indexed selection).
            out_sharding: Optional sharding specification for the output. The
                result has the *source*'s axes, so this is canonicalized
                against ``self.source.axes``.
            **kwargs: Additional keyword arguments forwarded to
                :meth:`jax.Array.at[...].set` (e.g. ``mode``,
                ``indices_are_sorted``, ``unique_indices``,
                ``wrap_negative_indices``). See :func:`jax.numpy.ndarray.at`.

        Returns:
            A `NamedArray` with the same axes as the source.
        """
        return self._update("set", value, out_sharding=out_sharding, **kwargs)

    def apply(self, fn, **kwargs) -> NamedArray:
        """Return ``x`` with ``fn.at(x, idx)`` applied: the pure form of ``fn.at(x, idx)``.

        ``fn`` is applied once per occurrence of ``idx``, so unlike
        ``x.at[idx].set(fn(x[idx]))`` repeated indices invoke ``fn`` multiple
        times. ``scatter_apply`` (and therefore this method) is not currently
        compatible with automatic differentiation.

        Note that unlike the other update methods, ``.apply`` does **not**
        accept ``out_sharding`` -- the underlying ``scatter_apply`` does not
        propagate sharding through its function argument.

        Args:
            fn: A unary elementwise function to apply at the indexed
                positions.
            **kwargs: Additional keyword arguments forwarded to
                :meth:`jax.Array.at[...].apply` (e.g. ``mode``,
                ``indices_are_sorted``, ``unique_indices``,
                ``wrap_negative_indices``). See :func:`jax.numpy.ndarray.at`.

        Returns:
            A `NamedArray` with the same axes as the source.
        """
        updated = self.source.array.at[self.positional].apply(fn, **kwargs)
        return constructors.array(updated, self.source.axes)

    def add(self, value: NamedArrayLike, *, out_sharding: ShardingLike | None = None, **kwargs) -> NamedArray:
        """Return ``x`` with ``x[idx] += value`` applied: the pure form of ``x[idx] += value``.

        Args:
            value: The values to add.
            out_sharding: Optional sharding specification for the output,
                canonicalized against ``self.source.axes``.
            **kwargs: Forwarded to :meth:`jax.Array.at[...].add`; see
                :func:`jax.numpy.ndarray.at`.

        Returns:
            A `NamedArray` with the same axes as the source.
        """
        return self._update("add", value, out_sharding=out_sharding, **kwargs)

    def multiply(self, value: NamedArrayLike, *, out_sharding: ShardingLike | None = None, **kwargs) -> NamedArray:
        """Return ``x`` with ``x[idx] *= value`` applied: the pure form of ``x[idx] *= value``.

        Args:
            value: The values to multiply by.
            out_sharding: Optional sharding specification for the output,
                canonicalized against ``self.source.axes``.
            **kwargs: Forwarded to :meth:`jax.Array.at[...].multiply`; see
                :func:`jax.numpy.ndarray.at`.

        Returns:
            A `NamedArray` with the same axes as the source.
        """
        return self._update("multiply", value, out_sharding=out_sharding, **kwargs)

    def min(self, value: NamedArrayLike, *, out_sharding: ShardingLike | None = None, **kwargs) -> NamedArray:
        """Return ``x`` with ``x[idx] = minimum(x[idx], value)`` applied.

        Args:
            value: The values to take the elementwise minimum against.
            out_sharding: Optional sharding specification for the output,
                canonicalized against ``self.source.axes``.
            **kwargs: Forwarded to :meth:`jax.Array.at[...].min`; see
                :func:`jax.numpy.ndarray.at`.

        Returns:
            A `NamedArray` with the same axes as the source.
        """
        return self._update("min", value, out_sharding=out_sharding, **kwargs)

    def max(self, value: NamedArrayLike, *, out_sharding: ShardingLike | None = None, **kwargs) -> NamedArray:
        """Return ``x`` with ``x[idx] = maximum(x[idx], value)`` applied.

        Args:
            value: The values to take the elementwise maximum against.
            out_sharding: Optional sharding specification for the output,
                canonicalized against ``self.source.axes``.
            **kwargs: Forwarded to :meth:`jax.Array.at[...].max`; see
                :func:`jax.numpy.ndarray.at`.

        Returns:
            A `NamedArray` with the same axes as the source.
        """
        return self._update("max", value, out_sharding=out_sharding, **kwargs)

    def _update(
        self,
        method_name: str,
        value: NamedArrayLike,
        *,
        out_sharding: ShardingLike | None = None,
        **kwargs,
    ) -> NamedArray:
        # `value` is broadcast against the indexed selection's shape...
        named_value = util.ensure_named(f"NamedArray.at.{method_name}", value)
        named_value = operations.broadcast_to(named_value, self.output_axes)

        jax_sharding = canonicalize_sharding(self.source.axes, out_sharding, f"NamedArray.at.{method_name}")

        updated = getattr(self.source.array.at[self.positional], method_name)(
            named_value.array, out_sharding=jax_sharding, **kwargs
        )
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
    """Gather along a single named axis.

    NamedArray-aware :func:`jax.numpy.take`. ``index``'s own axes replace ``axis`` in the output, in the position
    ``axis`` occupied; every other axis of ``array`` is untouched. Fully jit-compatible: ``index``'s *values* may be
    traced, since the output shape depends only on ``index``'s (static) shape, not its contents.

    Unlike `take_along_axis`, ``index`` is not required to share ``array``'s rank or shape elsewhere -- it can be any
    shape, including one that introduces axes ``array`` did not have at all (an embedding-table lookup: ``array``
    is ``(vocab,)``, ``index`` is ``(batch, seq)``, output is ``(batch, seq)``). If ``index`` happens to carry an axis
    name that collides with one of ``array``'s *other* (non-gathered) axes, that is a genuine ambiguity in the output
    and raises, via the same uniqueness check `axis_util.check_unique_axis_names` uses elsewhere.

    Args:
        array: The `NamedArray` to gather from.
        axis: The axis of ``array`` to gather along.
        index: The index array. Its axes replace ``axis`` in the output.
        mode: How to treat out-of-bounds indices. One of ``None`` (default), ``'fill'``, ``'clip'``, or
        ``'promise_in_bounds'``. See :func:`jax.numpy.take` for details.
        fill_value: Value to use for out-of-bounds lookups when ``mode='fill'``.
        unique_indices: If ``True``, assume ``index`` contains no duplicates. May allow a more efficient implementation
            on some backends.
        indices_are_sorted: If ``True``, assume ``index`` is sorted in ascending order. May allow a more efficient
            implementation on some backends.

    Returns:
        A `NamedArray` with ``axis`` replaced by ``index``'s axes in place.

    Raises:
        ValueError: If ``axis`` is not one of ``array``'s axes, or if ``index`` introduces an axis name that collides
        with a surviving axis of ``array``.
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
    """Gather values along an axis using an index array of the same rank.

    NamedArray-aware :func:`jax.numpy.take_along_axis`. This is the natural companion to
    ``argsort` / ``argmax`` / ``argmin`` /``nanargmax`` / ``nanargmin``: gather the values an index array points to,
    one gathered element per output position, rather than `take`'s "replace the axis with the index array's own shape"
    semantics. The classic pattern this exists for::

        take_along_axis(values, reductions.argsort(values, axis="t"), axis="t")

    ``indices`` must have the same rank as ``array``; every axis except ``axis`` must agree in size with ``array``
    there (only ``axis`` itself may differ -- that difference is what determines the output's size along ``axis``).
    Axis *names* are cross-checked positionally the same way `dynamic_update_slice` checks ``update`` against
    ``array``: a name mismatch at a shared position raises, but either side may be anonymous, and the position at
    ``axis`` itself is exempt from the check since resizing it is the entire point.

    Args:
        array: The `NamedArray` to gather from.
        indices: An index array with the same rank as ``array``. All axes except ``axis`` must match ``array`` in size
        and (where named) in name.
        mode: How to treat out-of-bounds indices. One of ``None`` (default), ``'fill'``, ``'clip'``, or
        ``'promise_in_bounds'``. See :func:`jax.numpy.take_along_axis` for details.

    Returns:
        A `NamedArray` with the same axes as ``array``, with the size of ``axis`` replaced by ``indices``' size along
        that position.

    Raises:
        ValueError: If ``axis`` is not one of ``array``'s axes, or if ``indices`` disagrees with ``array`` on axis
            name or size at a shared position other than ``axis``.
    """
    position = _resolve_index(array.axes, axis, "take_along_axis")
    indices = util.ensure_named("take_along_axis", indices)
    _check_axis_names_agree(array.axes, indices.axes, "take_along_axis", skip=frozenset({position}))

    result = jnp.take_along_axis(array.array, indices.array, axis=position, mode=mode)
    new_size = indices.array.shape[position]
    new_axes = array.axes[:position] + (array.axes[position].resize(new_size),) + array.axes[position + 1 :]
    return constructors.array(result, new_axes)
