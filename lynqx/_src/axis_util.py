from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from functools import wraps
from inspect import signature
from types import UnionType
from typing import Literal, NamedTuple, overload, ParamSpec, Sequence, TypeVar

from lynqx._src.axis import Axis, AxisDict, AxisDim, AxisLike, AxisSelection, AxisSelector, AxisShape, AxisSpec


# validation

P = ParamSpec("P")
T = TypeVar("T")


@overload
def validate_unique_axes(fn: Callable[P, T], *, arg_names: Iterable[str]) -> Callable[P, T]: ...


@overload
def validate_unique_axes(*, arg_names: Iterable[str]) -> Callable[[Callable[P, T]], Callable[P, T]]: ...


def validate_unique_axes(fn=None, *, arg_names: Iterable[str]):
    if fn is None:
        return lambda func: _validate_unique_axes(func, arg_names=arg_names)
    return _validate_unique_axes(fn, arg_names=arg_names)


def _validate_unique_axes(fn: Callable[P, T], *, arg_names: Iterable[str]) -> Callable[P, T]:
    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        sig = signature(fn)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        for name in arg_names:
            value = bound.arguments.get(name)
            if value is not None:
                check_unique_axis_names(bound.arguments[name])

        return fn(*args, **kwargs)

    return wrapper


def check_unique_axis_names(axes: AxisSpec):
    if isinstance(axes, (Axis, str)):
        axes = (axes,)

    axes = axis_spec_to_tuple(axes)

    if not isinstance(axes, Sequence) or isinstance(axes, (bytes, bytearray)):
        raise TypeError(f"Expected AxisSelector, got {type(axes).__name__}")

    name_to_indices = defaultdict(list)

    for i, axis in enumerate(axes):
        if not isinstance(axis, AxisLike):
            raise TypeError(f"Expected `Axis` or `str`  at index {i} in {axes}, but got {type(axis)}.")

        name = _axis_name(axis)
        if name is None:
            continue

        name_to_indices[name].append(i)

    duplicates = {name: indices for name, indices in name_to_indices.items() if len(indices) > 1}
    if duplicates:
        msg = ", ".join([f"'{n}' at {idx}" for n, idx in duplicates.items()])
        raise ValueError(f"Duplicate axis names detected: {msg}")


def is_axis_compatible(a: Axis, b: Axis, *, allow_broadcast: bool = False) -> bool:
    """Whether two axes represent the same concrete axis.

    Named axes require matching names and sizes.

    Anonymous axes have no semantic identity, so equality is structural:
    two anonymous axes are equal when their sizes match.

    Args:
        a: First axis to compare.
        b: Second axis to compare.
        allow_broadcast: optionally allow for a, b to be broadcast compatible, Defaults to `False`

    Returns:
        Whether the axes are compatible.
    """
    if a.name is None or b.name is None:
        if a.name is not None or b.name is not None:
            return False
        if allow_broadcast:
            return a.size == b.size or a.size == 1 or b.size == 1
        return a.size == b.size
    return a.name == b.name and a.size == b.size


# normalization
def _as_tuple(x, single: type | UnionType) -> tuple:
    """Convert a single value or iterable to a tuple.

    Args:
        x: Value or iterable to normalize.
        single: Type, or union of types, treated as one value.

    Returns:
        A tuple containing the normalized values.
    """
    return (x,) if isinstance(x, single) else tuple(x)


def axis_spec_to_tuple(axis: AxisSpec) -> tuple[AxisLike, ...]:
    """Normalize an axis specification to a tuple.

    Args:
        axis: Axis object, axis name, or iterable of axis specifications.

    Returns:
        A tuple of axis-like values.
    """
    return _as_tuple(axis, AxisLike)


def axis_shape_to_tuple(axis: AxisShape) -> tuple[Axis, ...]:
    """Normalize an axis shape to a tuple of axes.

    Mapping inputs are interpreted as size-to-name mappings and converted to
    ``Axis`` objects.

    Args:
        axis: Axis, iterable of axes, or mapping of sizes to names.

    Returns:
        A tuple of ``Axis`` objects.

    Raises:
        TypeError: If an iterable contains a non-``Axis`` value.
    """
    if isinstance(axis, Mapping):
        return tuple(Axis(size, name) for name, size in axis.items())
    axes = _as_tuple(axis, AxisDim)
    if not all(isinstance(axis, AxisDim) for axis in axes):
        raise TypeError("`axis` must only contain `Axis` or `int` values")

    return tuple(axis if isinstance(axis, Axis) else Axis(axis) for axis in axes)


def axis_selection_to_tuple(axis: AxisSelection) -> tuple[AxisSelector, ...]:
    """Normalize an axis selection to a tuple of selectors.

    Args:
        axis: Axis selector or iterable of selectors.

    Returns:
        A tuple of axis selectors.
    """
    return _as_tuple(axis, AxisSelector)


# properties


def _axis_name(axis: AxisLike) -> str | None:
    if isinstance(axis, Axis):
        return axis.name
    return axis


def axis_names(axis: Sequence[AxisLike]) -> tuple[str | None, ...]:
    """Return the names of a sequence of axes or axis-like values.

    Args:
        axis: Axis-like values to inspect.

    Returns:
        A tuple containing each axis name, or ``None`` for anonymous axes.
    """
    return tuple(_axis_name(ax) for ax in axis)


def axis_sizes(axis: Sequence[Axis] | AxisDict) -> tuple[int, ...]:
    """Return the sizes of axes in sequence or mapping form.

    Args:
        axis: Axes to inspect, or a mapping whose values are sizes.

    Returns:
        A tuple of axis sizes.
    """
    if isinstance(axis, Axis):
        return (axis.size,)
    if isinstance(axis, Mapping):
        return tuple(size for size in axis.values())
    return tuple(ax.size for ax in axis)


def is_named_axis(axis: AxisLike):
    return _axis_name(axis) is not None


def is_anonymous_axis(axis: AxisLike):
    return _axis_name(axis) is None


# indexing


def axis_index(selection: Sequence[Axis], axis: AxisSelector) -> int | None:
    """Find the position of an axis selected by name, position, or value.

    Integer selectors support negative indexing. A missing selector returns
    ``None``; a selector matching multiple axes raises an error.

    Args:
        selection: Axes in which to search.
        axis: Name, integer position, or axis value to find.

    Returns:
        The matching zero-based position, or ``None`` if no axis matches.

    Raises:
        ValueError: If the selector matches multiple axes.
    """
    if isinstance(axis, int):
        if axis < 0:
            axis += len(selection)
        if axis < 0 or axis >= len(selection):
            return None
        return axis

    if isinstance(axis, str):
        matches = [i for i, ax in enumerate(selection) if _axis_name(ax) == axis]
    else:
        matches = [i for i, ax in enumerate(selection) if ax == axis]

    if not matches:
        return None

    if len(matches) > 1:
        raise ValueError(
            f"`axis` {axis} is abmigious: it matches position {matches}. "
            f" Use a positional/integer selector for anonymous axes"
        )
    return matches[0]


@overload
def axis_indices(
    selection: Sequence[Axis], axis: AxisSelection, *, strict: Literal[False]
) -> tuple[int | None, ...]: ...


@overload
def axis_indices(
    selection: Sequence[Axis], axis: AxisSelection, *, strict: Literal[True] = True
) -> tuple[int, ...]: ...


@overload
def axis_indices(selection: Sequence[Axis], axis: AxisSelection, *, strict: bool) -> tuple[int | None, ...]: ...


def axis_indices(selection: Sequence[Axis], axis: AxisSelection, *, strict: bool = True) -> tuple[int | None, ...]:
    """Resolve one or more axis selectors to positions.

    Args:
        selection: Axes in which to search.
        axis: Selector or selectors to resolve.
        strict: Whether unresolved selectors should raise an error.

    Returns:
        A tuple of matching positions. Unresolved selectors produce ``None``
        when ``strict`` is false.

    Raises:
        ValueError: If a selector is ambiguous or unresolved in strict mode.
    """
    axis = axis_selection_to_tuple(axis)

    indices = tuple(axis_index(selection, ax) for ax in axis)

    if strict:
        none_positions = [i for i, idx in enumerate(indices) if idx is None]
        if none_positions:
            failed_spec = [selection[i] for i in none_positions]
            specs_str = ", ".join(f"'{spec}' (at position {pos})" for pos, spec in zip(none_positions, failed_spec))
            raise ValueError(f"Axes could not be resolved: {specs_str} in selection {selection}")

    return indices


def resolve_axes(selection: Sequence[Axis], axis: AxisSelection) -> tuple[Axis, ...]:
    """Resolve axis selectors to the corresponding axes.

    Args:
        selection: Axes in which to search.
        axis: Selector or selectors to resolve.

    Returns:
        A tuple of selected axes.

    Raises:
        ValueError: If a selector is ambiguous or cannot be resolved.
    """
    indices = axis_indices(selection, axis)

    return tuple(selection[idx] for idx in indices)


# matching, alignment and permutation


class AxisMatch(NamedTuple):
    """Pair a source-axis position with a target-axis position."""

    source: int
    target: int


class MatchAxisResult(NamedTuple):
    """Result of matching axes between two collections."""

    matches: tuple[AxisMatch, ...]
    unmatched_source: tuple[int, ...]
    unmatched_target: tuple[int, ...]


def match_axes(
    source: Sequence[Axis],
    target: Sequence[Axis],
    *,
    allow_positional_fallback: bool = False,
) -> MatchAxisResult:
    """Matches axes between source and target collections.

    1. Matches named axes first by name.
    2. If allow_positional_fallback is True, attempts positional matching on
       unmatched anonymous / remaining slots.

    Args:
        source: Axes to match from.
        target: Axes to match against.
        allow_positional_fallback: Whether to match remaining axes by
            position when at least one axis is anonymous.

    Returns:
        Matching positions and the unmatched source and target positions.
    """
    matches: list[AxisMatch] = []
    unmatched_src: list[int] = list(range(len(source)))
    unmatched_tgt: list[int] = list(range(len(target)))

    for tgt_idx in list(unmatched_tgt):
        tgt_name = _axis_name(target[tgt_idx])
        if tgt_name is None:
            continue

        for src_idx in list(unmatched_src):
            if _axis_name(source[src_idx]) == tgt_name:
                matches.append(AxisMatch(src_idx, tgt_idx))
                unmatched_src.remove(src_idx)
                unmatched_tgt.remove(tgt_idx)

    if allow_positional_fallback:
        for src_i in reversed(list(unmatched_src)):
            if not is_anonymous_axis(source[src_i]):
                continue
            tgt_i = next((t for t in reversed(unmatched_tgt) if is_anonymous_axis(target[t])), None)
            if tgt_i is None:
                continue
            if not is_axis_compatible(source[src_i], target[tgt_i], allow_broadcast=True):
                raise ValueError(
                    f"Cannot align anonymous axes at matching trailing position: size {source[src_i].size} vs "
                    f"size {target[tgt_i].size}"
                )
            matches.append(AxisMatch(source=src_i, target=tgt_i))
            unmatched_src.remove(src_i)
            unmatched_tgt.remove(tgt_i)

    # Sort matches by target index order to keep layout predictable
    matches.sort(key=lambda m: m.target)

    return MatchAxisResult(
        matches=tuple(matches),
        unmatched_source=tuple(unmatched_src),
        unmatched_target=tuple(unmatched_tgt),
    )


# operations


@validate_unique_axes(arg_names=("a", "b"))
def union_axes(a: Sequence[Axis], b: Sequence[Axis]) -> tuple[Axis, ...]:
    """Return the ordered union of two axis sequences.

    Named axes are included once, and anonymous axes are retained. Named axes
    with the same name must also have the same size.

    Args:
        a: First axis sequence.
        b: Second axis sequence.

    Returns:
        Axes from ``a`` followed by new axes from ``b``.

    Raises:
        ValueError: If a shared named axis has conflicting sizes.
    """
    named_a = {ax.name: ax for ax in a if ax.name is not None}
    seen_names = set(named_a)

    result = list(a)
    for axis in b:
        name = axis.name
        if name is None:
            result.append(axis)
            continue
        if name in seen_names:
            existing = named_a[name]
            if isinstance(existing, Axis) and isinstance(axis, Axis) and existing.size != axis.size:
                raise ValueError(f"Axis {name!r} has conflicting sizes in `a` and `b`: {existing.size} vs {axis.size}")
            continue
        result.append(axis)
        seen_names.add(name)

    return tuple(result)


@validate_unique_axes(arg_names=("a", "b"))
def intersect_axes(a: Sequence[Axis], b: Sequence[Axis]) -> tuple[Axis, ...]:
    """Return axes from ``a`` whose names occur in ``b``.

    Matching named axes must have the same size. Anonymous axes are omitted.

    Args:
        a: First axis sequence.
        b: Second axis sequence.

    Returns:
        Matching axes in the order used by ``a``.

    Raises:
        ValueError: If a shared named axis has conflicting sizes.
    """
    named_b = {axis.name: axis for axis in b if axis.name is not None}

    result = []
    for axis in a:
        name = axis.name
        # check here
        if name is None or name not in named_b:
            continue
        other = named_b[name]
        if isinstance(axis, Axis) and isinstance(other, Axis) and axis.size != other.size:
            raise ValueError(f"Axis {name!r} has conflicting sizes in `a` and `b`: {axis.size} vs {other.size}")
        result.append(axis)

    return tuple(result)


def concatenate_axes(a: Sequence[Axis], b: Sequence[Axis]):
    """Concatenate two axis sequences while rejecting named collisions.

    Args:
        a: First axis sequence.
        b: Second axis sequence.

    Returns:
        The axes from ``a`` followed by the axes from ``b``.

    Raises:
        ValueError: If both sequences contain the same named axis.
    """
    a_tuple = tuple(a)
    b_tuple = tuple(b)

    a_names = {axis.name for axis in a_tuple if axis.name is not None}
    b_names = {axis.name for axis in b_tuple if axis.name is not None}
    clashing = a_names & b_names

    if clashing:
        raise ValueError(
            f"Cannot concatenate axis specs sharing names {sorted(clashing)}; "
            "the result would have ambiguous named axes."
        )

    return a_tuple + b_tuple


def remove_axes(axes: Sequence[Axis], to_remove: AxisSelection):
    """Remove selected axes from a sequence.

    Args:
        axes: Axes from which to remove values.
        to_remove: Selector or selectors identifying axes to remove.

    Returns:
        The remaining axes in their original order.

    Raises:
        ValueError: If a selector is ambiguous or cannot be resolved.
    """
    positions = axis_indices(axes, to_remove)

    return tuple(axis for i, axis in enumerate(axes) if i not in positions)


def replace_axes(axes: Sequence[Axis], old: AxisSelection, new: AxisShape):
    """Replace selected axes with a new axis shape.

    Args:
        axes: Axes to update.
        old: Selector or selectors identifying axes to replace.
        new: Replacement axis or axis shape.

    Returns:
        The updated axes in their original positional layout.

    Raises:
        ValueError: If selectors cannot be resolved, replacement names are
            duplicated, or replacement names collide with retained axes.
    """
    axes_list = list(axes)
    positions = axis_indices(axes, old)
    new_axes = axis_shape_to_tuple(new)

    new_names = [a.name for a in new_axes if a.name is not None]
    if len(new_names) != len(set(new_names)):
        raise ValueError(f"Replacement shape contains duplicate axis names: {new_names}")

    pos_set = set(positions)
    other_names = {axis.name for i, axis in enumerate(axes_list) if i not in pos_set and axis.name is not None}
    clashing = set(new_names) & other_names
    if clashing:
        raise ValueError(f"Replacing axis {old} with {new_axes} collides with existing axis names {sorted(clashing)}")

    for idx, axis in zip(positions, new_axes, strict=True):
        axes_list[idx] = axis

    return tuple(axes_list)


def make_axes(**kwargs: int):
    """Construct named axes from keyword arguments."""
    return tuple(Axis(size, name) for name, size in kwargs.items())
