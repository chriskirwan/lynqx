from collections.abc import Collection, Sequence
from typing import Any, overload, TypeVar

import jax.numpy as jnp

from lynqx._src.array import NamedArrayImpl
from lynqx._src.axis_util import match_axes
from lynqx._src.filters import is_named_array, is_scalar
from lynqx._src.typing import Axis, AxisLike, NamedArray, NamedArrayLike


T = TypeVar("T")


@overload
def ensure_named(fn_name: str, /) -> tuple[()]: ...


@overload
def ensure_named(fn_name: str, a1: Any, /) -> NamedArray: ...


@overload
def ensure_named(fn_name: str, /, *args: Any) -> tuple[NamedArray, ...]: ...


def ensure_named(fn_name: str, /, *args: Any) -> NamedArray | tuple[NamedArray, ...]:
    """Check that the arguments are `NamedArrayLike` and are convertible to `NamedArrays`"""
    check_namedlike(fn_name, *args)
    if len(args) == 1:
        return _namedarraylike_to_namedarray(args[0])
    return tuple(_namedarraylike_to_namedarray(arg) for arg in args)


def ensure_named_tuple(fn_name: str, tup: Collection[Any]) -> tuple[NamedArray, ...]:
    check_namedlike(fn_name, *tup)
    return tuple(_namedarraylike_to_namedarray(arg) for arg in tup)


def check_namedlike(fn_name: str, *args: Any):
    assert isinstance(fn_name, str), f"fn_name must be a string. Got {type(fn_name)}"
    if any(not _is_namedarraylike(arg) for arg in args):
        pos, arg = next((i, arg) for i, arg in enumerate(args) if not _is_namedarraylike(arg))
        raise TypeError(f"{fn_name} requires `NamedArray` or scalar arguments, got {type(arg)} at position {pos}")


def _namedarraylike_to_namedarray(x: NamedArrayLike) -> NamedArray:
    if isinstance(x, NamedArray):
        return x
    x = jnp.asarray(x)
    return NamedArrayImpl(x, ())


def _is_namedarraylike(x):
    """`NamedArray`, or JAX scalar objects (zero-dimensional arrays are scalar)"""
    return is_named_array(x) or jnp.isscalar(x)


def align_shapes_for_broadcast(
    lhs: Sequence[Axis], rhs: Sequence[Axis]
) -> tuple[tuple[Axis, ...], tuple[Axis, ...], tuple[Axis, ...]]:
    """Aligns LHS and RHS axis tuples for broadcasting operations (e.g. elementwise binary ops).

    Returns:
        (aligned_lhs_axes, aligned_rhs_axes, broadcast_out_axes)
    """
    matched = match_axes(lhs, rhs, allow_positional_fallback=True)

    # Validate named axis sizes match
    for m in matched.matches:
        l_ax, r_ax = lhs[m.source], rhs[m.target]
        if l_ax.size != r_ax.size and l_ax.size != 1 and r_ax.size != 1:
            raise ValueError(f"Incompatible sizes for axis {l_ax.label!r}: {l_ax.size} vs {r_ax.size}")

    matched_by_source = {m.source: m for m in matched.matches}
    out_axes: list[Axis] = []

    for i, l_ax in enumerate(lhs):
        if i in matched_by_source:
            r_ax = rhs[matched_by_source[i].target]
            out_axes.append(Axis(max(l_ax.size, r_ax.size), l_ax.name))
        else:
            out_axes.append(l_ax)

    for j in matched.unmatched_target:
        out_axes.append(rhs[j])

    out_tuple = tuple(out_axes)
    return _align_source_and_target(lhs, out_tuple), _align_source_and_target(rhs, out_tuple), out_tuple


def _align_source_and_target(source: Sequence[Axis], target: Sequence[Axis]) -> tuple[Axis, ...]:
    """Expands `source` with unit dummy axes (size 1) to match `target` shape."""
    res = match_axes(source, target, allow_positional_fallback=True)
    tgt_to_src = {m.target: m.source for m in res.matches}

    result: list[Axis] = []
    for tgt_i, tgt_ax in enumerate(target):
        if tgt_i in tgt_to_src:
            result.append(source[tgt_to_src[tgt_i]])
        else:
            result.append(Axis(name=tgt_ax.name, size=1))
    return tuple(result)


def scalar_namedarray_to_jax_scalar(element: Any):
    if not is_scalar(element):
        raise ValueError("`element` must be a scalar NamedArray")

    if isinstance(element, NamedArray):
        return element.array
    return element


def check_scalarlike(fn_name: str, *args: Any):
    assert isinstance(fn_name, str), f"fn_name must be a string. Got {type(fn_name)}"
    if any(not is_scalar(arg) for arg in args):
        pos, arg = next((i, arg) for i, arg in enumerate(args) if not is_scalar(arg))
        if isinstance(arg, NamedArray):
            raise TypeError(f"{fn_name} requires `ScalarLike` arguments, got {arg} at position {pos}")
        raise TypeError(
            f"{fn_name} requires scalar (or scalar NamedArray) arguments, got {type(arg)} at position {pos}"
        )


@overload
def ensure_scalar(fn_name: str, /) -> tuple[()]: ...


@overload
def ensure_scalar(fn_name: str, a1: Any, /) -> NamedArray: ...


@overload
def ensure_scalar(fn_name: str, /, *args: Any) -> tuple[NamedArray, ...]: ...


def ensure_scalar(fn_name: str, /, *args: Any) -> NamedArray | tuple[NamedArray, ...]:
    check_scalarlike(fn_name, *args)
    if len(args) == 1:
        return _namedarraylike_to_namedarray(args[0])
    return tuple(_namedarraylike_to_namedarray(arg) for arg in args)


def resolve_new_axis(spec: AxisLike | None, size: int, fn_name: str, param: str) -> Axis:
    """Build an output axis of a routine that has no counterpart in the input.

    This is the shared axis-inference rule for the linalg submodule:

    * ``None``  -> anonymous ``Axis(size)``
    * ``str``   -> ``Axis(size, name)``
    * ``Axis``  -> used as given; its size must equal ``size``
    """
    if spec is None:
        return Axis(size)
    if isinstance(spec, str):
        return Axis(size, spec)
    if isinstance(spec, Axis):
        if spec.size != size:
            raise ValueError(f"{fn_name}: `{param}` has size {spec.size} but the result axis has size {size}")
        return spec
    raise TypeError(
        f"{fn_name}: `{param}` must be None, a str or an Axis, got {type(spec).__name__}. "
        "Integer selectors cannot name a new axis."
    )
