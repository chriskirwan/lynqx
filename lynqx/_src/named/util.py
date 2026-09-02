from collections.abc import Collection
from typing import Any, overload, TypeVar

import jax.numpy as jnp
from jax.typing import ArrayLike

from lynqx._src.array import NamedArrayImpl
from lynqx._src.filters import is_named_array
from lynqx._src.typing import NamedArray


NamedArrayLike = NamedArray | ArrayLike

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
