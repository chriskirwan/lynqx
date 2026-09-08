from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.tree

from lynqx._src.typing import NamedArray


def is_jax_array(element: Any) -> bool:
    """Returns `True` if `element` is a jax `Array` or numpy `NDArray`."""
    return eqx.is_array(element)


def is_jax_array_like(element: Any) -> bool:
    """Returns True if `element` is a jax `Array`, numpy `NDArray` or a python `float`/`complex`/`bool`/`int`."""
    return eqx.is_array_like(element)


def is_inexact_jax_array(element: Any) -> bool:
    """Returns True if `element` is a jax `Array`, numpy `NDArray` or a python `float`/`complex`/`bool`/`int`."""
    return eqx.is_inexact_array(element)


def is_inexact_jax_array_like(element: Any) -> bool:
    """
    Returns `True` if `element` is an inexact jax `Array`, numpy `NDArray`, or python `float`
    or `complex`.
    """
    return eqx.is_inexact_array_like(element)


def is_named_array(element: Any) -> bool:
    """Returns `True` if `element` is a `NamedArray`"""
    return isinstance(element, NamedArray)


def is_inexact_named_array(element: Any) -> bool:
    """Returns `True` if `element` is an inexact `NamedArray`"""
    return isinstance(element, NamedArray) and eqx.is_inexact_array(element.array)


def is_arrayish(element: Any) -> bool:
    """Returns `True` if `element` is a `NamedArray`, jax `Array` or numpy `NDArray`"""
    return isinstance(element, NamedArray) or eqx.is_array(element)


def is_inexact_arrayish(element: Any) -> bool:
    """Returns `True` if `element` is an inexact `NamedArray`, jax `Array` or numpy `NDArray`"""
    if isinstance(element, NamedArray):
        element = element.array
    return eqx.is_inexact_array(element)


def is_scalar(element: Any) -> bool:
    if isinstance(element, NamedArray):
        return element.ndim == 0
    return jnp.isscalar(element)


def assert_single_array_leaf(x: Any, context: str) -> None:
    """Assert that a NamedArray contributes exactly one leaf to a pytree flatten operation."""
    if is_named_array(x):
        n_leaves = len(jax.tree.leaves(x))
        if n_leaves != 1:
            raise AssertionError(
                f"{context}: expected NamedArray to have exactly 1 array leaf, "
                f"got {n_leaves}. NamedArray's pytree structure may have changed "
                f"— code assuming a 1:1 NamedArray-to-leaf mapping needs review."
            )
