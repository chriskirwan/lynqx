from collections.abc import Callable
from functools import partial
from typing import Any

import jax.sharding
import jax.tree
from jax import Device, NamedSharding, P
from jax.api_util import flatten_axes
from jaxtyping import PyTree

from lynqx._src.filters import assert_single_array_leaf, is_named_array
from lynqx._src.typing import AxisSpec, NamedArray, PM, ShardingLike


def canonicalize_sharding(axes: AxisSpec, sharding: ShardingLike | None, fn_name: str) -> NamedSharding | P | None:
    """Convert a Lynqx sharding specification to a JAX sharding specification.

    ``PM`` specifications are resolved against the current JAX mesh and the
    supplied logical axes.
    """
    if sharding is None:
        return None

    if isinstance(sharding, (NamedSharding, P)):
        return sharding

    if not isinstance(sharding, ShardingLike):
        raise TypeError(
            f"`out_sharding` argument of {fn_name} only supports instances of "
            f"`NamedSharding`, `PartitionSpec`, or `PM`. Got {sharding} of type: {type(sharding)}"
        )

    mesh = jax.sharding.get_abstract_mesh()
    if isinstance(sharding, PM) and mesh.empty:
        raise ValueError(
            "Using `PartitionSpec` when you are not inside a mesh context is not allowed. Please pass a "
            "`NamedSharding` instance or enter into a mesh context via `jax.set_mesh`"
        )

    partition_spec = sharding.partition_spec(axes)
    return partition_spec


def auto_mesh_axes(fn: Callable[..., Any], axes: AxisSpec, out_sharding: ShardingLike | None, **hoist_kwargs):
    """Switches some or all mesh axis types to `Auto`, which allows adding an `out_sharding` argument to any
    composition of operations.

    Mesh axes can be either `Auto` or `Explicit`, but not `Manual`.

    Keyword arguments in ``hoist_kwargs`` are bound to ``fn`` before applying
    JAX's automatic mesh transformation.
    """
    f = partial(fn, **hoist_kwargs)

    if out_sharding is None:
        return f

    jax_sharding = canonicalize_sharding(axes, out_sharding, "auto_mesh")
    return jax.sharding.auto_axes(f, out_sharding=jax_sharding)


def _canonicalize_named_sharding(
    fn_name: str, x: PyTree, sharding: Device | ShardingLike | None
) -> Device | NamedSharding | P | None:
    """Canonicalize a per-leaf sharding, resolving PM for NamedArray leaves."""
    if isinstance(x, NamedArray):
        return canonicalize_sharding(x.axes, sharding, fn_name)
    if isinstance(sharding, PM):
        raise TypeError(
            f"{fn_name}: `PM` PartitionSpec supplied for a non-NamedArray leaf "
            f"(type {type(x).__name__}); use `jax.P` or `jax.NamedSharding` instead."
        )
    return sharding


def _flatten_pytree_shardings(
    fn_name: str,
    xs: PyTree,
    shardings: PyTree,
) -> tuple[list[PyTree], list[Device | NamedSharding | P | None], PyTree]:
    flat_xs, treedef = jax.tree.flatten(xs, is_leaf=is_named_array)
    for x in flat_xs:
        assert_single_array_leaf(x, fn_name)
    flat_shardings = flatten_axes(fn_name, treedef, shardings)
    canonical_shardings = [
        _canonicalize_named_sharding(fn_name, x, shard) for x, shard in zip(flat_xs, flat_shardings)
    ]
    return flat_xs, canonical_shardings, treedef


def _map_canonicalized(fn_name: str, xs: PyTree, shardings: PyTree, op: Callable[[Any, Any], Any]) -> PyTree:
    flat_xs, canonical, treedef = _flatten_pytree_shardings(fn_name, xs, shardings)
    flat_out = [op(xi, c) for xi, c in zip(flat_xs, canonical)]
    return jax.tree.unflatten(treedef, flat_out)


def reshard(xs: PyTree, out_shardings: ShardingLike) -> PyTree:
    """Reshard the array leaves of a pytree.

    NamedArray leaves use their logical axes to resolve ``PM`` specifications.

    Args:
        xs: Pytree whose array leaves should be resharded.
        out_shardings: Sharding specification for the output leaves. A
            ``PM`` is resolved using each NamedArray's logical axes.

    Returns:
        A pytree with the same structure as ``xs`` and the requested sharding.
    """
    return _map_canonicalized("reshard", xs, out_shardings, jax.reshard)


def device_put(x, device=None, *, src=None, may_alias=False, donate=False):
    """Copy array leaves of a pytree to a device or sharding.

    This follows :func:`jax.device_put` for individual leaves while preserving
    NamedArray metadata and supporting ``PM`` specifications for NamedArrays.

    Args:
        x: Value or pytree to place on a device.
        device: Device or sharding specification for the output leaves. If
            omitted, JAX chooses the placement for each leaf.
        src: Optional source device or sharding, as in ``jax.device_put``.
        may_alias: Whether the input and output buffers may alias.
        donate: Whether to donate the input buffers.

    Returns:
        ``x`` placed on the requested device or sharding.
    """
    op = partial(jax.device_put, src=src, may_alias=may_alias, donate=donate)
    if device is None:
        return jax.tree.map(op, x, is_leaf=is_named_array)
    return _map_canonicalized("device_put", x, device, op)


def with_sharding_constraint(x: PyTree, shardings: PyTree) -> PyTree:
    """Constrain the sharding of each array leaf in a pytree.

    This follows :func:`jax.lax.with_sharding_constraint` for individual
    leaves. ``PM`` specifications are resolved using the logical axes of
    NamedArray leaves.

    Args:
        x: Pytree whose array leaves should be constrained.
        shardings: Pytree of sharding specifications with a structure matching
            ``x``. A ``PM`` is resolved using each NamedArray's logical axes.

    Returns:
        A pytree with the same structure as ``x`` and constrained sharding.
    """
    return _map_canonicalized("with_sharding_constraint", x, shardings, jax.lax.with_sharding_constraint)
