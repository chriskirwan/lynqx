from collections.abc import Callable
from functools import partial
from typing import Any, TypeVar

import jax.sharding
import jax.tree
from jax import NamedSharding, P

from lynqx._src.typing import AxisSpec, PM, ShardingLike


T = TypeVar("T")
F = TypeVar("F", bound=Callable)
G = TypeVar("G", bound=Callable)

MeshLike = jax.sharding.Mesh | jax.sharding.AbstractMesh


def canonicalize_sharding(
    sharding: ShardingLike | None, axes: AxisSpec | None, fn_name: str
) -> NamedSharding | P | None:
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

    for logical_axis, physical_spec in sharding._mapping.items():
        if physical_spec is None:
            raise ValueError(
                f"Physical mapping for logical axis {logical_axis} is None. "
                "Expected a mesh axis name or a sequence of mesh axis names."
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

    jax_sharding = canonicalize_sharding(out_sharding, axes, "auto_mesh")
    return jax.sharding.auto_axes(f, out_sharding=jax_sharding)
