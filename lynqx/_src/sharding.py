from collections.abc import Callable
from typing import TypeVar

import jax.sharding
import jax.tree
from jax import NamedSharding, P

from lynqx._src.typing import AxisSpec, PM, ShardingLike
from lynqx._src.util import cache


T = TypeVar("T")
F = TypeVar("F", bound=Callable)
G = TypeVar("G", bound=Callable)

MeshLike = jax.sharding.Mesh | jax.sharding.AbstractMesh


@cache()
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
