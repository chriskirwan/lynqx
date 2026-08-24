from collections.abc import Mapping

from jax import P

from lynqx._src.axis import Axis
from lynqx._src.axis_util import axis_spec_to_tuple
from lynqx._src.typing import AxisLike, AxisSpec


LogicalAxis = str | Axis
PhysicalAxis = str | None


class PM:
    """A class describing how to partition a NamedArray across a mesh of devices, as an analogue to
    `jax.P`
    """

    __slots__ = ("_mapping", "_partitions")

    def __init__(
        self,
        mapping: Mapping[LogicalAxis, PhysicalAxis] = {},
        partition: tuple[PhysicalAxis, ...] = (),
    ):
        self._mapping = dict(mapping or {})
        self._partitions = tuple(partition)

    def partition_spec(self, axes: AxisSpec | None = None) -> P:
        if axes is None:
            return P()

        axes = axis_spec_to_tuple(axes)
        anonymous = iter(self._partitions)

        result = []
        for axis in axes:
            if isinstance(axis, Axis) and axis.name is not None:
                result.append(self._mapping.get(axis.name))
            elif isinstance(axis, str):
                result.append(self._mapping.get(axis))
            else:
                result.append(next(anonymous, None))

        return P(*result)

    def _lookup(self, axis: AxisLike) -> str | None:
        if axis in self._mapping:
            return self._mapping[axis]

        if isinstance(axis, Axis) and axis.name is not None:
            return self._mapping.get(axis.name)

        return None

    def __or__(self, other: PM) -> PM:
        return union_partitions(self, other)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PM) and self._mapping == other._mapping

    def __hash__(self) -> int:
        return hash(frozenset(self._mapping.items()))


def union_partitions(left: PM, right: PM) -> PM:
    """Union two PartitionMappings.

    Conflicting non-None assignments raise ValueError.
    A non-None assignment takes precedence over None.
    """
    merged = dict(left._mapping)

    for axis, partition in right._mapping.items():
        if axis in merged:
            existing = merged[axis]

            if existing is not None and partition is not None and existing != partition:
                raise ValueError(f"Conflict in PM union for axis {axis!r}: {existing!r} vs {partition!r}")

            if partition is not None:
                merged[axis] = partition
        else:
            merged[axis] = partition

    return PM(merged)
