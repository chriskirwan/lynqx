from collections.abc import Mapping

from jax import NamedSharding, P

from lynqx._src.axis import Axis, AxisLike, AxisSpec
from lynqx._src.axis_util import axis_spec_to_tuple


LogicalAxis = str | Axis
PhysicalAxis = str | None


class PM:
    __slots__ = ("_mapping", "_partition")

    def __init__(
        self, mapping: Mapping[LogicalAxis, PhysicalAxis] | None = None, partition: tuple[PhysicalAxis, ...] = ()
    ):
        self._mapping = dict(mapping or {})
        self._partition = tuple(partition)

    def get_physical_axis(self, axis: AxisLike) -> PhysicalAxis:
        if axis in self._mapping:
            return self._mapping[axis]

        if isinstance(axis, Axis) and axis.name is not None:
            return self._mapping.get(axis.name)

        return None

    def get_logical_axes(self) -> tuple[LogicalAxis, ...]:
        return tuple(self._mapping)

    def is_mapped(self, axis: AxisLike) -> bool:
        if axis in self._mapping:
            return True
        return isinstance(axis, Axis) and axis.name is not None and axis.name in self._mapping

    def with_mappings(self, **update: PhysicalAxis) -> PM:
        return PM({**self._mapping, **update}, self._partition)

    def with_partitions(self, *partition: PhysicalAxis) -> "PM":
        return PM(self._mapping, partition)

    @property
    def partition(self) -> tuple[PhysicalAxis, ...]:
        """Physical placements for anonymous axes, consumed positionally in partition_spec."""
        return self._partition

    def partition_spec(self, axes: AxisSpec | None = None) -> P:
        if axes is None:
            return P()

        axes = axis_spec_to_tuple(axes)
        anonymous = iter(self._partition)

        result = []
        for axis in axes:
            physical = self.get_physical_axis(axis)
            if physical is None:
                # Use fallback for anonymous axes
                physical = next(anonymous, None)

            result.append(physical)

        return P(*result)

    def __hash__(self) -> int:
        return hash((frozenset(self._mapping.items()), self._partition))

    def __eq__(self, other) -> bool:
        return isinstance(other, PM) and self._mapping == other._mapping and self._partition == other._partition


ShardingLike = NamedSharding | P | PM
