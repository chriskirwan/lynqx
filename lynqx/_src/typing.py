from collections.abc import Mapping
from types import EllipsisType

from lynqx._src.axis import (
    Axis as Axis,
    AxisDict as AxisDict,
    AxisLike as AxisLike,
    AxisSelection as AxisSelection,
    AxisSelector as AxisSelector,
    AxisShape as AxisShape,
    AxisSpec as AxisSpec,
)
from lynqx._src.basearray import NamedArray as NamedArray, NamedArrayLike as NamedArrayLike
from lynqx._src.partition import PM as PM, ShardingLike as ShardingLike


AxisIndex = int | slice | NamedArray
NamedIndex = AxisIndex | EllipsisType | tuple[AxisIndex | EllipsisType, ...] | Mapping[AxisSelector, AxisIndex]
