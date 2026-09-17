from collections.abc import Mapping, Sequence
from types import EllipsisType
from typing import Any, Protocol

import numpy as np

from lynqx._src.axis import (
    Axis as Axis,
    AxisDict as AxisDict,
    AxisDim,
    AxisLike as AxisLike,
    AxisSelection as AxisSelection,
    AxisSelector as AxisSelector,
    AxisShape as AxisShape,
    AxisSpec as AxisSpec,
)
from lynqx._src.basearray import NamedArray as NamedArray, NamedArrayLike as NamedArrayLike
from lynqx._src.partition import PM as PM, ShardingLike as ShardingLike


DType = np.dtype
Shape = Sequence[AxisDim | Any]


class DuckTypedArray(Protocol):
    @property
    def dtype(self) -> np.dtype: ...

    @property
    def shape(self) -> Sequence[int | Any]: ...


AxisIndex = int | slice | NamedArray
NamedIndex = AxisIndex | EllipsisType | tuple[AxisIndex | EllipsisType, ...] | Mapping[AxisSelector, AxisIndex]
