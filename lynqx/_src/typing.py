from collections.abc import Mapping, Sequence
from typing import TypeAlias

from lynqx._src.axis import Axis
from lynqx._src.basearray import NamedArray as NamedArray


AxisDict = Mapping[str, int]
AxisLike: TypeAlias = str | Axis

AxisShape: TypeAlias = Axis | Sequence[Axis] | AxisDict
AxisSpec: TypeAlias = AxisLike | Sequence[AxisLike]

AxisSelector: TypeAlias = int | AxisLike
AxisSelection: TypeAlias = AxisSelector | Sequence[int | AxisLike]
