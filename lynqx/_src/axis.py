import keyword
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace


_AXIS_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_axis_name(name) -> str:
    if not isinstance(name, str):
        raise TypeError(f"Axis name must be a string, got {type(name).__name__}")

    if not _AXIS_NAME.fullmatch(name):
        raise ValueError(f"Invalid axis name: {name!r}")

    if keyword.iskeyword(name) or keyword.issoftkeyword(name):
        raise ValueError(f"Axis name is a Python keyword: {name!r}")

    return name


@dataclass(frozen=True, slots=True)
class Axis:
    size: int
    name: str | None = None

    def __post_init__(self):
        if isinstance(self.size, bool) or not isinstance(self.size, int):
            raise TypeError(f"Axis size must be an integer, got {type(self.size).__name__}")
        if self.size <= 0:
            raise ValueError(f"Axis size must be positive, got {self.size}")

        if self.name is not None:
            validate_axis_name(self.name)

    def rename(self, name: str):
        return replace(self, name=name)

    def resize(self, size: int):
        return replace(self, size=size)

    @property
    def label(self) -> str:
        return self.name if self.name is not None else "?"


AxisLike = str | Axis

AxisDict = Mapping[str, int]
AxisDim = int | Axis
AxisShape = AxisDim | Sequence[AxisDim] | AxisDict

AxisSpec = AxisLike | Sequence[AxisLike]

AxisSelector = int | AxisLike
AxisSelection = AxisSelector | Sequence[int | AxisLike]
