# from collections.abc import Mapping
# from typing import Any

import numpy as np
from jax import Array
from jax.sharding import Sharding

# from jax.typing import DTypeLike
from lynqx._src.axis import Axis

# from lynqx._src.named.methods import _NamedIndexUpdateHelper
# from lynqx._src.typing import AxisOrName, AxisSelector, NamedArrayLike

class NamedArray:
    array: Array
    axes: tuple[Axis, ...]

    @property
    def dtype(self) -> np.dtype: ...
    @property
    def nbytes(self) -> int: ...
    @property
    def ndim(self) -> int: ...
    @property
    def size(self) -> int: ...
    @property
    def sharding(self) -> Sharding: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
