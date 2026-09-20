from collections.abc import Mapping, Sequence
from types import EllipsisType
from typing import Any, Protocol, TYPE_CHECKING

import numpy as np
from jax.typing import ArrayLike
from jaxtyping import ScalarLike

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
from lynqx._src.custom_types import _make_dtype_specifier
from lynqx._src.partition import PM as PM, ShardingLike as ShardingLike


if TYPE_CHECKING:
    from jaxtyping import (
        BFloat16 as BFloat16,
        Bool as Bool,
        Complex as Complex,
        Complex64 as Complex64,
        Complex128 as Complex128,
        Float as Float,
        Float8e4m3b11fnuz as Float8e4m3b11fnuz,
        Float8e4m3fn as Float8e4m3fn,
        Float8e4m3fnuz as Float8e4m3fnuz,
        Float8e5m2 as Float8e5m2,
        Float8e5m2fnuz as Float8e5m2fnuz,
        Float16 as Float16,
        Float32 as Float32,
        Float64 as Float64,
        Inexact as Inexact,
        Int as Int,
        Int2 as Int2,
        Int4 as Int4,
        Int8 as Int8,
        Int16 as Int16,
        Int32 as Int32,
        Int64 as Int64,
        Integer as Integer,
        Key as Key,
        Num as Num,
        Real as Real,
        Scalar as Scalar,
        ScalarLike as ScalarLike,
        Shaped as Shaped,
        UInt as UInt,
        UInt2 as UInt2,
        UInt4 as UInt4,
        UInt8 as UInt8,
        UInt16 as UInt16,
        UInt32 as UInt32,
        UInt64 as UInt64,
    )
else:
    import jaxtyping as jt

    Bool = _make_dtype_specifier(jt.Bool)

    Int = _make_dtype_specifier(jt.Int)
    Integer = _make_dtype_specifier(jt.Integer)
    Int2 = _make_dtype_specifier(jt.Int2)
    Int4 = _make_dtype_specifier(jt.Int4)
    Int8 = _make_dtype_specifier(jt.Int8)
    Int16 = _make_dtype_specifier(jt.Int16)
    Int32 = _make_dtype_specifier(jt.Int32)
    Int64 = _make_dtype_specifier(jt.Int64)
    UInt = _make_dtype_specifier(jt.UInt)
    UInt2 = _make_dtype_specifier(jt.UInt2)
    UInt4 = _make_dtype_specifier(jt.UInt4)
    UInt8 = _make_dtype_specifier(jt.UInt8)
    UInt16 = _make_dtype_specifier(jt.UInt16)
    UInt32 = _make_dtype_specifier(jt.UInt32)
    UInt64 = _make_dtype_specifier(jt.UInt64)

    BFLoat16 = _make_dtype_specifier(jt.BFloat16)
    Float = _make_dtype_specifier(jt.Float)
    Float8e4m3b11fnuz = _make_dtype_specifier(jt.Float8e4m3b11fnuz)
    Float8e4m3fnuz = _make_dtype_specifier(jt.Float8e4m3fnuz)
    Float8e4m3fn = _make_dtype_specifier(jt.Float8e4m3fn)
    Float8e5m2 = _make_dtype_specifier(jt.Float8e5m2)
    Float8e5m2fnuz = _make_dtype_specifier(jt.Float8e5m2fnuz)

    Complex = _make_dtype_specifier(jt.Complex)
    Complex64 = _make_dtype_specifier(jt.Complex64)
    Complex128 = _make_dtype_specifier(jt.Complex128)

    Inexact = _make_dtype_specifier(jt.Inexact)
    Real = _make_dtype_specifier(jt.Real)
    Num = _make_dtype_specifier(jt.Num)
    Shaped = _make_dtype_specifier(jt.Num)

    Key = _make_dtype_specifier(jt.Key)

DType = np.dtype
Shape = Sequence[AxisDim | Any]


class DuckTypedNamedArray(Protocol):
    @property
    def dtype(self) -> np.dtype: ...

    @property
    def shape(self) -> Sequence[Axis | Any]: ...


AxisIndex = int | slice | NamedArray
NamedIndex = AxisIndex | EllipsisType | tuple[AxisIndex | EllipsisType, ...] | Mapping[AxisSelector, AxisIndex]


NamedScalar = Shaped[NamedArray, ""]
NamedScalarLike = NamedScalar | ScalarLike
RealScalarLike = Real[NamedArray, ""] | Real[ArrayLike, ""]
IntScalarLike = Int[NamedArray, ""] | Int[ArrayLike, ""]
