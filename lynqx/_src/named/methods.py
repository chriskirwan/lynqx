import numpy as np
from jax import Device
from jax.lax import PrecisionLike
from jax.typing import DTypeLike

from lynqx._src.axis import AxisSelector
from lynqx._src.named import constructors, indexing, operations, reductions, tensor_contractions, ufuncs, util
from lynqx._src.sharding import device_put
from lynqx._src.typing import AxisLike, AxisSelection, AxisShape, NamedArray, NamedArrayLike, NamedIndex, ShardingLike


def _all(
    self: NamedArray, axis: AxisSelection | None = None, keepdims: bool = False, *, where: NamedArrayLike | None = None
) -> NamedArray:
    return reductions.all(self, axis, keepdims, where=where)


def _any(
    self: NamedArray, axis: AxisSelection | None = None, keepdims: bool = False, *, where: NamedArrayLike | None = None
) -> NamedArray:
    return reductions.any(self, axis, keepdims, where=where)


def _argmax(self: NamedArray, axis: AxisSelection | None = None, keepdims: bool = False) -> NamedArray:
    return reductions.argmax(self, axis, keepdims)


def _argmin(self: NamedArray, axis: AxisSelection | None = None, keepdims: bool = False) -> NamedArray:
    return reductions.argmin(self, axis, keepdims)


def _argpartition(self, kth: int, axis: AxisSelection) -> NamedArray:
    return reductions.argpartition(self, kth, axis)


def _argsort(
    self: NamedArray,
    axis: AxisSelection | None = None,
    *,
    kind: str | None = None,
    order: str | None = None,
    dtype: DTypeLike | None = None,
) -> NamedArray:
    return reductions.argsort(self, axis, kind=kind, order=order, dtype=dtype)


def _astype(self: NamedArray, copy) -> NamedArray:
    raise NotImplementedError()


def _byteswap(self) -> NamedArray:
    """Swap the bytes of the array elements.

    This switches between a little-endian and big-endian data representation.

    Returns:
        An array with the same dtype as ``self``, with underlying bytes of each entry reversed.
    """
    raise NotImplementedError()


def _choose(self: NamedArray) -> NamedArray:
    raise NotImplementedError()


def _clip(self: NamedArray) -> NamedArray:
    raise NotImplementedError()


def _compress(self: NamedArray) -> NamedArray:
    raise NotImplementedError()


def _conj(self) -> NamedArray:
    return operations.conj(self)


def _conjugate(self) -> NamedArray:
    return operations.conjugate(self)


def _contains(self: NamedArray, other: NamedArrayLike) -> NamedArray:
    """Implements __contains__ for JAX arrays.

    This is used by the Python ``in`` operator. This follows `JAX` behaviour.
    """
    if other is None or isinstance(other, str):
        raise TypeError(f"Array.__contains__: unsupported operand type {type(other)}.")

    query = util.ensure_named("Array.__contains__", other)
    if self.ndim != 1:
        raise ValueError(f"Array.__contains__: search array must be one-dimensional, got arr.shape={self.shape}.")
    if query.ndim != 0:
        raise ValueError(f"Array.__contains__: query value must be a scalar, got {query.shape=}")
    return reductions.any(self == query)


def _copy(self) -> NamedArray:
    raise NotImplementedError()


def _cumprod(self: NamedArray, axis: AxisLike, dtype: DTypeLike | None = None) -> NamedArray:
    return reductions.cumprod(self, axis, dtype)


def _cumsum(self: NamedArray, axis: AxisLike, dtype: DTypeLike | None = None) -> NamedArray:
    return reductions.cumsum(self, axis, dtype)


def _diagonal(
    self: NamedArray,
    axis1: AxisSelector = 0,
    axis2: AxisSelector = 1,
    diagonal: AxisLike | None = None,
    offset: int = 0,
) -> NamedArray:
    raise NotImplementedError()


def _dot(
    self: NamedArray,
    b: NamedArray,
    spec: tensor_contractions.DotContraction,
    *,
    precision: PrecisionLike,
    preferred_element_type: DTypeLike | None = None,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    return tensor_contractions.dot(
        self, b, spec, precision=precision, preferred_element_type=preferred_element_type, out_sharding=out_sharding
    )


def _flatten(self: NamedArray, *, out_sharding: ShardingLike | None = None) -> NamedArray:
    raise NotImplementedError()


def _imag_property(self: NamedArray) -> NamedArray:
    """Return the imaginary part of the array"""
    return ufuncs.imag(self)


def _item(self: NamedArray) -> bool | int | float | complex:
    return self.array.item()


def _max(
    self: NamedArray,
    axis: AxisSelection | None = None,
    keepdims: bool = False,
    initial: NamedArrayLike | None = None,
    where: NamedArrayLike | None = None,
) -> NamedArray:
    return reductions.max(self, axis, keepdims, initial, where)


def _mean(
    self: NamedArray,
    axis: AxisSelection | None = None,
    dtype: DTypeLike | None = None,
    keepdims: bool = False,
    where: NamedArrayLike | None = None,
) -> NamedArray:
    return reductions.mean(self, axis, dtype, keepdims, where=where)


def _min(
    self: NamedArray,
    axis: AxisSelection | None = None,
    keepdims: bool = False,
    initial: NamedArrayLike | None = None,
    where: NamedArrayLike | None = None,
) -> NamedArray:
    return reductions.min(self, axis, keepdims, initial, where)


def _nonzero(self: NamedArray, *, size: int | None = None, fill_value: NamedArrayLike | None = None) -> NamedArray:
    raise NotImplementedError()


def _prod(
    self: NamedArray,
    axis: AxisSelection | None = None,
    dtype: DTypeLike | None = None,
    keepdims: bool = False,
    initial: NamedArrayLike | None = None,
    where: NamedArrayLike | None = None,
    promote_integers: bool = True,
) -> NamedArray:
    return reductions.prod(self, axis, dtype, keepdims, initial, where, promote_integers)


def _ptp(self: NamedArray, axis: AxisSelection | None = None, keepdims: bool = False) -> NamedArray:
    return reductions.ptp(self, axis, keepdims)


def _real_property(self: NamedArray) -> NamedArray:
    return ufuncs.real(self)


def _repeat(
    self: NamedArray,
    repeats: NamedArrayLike,
    *,
    total_repeat_length: int | None = None,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    raise NotImplementedError()


def _reshape(self: NamedArray, shape: AxisShape, *, out_sharding: ShardingLike | None = None) -> NamedArray:
    raise NotImplementedError()


def _round(self: NamedArray, decimals: int = 0) -> NamedArray:
    raise NotImplementedError()


def _searchsorted(
    self: NamedArray,
    v: NamedArrayLike,
    side: str = "left",
    sorter: NamedArrayLike | None = None,
    *,
    method: str = "scan",
) -> NamedArray:
    raise NotImplementedError


def _sort(
    self: NamedArray, axis: AxisSelector | None = -1, *, stable: bool = True, descending: bool = False
) -> NamedArray:
    raise NotImplementedError()


def _squeeze(self: NamedArray, axis: AxisSelection | None = None) -> NamedArray:
    raise NotImplementedError()


def _std(
    self: NamedArray,
    axis: AxisSelection,
    dtype: DTypeLike | None = None,
    ddof: int = 0,
    keepdims: bool = False,
    *,
    where: NamedArrayLike | None = None,
    correction: int | float | None = None,
) -> NamedArray:
    return reductions.std(self, axis, dtype, ddof, keepdims, where=where, correction=correction)


def _sum(
    self: NamedArray,
    axis: AxisSelection,
    dtype: DTypeLike | None = None,
    keepdims: bool = False,
    initial: NamedArrayLike | None = None,
    where: NamedArrayLike | None = None,
    promote_integers: bool = False,
) -> NamedArray:
    return reductions.sum(self, axis, dtype, keepdims, initial, where, promote_integers)


def _swapaxes(self: NamedArray, axis1: AxisSelector, axis2: AxisSelector) -> NamedArray:
    raise NotImplementedError()


def _take(
    self: NamedArray,
    axis: AxisSelector,
    index: NamedArrayLike,
    *,
    mode: str | None = None,
    fill_value: indexing.StaticScalar | None = None,
    unique_indices: bool = False,
    indices_are_sorted: bool = False,
):
    return indexing.take(
        self,
        axis,
        index,
        mode=mode,
        fill_value=fill_value,
        unique_indices=unique_indices,
        indices_are_sorted=indices_are_sorted,
    )


def _to_device(self: NamedArray, device: Device | ShardingLike) -> NamedArray:
    return device_put(self, device)


def _to_numpy(self: NamedArray) -> np.ndarray:
    """Convert a NamedArray to a NumPy ndarray.

    Returns:
        A NumPy ndarray with the same data as the NamedArray.
    """
    return np.asarray(self.array)


def _trace(
    self: NamedArray, offset: int = 0, axis1: AxisSelector = 0, axis2: AxisSelector = 1, dtype: DTypeLike | None = None
) -> NamedArray:
    return tensor_contractions.trace(self, offset, axis1, axis2, dtype=dtype)


def _transpose(self: NamedArray, permutation: AxisSelection | None = None) -> NamedArray:
    raise NotImplementedError()


def _transpose_proeprty(self: NamedArray) -> NamedArray:
    raise NotImplementedError()


def _var(
    self: NamedArray,
    axis: AxisSelection | None = None,
    dtype: DTypeLike | None = None,
    ddof: int = 0,
    keepdims: bool = False,
    *,
    where: NamedArrayLike | None = None,
    correction: int | float | None = None,
):
    return reductions.var(self, axis, dtype, ddof, keepdims, where=where, correction=correction)


def _view(self: NamedArray, dtype: DTypeLike | None = None):
    return constructors.array(self.array.view(dtype), self.axes)


def _operator_eq(self, other) -> NamedArray:
    return ufuncs.equal(self, other)


def _operator_ne(self, other) -> NamedArray:
    return ufuncs.not_equal(self, other)


def _operator_lt(self, other) -> NamedArray:
    return ufuncs.less(self, other)


def _operator_le(self, other) -> NamedArray:
    return ufuncs.less_equal(self, other)


def _operator_gt(self, other) -> NamedArray:
    return ufuncs.greater(self, other)


def _operator_ge(self, other) -> NamedArray:
    return ufuncs.greater_equal(self, other)


def _operator_add(self, other) -> NamedArray:
    return ufuncs.add(self, other)


def _operator_radd(self, other) -> NamedArray:
    return ufuncs.add(other, self)


def _operator_sub(self, other) -> NamedArray:
    return ufuncs.subtract(self, other)


def _operator_rsub(self, other) -> NamedArray:
    return ufuncs.subtract(other, self)


def _operator_mul(self, other) -> NamedArray:
    return ufuncs.multiply(self, other)


def _operator_rmul(self, other) -> NamedArray:
    return ufuncs.multiply(other, self)


def _operator_truediv(self, other) -> NamedArray:
    return ufuncs.true_divide(self, other)


def _operator_rtruediv(self, other) -> NamedArray:
    return ufuncs.true_divide(other, self)


def _operator_floordiv(self, other) -> NamedArray:
    return ufuncs.floor_divide(self, other)


def _operator_rfloordiv(self, other) -> NamedArray:
    return ufuncs.floor_divide(other, self)


def _operator_divmod(self, other) -> tuple[NamedArray, NamedArray]:
    return ufuncs.divmod(self, other)


def _operator_rdivmod(self, other) -> tuple[NamedArray, NamedArray]:
    return ufuncs.divmod(other, self)


def _operator_mod(self, other) -> NamedArray:
    return ufuncs.mod(self, other)


def _operator_rmod(self, other) -> NamedArray:
    return ufuncs.mod(other, self)


def _operator_pow(self, other) -> NamedArray:
    return ufuncs.pow(self, other)


def _operator_rpow(self, other) -> NamedArray:
    return ufuncs.pow(other, self)


def _operator_and(self, other) -> NamedArray:
    return ufuncs.bitwise_and(self, other)


def _operator_rand(self, other) -> NamedArray:
    return ufuncs.bitwise_and(other, self)


def _operator_or(self, other) -> NamedArray:
    return ufuncs.bitwise_or(self, other)


def _operator_ror(self, other) -> NamedArray:
    return ufuncs.bitwise_or(other, self)


def _operator_xor(self, other) -> NamedArray:
    return ufuncs.bitwise_xor(self, other)


def _operator_rxor(self, other) -> NamedArray:
    return ufuncs.bitwise_xor(other, self)


def _operator_lshift(self, other) -> NamedArray:
    return ufuncs.left_shift(self, other)


def _operator_rshift(self, other) -> NamedArray:
    return ufuncs.right_shift(self, other)


def _operator_rlshift(self, other) -> NamedArray:
    return ufuncs.left_shift(other, self)


def _operator_rrshift(self, other) -> NamedArray:
    return ufuncs.right_shift(other, self)


def _getitem(self: NamedArray, index: NamedIndex) -> NamedArray:
    positional, output_axes = indexing.resolve_index(self.axes, index, fn_name="__getitem__")
    return constructors.array(self.array[positional], output_axes)


def _unimplemented_setitem(self, i, x) -> NamedArray:
    msg = (
        "JAX arrays are immutable and do not support in-place item assignment."
        " Instead of x[idx] = y, use x = x.at[idx].set(y) or another .at[] method:"
        " https://docs.jax.dev/en/latest/_autosummary/jax.numpy.ndarray.at.html"
    )
    raise TypeError(msg.format(type(self)))


def _unimplemented_matmul(self, b) -> NamedArray:
    msg = (
        "`NamedArray` objects do not support inline matrix multiplication as this axis-wise information cannot be "
        "implicitly inferred."
        " Instead of x @ y, use `lynqx.named.linalg.matmul` or other contraction functions."
    )
    raise TypeError(msg.format(type(self)))


_jaxarray_operators = {
    "getitem": _getitem,
    "setitem": _unimplemented_setitem,
    "matmul": _unimplemented_matmul,
    "rmatmul": _unimplemented_matmul,
    "neg": ufuncs.negative,
    "pos": ufuncs.positive,
    "abs": ufuncs.abs,
    "invert": ufuncs.invert,
    "eq": _operator_eq,
    "ne": _operator_ne,
    "lt": _operator_lt,
    "le": _operator_le,
    "gt": _operator_gt,
    "ge": _operator_ge,
    "add": _operator_add,
    "radd": _operator_radd,
    "sub": _operator_sub,
    "rsub": _operator_rsub,
    "mul": _operator_mul,
    "rmul": _operator_rmul,
    "truediv": _operator_truediv,
    "rtruediv": _operator_rtruediv,
    "floordiv": _operator_floordiv,
    "rfloordiv": _operator_rfloordiv,
    "divmod": _operator_divmod,
    "rdivmod": _operator_rdivmod,
    "mod": _operator_mod,
    "rmod": _operator_rmod,
    "pow": _operator_pow,
    "rpow": _operator_rpow,
    "and": _operator_and,
    "rand": _operator_rand,
    "or": _operator_or,
    "ror": _operator_ror,
    "xor": _operator_xor,
    "rxor": _operator_rxor,
    "lshift": _operator_lshift,
    "rshift": _operator_rshift,
    "rlshift": _operator_rlshift,
    "rrshift": _operator_rrshift,
}

_jaxarray_methods = {
    "all": _all,
    "any": _any,
    "argmax": _argmax,
    "argmin": _argmin,
    "argpartition": _argpartition,
    "argsort": _argsort,
    "astype": _astype,
    "byteswap": _byteswap,
    "choose": _choose,
    "clip": _clip,
    "compress": _compress,
    "conj": _conj,
    "conjugate": _conjugate,
    "copy": _copy,
    "cumprod": _cumprod,
    "cumsum": _cumsum,
    "diagonal": _diagonal,
    "dot": _dot,
    "flatten": _flatten,
    "item": _item,
    "max": _max,
    "mean": _mean,
    "min": _min,
    "nonzero": _nonzero,
    "prod": _prod,
    "ptp": _ptp,
    "ravel": _flatten,
    "repeat": _repeat,
    "reshape": _reshape,
    "round": _round,
    "searchsorted": _searchsorted,
    "sort": _sort,
    "squeeze": _squeeze,
    "std": _std,
    "sum": _sum,
    "swapaxes": _swapaxes,
    "take": _take,
    "to_device": _to_device,
    "trace": _trace,
    "transpose": _transpose,
    "var": _var,
    "view": _view,
}
_jaxarray_properties = {
    "at": indexing.NamedIndexUpdateHelper,
    "imag": _imag_property,
    "real": _real_property,
}
_named_methods = {
    "to_numpy": _to_numpy,
}
_named_properties = {}


def _set_namedarray_attributes(array_impl, include=None, exclude=None):
    # Forward operators, methods and properties on NamedArrays to `lynqx`
    # functions (with no Tracers involved; this forwarding is direct)

    def maybe_setattr(attr_name, target):
        if exclude is not None and attr_name in exclude:
            return
        if not include or attr_name in include:
            setattr(array_impl, attr_name, target)

    for operator_name, function in _jaxarray_operators.items():
        maybe_setattr(f"__{operator_name}__", function)
    for method_name, method in _jaxarray_methods.items():
        maybe_setattr(method_name, method)
    for property_name, prop in _jaxarray_properties.items():
        maybe_setattr(property_name, property(prop))
    for method_name, method in _named_methods.items():
        maybe_setattr(method_name, method)


def register_namedarray_methods():
    _set_namedarray_attributes(NamedArray)
