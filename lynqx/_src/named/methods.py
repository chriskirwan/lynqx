from lynqx._src.named import constructors, indexing, ufuncs
from lynqx._src.typing import NamedArray, NamedIndex


def _operator_eq(self, other):
    return ufuncs.equal(self, other)


def _operator_ne(self, other):
    return ufuncs.not_equal(self, other)


def _operator_lt(self, other):
    return ufuncs.less(self, other)


def _operator_le(self, other):
    return ufuncs.less_equal(self, other)


def _operator_gt(self, other):
    return ufuncs.greater(self, other)


def _operator_ge(self, other):
    return ufuncs.greater_equal(self, other)


def _operator_add(self, other):
    return ufuncs.add(self, other)


def _operator_radd(self, other):
    return ufuncs.add(other, self)


def _operator_sub(self, other):
    return ufuncs.subtract(self, other)


def _operator_rsub(self, other):
    return ufuncs.subtract(other, self)


def _operator_mul(self, other):
    return ufuncs.multiply(self, other)


def _operator_rmul(self, other):
    return ufuncs.multiply(other, self)


def _operator_truediv(self, other):
    return ufuncs.true_divide(self, other)


def _operator_rtruediv(self, other):
    return ufuncs.true_divide(other, self)


def _operator_floordiv(self, other):
    return ufuncs.floor_divide(self, other)


def _operator_rfloordiv(self, other):
    return ufuncs.floor_divide(other, self)


def _operator_mod(self, other):
    return ufuncs.mod(self, other)


def _operator_rmod(self, other):
    return ufuncs.mod(other, self)


def _operator_pow(self, other):
    return ufuncs.pow(self, other)


def _operator_rpow(self, other):
    return ufuncs.pow(other, self)


def _operator_and(self, other):
    return ufuncs.bitwise_and(self, other)


def _operator_rand(self, other):
    return ufuncs.bitwise_and(other, self)


def _operator_or(self, other):
    return ufuncs.bitwise_or(self, other)


def _operator_ror(self, other):
    return ufuncs.bitwise_or(other, self)


def _operator_xor(self, other):
    return ufuncs.bitwise_xor(self, other)


def _operator_rxor(self, other):
    return ufuncs.bitwise_xor(other, self)


def _operator_lshift(self, other):
    return ufuncs.left_shift(self, other)


def _operator_rshift(self, other):
    return ufuncs.right_shift(self, other)


def _operator_rlshift(self, other):
    return ufuncs.left_shift(other, self)


def _operator_rrshift(self, other):
    return ufuncs.right_shift(self, other)


def _getitem(self: NamedArray, index: NamedIndex) -> NamedArray:
    positional, output_axes = indexing.resolve_index(self.axes, index, fn_name="__getitem__")
    return constructors.array(self.array[positional], output_axes)


def _unimplemented_setitem(self, i, x):
    msg = (
        "JAX arrays are immutable and do not support in-place item assignment."
        " Instead of x[idx] = y, use x = x.at[idx].set(y) or another .at[] method:"
        " https://docs.jax.dev/en/latest/_autosummary/jax.numpy.ndarray.at.html"
    )
    raise TypeError(msg.format(type(self)))


def _unimplemented_matmul(self, b):
    msg = (
        "`NamedArray` objects do not support inline matrix multiplication as this axis-wise information cannot be "
        "implicitly inferred."
        " Instead of x @ y, use `lynqx.numops.matmul` or other contraction functions."
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

_jaxarray_methods = {}
_jaxarray_properties = {"at": indexing.NamedIndexUpdateHelper}
_named_methods = {}
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
