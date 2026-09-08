from collections.abc import Callable
from functools import wraps

import jax.lax as lax
import jax.numpy as jnp
from jax import Array

from lynqx._src.named import constructors, operations, util
from lynqx._src.typing import NamedArray, NamedArrayLike


def wrap_elementwise_unary_op(fn: Callable[..., Array]) -> Callable[..., NamedArray]:
    @wraps(fn)
    def wrapped(x: NamedArrayLike, *args, **kwargs) -> NamedArray:
        x = util.ensure_named("wrap_elementwise_unary", x)
        return constructors.array(fn(x.array, *args, **kwargs), x.axes)

    return wrapped


def wrap_elementwise_binary_op(fn: Callable[..., Array]) -> Callable[..., NamedArray]:
    @wraps(fn)
    def wrapped(x: NamedArrayLike, y: NamedArrayLike, *args, **kwargs):
        x, y = util.ensure_named("wrap_elementwise_binary_op", x, y)
        (a, b), axes = operations._broadcast_arrays(x, y)
        return constructors.array(fn(a.array, b.array, *args, **kwargs), axes)

    return wrapped


def abs(x: NamedArrayLike) -> NamedArray:
    return wrap_elementwise_unary_op(jnp.abs)(x)


def absolute(x: NamedArrayLike) -> NamedArray:
    return wrap_elementwise_unary_op(jnp.absolute)(x)


def acos(x: NamedArrayLike) -> NamedArray:
    return wrap_elementwise_unary_op(jnp.acos)(x)


def acosh(x: NamedArrayLike) -> NamedArray:
    return wrap_elementwise_unary_op(jnp.acosh)(x)


def arccos(x: NamedArrayLike) -> NamedArray:
    return wrap_elementwise_unary_op(jnp.arccos)(x)


def arccosh(x: NamedArrayLike) -> NamedArray:
    return wrap_elementwise_unary_op(jnp.arccosh)(x)


def arcsin(x: NamedArrayLike) -> NamedArray:
    return wrap_elementwise_unary_op(jnp.arcsin)(x)


def arcsinh(x: NamedArrayLike) -> NamedArray:
    return wrap_elementwise_unary_op(jnp.arcsinh)(x)


def arctan(x: NamedArrayLike) -> NamedArray:
    return wrap_elementwise_unary_op(jnp.arctan)(x)


def arctanh(x: NamedArrayLike) -> NamedArray:
    return wrap_elementwise_unary_op(jnp.arctanh)(x)


def asin(x: NamedArrayLike) -> NamedArray:
    return wrap_elementwise_unary_op(jnp.asin)(x)


def asinh(x: NamedArrayLike) -> NamedArray:
    return wrap_elementwise_unary_op(jnp.asinh)(x)


def atan(x: NamedArrayLike) -> NamedArray:
    return wrap_elementwise_unary_op(jnp.atan)(x)


def atanh(x: NamedArrayLike) -> NamedArray:
    return wrap_elementwise_unary_op(jnp.atanh)(x)


def around(x: NamedArrayLike) -> NamedArray:
    return wrap_elementwise_unary_op(jnp.around)(x)


def bitwise_count(x: NamedArrayLike, /) -> NamedArray:
    return wrap_elementwise_unary_op(jnp.bitwise_count)(x)


def bitwise_not(x: NamedArrayLike) -> NamedArray:
    return wrap_elementwise_unary_op(jnp.bitwise_not)(x)


def cbrt(x: NamedArrayLike) -> NamedArray:
    return wrap_elementwise_unary_op(jnp.cbrt)(x)


def ceil(x: NamedArrayLike) -> NamedArray:
    return wrap_elementwise_unary_op(jnp.ceil)(x)


def conj(x: NamedArrayLike) -> NamedArray:
    return wrap_elementwise_unary_op(jnp.conj)(x)


def conjugate(x: NamedArrayLike) -> NamedArray:
    return wrap_elementwise_unary_op(jnp.conjugate)(x)


def copy(x: NamedArrayLike) -> NamedArray:
    return wrap_elementwise_unary_op(jnp.copy)(x)


def cos(x: NamedArrayLike) -> NamedArray:
    return wrap_elementwise_unary_op(jnp.cos)(x)


def cosh(x: NamedArrayLike) -> NamedArray:
    return wrap_elementwise_unary_op(jnp.cosh)(x)


def deg2rad(x: NamedArrayLike) -> NamedArray:
    return wrap_elementwise_unary_op(jnp.deg2rad)(x)


def degrees(x: NamedArrayLike) -> NamedArray:
    return wrap_elementwise_unary_op(jnp.degrees)(x)


def exp(x: NamedArrayLike) -> NamedArray:
    return wrap_elementwise_unary_op(jnp.exp)(x)


def exp2(x: NamedArrayLike) -> NamedArray:
    return wrap_elementwise_unary_op(jnp.exp2)(x)


def expm1(x: NamedArrayLike) -> NamedArray:
    return wrap_elementwise_unary_op(jnp.expm1)(x)


def fabs(x: NamedArrayLike) -> NamedArray:
    return wrap_elementwise_unary_op(jnp.fabs)(x)


def floor(x: NamedArrayLike) -> NamedArray:
    return wrap_elementwise_unary_op(jnp.floor)(x)


def frexp(x: NamedArrayLike) -> tuple[NamedArray, NamedArray]:
    x = util.ensure_named("frexp", x)
    m, e = jnp.frexp(x.array)
    return constructors.array(m, x.axes), constructors.array(e, x.axes)


def imag(x: NamedArrayLike) -> NamedArray:
    return wrap_elementwise_unary_op(jnp.imag)(x)


def invert(x: NamedArrayLike) -> NamedArray:
    return wrap_elementwise_unary_op(jnp.invert)(x)


def isfinite(x: NamedArrayLike) -> NamedArray:
    return wrap_elementwise_unary_op(jnp.isfinite)(x)


def isinf(x: NamedArrayLike) -> NamedArray:
    return wrap_elementwise_unary_op(jnp.isinf)(x)


def isnan(x: NamedArrayLike) -> NamedArray:
    return wrap_elementwise_unary_op(jnp.isnan)(x)


def isneginf(x: NamedArrayLike) -> NamedArray:
    return wrap_elementwise_unary_op(jnp.isneginf)(x)


def isposinf(x: NamedArrayLike) -> NamedArray:
    return wrap_elementwise_unary_op(jnp.isposinf)(x)


def log(x: NamedArrayLike) -> NamedArray:
    return wrap_elementwise_unary_op(jnp.log)(x)


def log10(x: NamedArrayLike) -> NamedArray:
    return wrap_elementwise_unary_op(jnp.log10)(x)


def log1p(x: NamedArrayLike) -> NamedArray:
    return wrap_elementwise_unary_op(jnp.log1p)(x)


def log2(x: NamedArrayLike) -> NamedArray:
    return wrap_elementwise_unary_op(jnp.log2)(x)


def logical_not(x: NamedArrayLike) -> NamedArray:
    return wrap_elementwise_unary_op(jnp.logical_not)(x)


def modf(x: NamedArrayLike, /) -> tuple[NamedArray, NamedArray]:
    x = util.ensure_named("modf", x)
    out0, out1 = jnp.modf(x.array)
    return constructors.array(out0, x.axes), constructors.array(out1, x.axes)


def ndim(x: NamedArrayLike) -> int:
    if isinstance(x, NamedArray):
        return x.array.ndim
    return jnp.ndim(x)


def negative(x: NamedArrayLike) -> NamedArray:
    return wrap_elementwise_unary_op(jnp.negative)(x)


def positive(x: NamedArrayLike) -> NamedArray:
    return wrap_elementwise_unary_op(jnp.positive)(x)


def rad2deg(x: NamedArrayLike) -> NamedArray:
    return wrap_elementwise_unary_op(jnp.rad2deg)(x)


def radians(x: NamedArrayLike) -> NamedArray:
    return wrap_elementwise_unary_op(jnp.radians)(x)


def real(x: NamedArrayLike) -> NamedArray:
    return wrap_elementwise_unary_op(jnp.real)(x)


def reciprocal(x: NamedArrayLike) -> NamedArray:
    return wrap_elementwise_unary_op(jnp.reciprocal)(x)


def rint(x: NamedArrayLike) -> NamedArray:
    return wrap_elementwise_unary_op(jnp.rint)(x)


def rsqrt(x: NamedArrayLike) -> NamedArray:
    return wrap_elementwise_unary_op(lax.rsqrt)(x)  # nb this is in lax


def sign(x: NamedArrayLike) -> NamedArray:
    return wrap_elementwise_unary_op(jnp.sign)(x)


def signbit(x: NamedArrayLike) -> NamedArray:
    return wrap_elementwise_unary_op(jnp.signbit)(x)


def sin(x: NamedArrayLike) -> NamedArray:
    return wrap_elementwise_unary_op(jnp.sin)(x)


def sinc(x: NamedArrayLike) -> NamedArray:
    return wrap_elementwise_unary_op(jnp.sinc)(x)


def sinh(x: NamedArrayLike) -> NamedArray:
    return wrap_elementwise_unary_op(jnp.sinh)(x)


def spacing(x: NamedArrayLike, /) -> NamedArray:
    return wrap_elementwise_unary_op(jnp.spacing)(x)


def square(x: NamedArrayLike) -> NamedArray:
    return wrap_elementwise_unary_op(jnp.square)(x)


def sqrt(x: NamedArrayLike) -> NamedArray:
    return wrap_elementwise_unary_op(jnp.sqrt)(x)


def tan(x: NamedArrayLike) -> NamedArray:
    return wrap_elementwise_unary_op(jnp.tan)(x)


def tanh(x: NamedArrayLike) -> NamedArray:
    return wrap_elementwise_unary_op(jnp.tanh)(x)


def trunc(x: NamedArrayLike) -> NamedArray:
    return wrap_elementwise_unary_op(jnp.trunc)(x)


# Binary Elementwise Operations
def add(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    return wrap_elementwise_binary_op(jnp.add)(x, y)


def atan2(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    return wrap_elementwise_binary_op(jnp.atan2)(x, y)


def arctan2(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    return wrap_elementwise_binary_op(jnp.arctan2)(x, y)


def bitwise_and(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    return wrap_elementwise_binary_op(jnp.bitwise_and)(x, y)


def bitwise_left_shift(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    return wrap_elementwise_binary_op(jnp.bitwise_left_shift)(x, y)


def bitwise_or(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    return wrap_elementwise_binary_op(jnp.bitwise_or)(x, y)


def bitwise_right_shift(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    return wrap_elementwise_binary_op(jnp.bitwise_right_shift)(x, y)


def bitwise_xor(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    return wrap_elementwise_binary_op(jnp.bitwise_xor)(x, y)


def divide(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    return wrap_elementwise_binary_op(jnp.divide)(x, y)


def divmod(x: NamedArrayLike, y: NamedArrayLike, /) -> tuple[NamedArray, NamedArray]:
    a, b = util.ensure_named("divmod", x, y)
    _, _, axes = util.align_shapes_for_broadcast(a.axes, b.axes)
    a_array, b_array = jnp.divmod(a.array, b.array)
    return constructors.array(a_array, axes), constructors.array(b_array, axes)


def equal(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    return wrap_elementwise_binary_op(jnp.equal)(x, y)


def float_power(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    return wrap_elementwise_binary_op(jnp.float_power)(x, y)


def floor_divide(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    return wrap_elementwise_binary_op(jnp.floor_divide)(x, y)


def fmod(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    return wrap_elementwise_binary_op(jnp.fmod)(x, y)


def greater(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    return wrap_elementwise_binary_op(jnp.greater)(x, y)


def greater_equal(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    return wrap_elementwise_binary_op(jnp.greater_equal)(x, y)


def heaviside(x: NamedArrayLike, y: NamedArrayLike) -> NamedArray:
    return wrap_elementwise_binary_op(jnp.heaviside)(x, y)


def hypot(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    return wrap_elementwise_binary_op(jnp.hypot)(x, y)


def left_shift(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    return wrap_elementwise_binary_op(jnp.left_shift)(x, y)


def less(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    return wrap_elementwise_binary_op(jnp.less)(x, y)


def ldexp(x: NamedArrayLike, y: NamedArrayLike) -> NamedArray:
    return wrap_elementwise_binary_op(jnp.ldexp)(x, y)


def less_equal(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    return wrap_elementwise_binary_op(jnp.less_equal)(x, y)


def logaddexp(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    return wrap_elementwise_binary_op(jnp.logaddexp)(x, y)


def logaddexp2(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    return wrap_elementwise_binary_op(jnp.logaddexp2)(x, y)


def logical_and(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    return wrap_elementwise_binary_op(jnp.logical_and)(x, y)


def logical_or(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    return wrap_elementwise_binary_op(jnp.logical_or)(x, y)


def logical_xor(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    return wrap_elementwise_binary_op(jnp.logical_xor)(x, y)


def maximum(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    return wrap_elementwise_binary_op(jnp.maximum)(x, y)


def minimum(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    return wrap_elementwise_binary_op(jnp.minimum)(x, y)


def mod(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    return wrap_elementwise_binary_op(jnp.mod)(x, y)


def multiply(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    return wrap_elementwise_binary_op(jnp.multiply)(x, y)


def nextafter(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    return wrap_elementwise_binary_op(jnp.nextafter)(x, y)


def not_equal(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    return wrap_elementwise_binary_op(jnp.not_equal)(x, y)


def pow(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    return wrap_elementwise_binary_op(jnp.pow)(x, y)


def power(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    return wrap_elementwise_binary_op(jnp.power)(x, y)


def remainder(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    return wrap_elementwise_binary_op(jnp.remainder)(x, y)


def right_shift(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    return wrap_elementwise_binary_op(jnp.right_shift)(x, y)


def subtract(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    return wrap_elementwise_binary_op(jnp.subtract)(x, y)


def true_divide(x: NamedArrayLike, y: NamedArrayLike, /) -> NamedArray:
    return wrap_elementwise_binary_op(jnp.true_divide)(x, y)
