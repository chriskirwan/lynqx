import operator

import jax.numpy as jnp
import pytest
from lynqx._src.axis import Axis
from lynqx._src.named import ufuncs
from lynqx._src.named.constructors import array
from lynqx._src.named.indexing import NamedIndexUpdateHelper
from lynqx._src.named.methods import _set_namedarray_attributes, register_namedarray_methods


register_namedarray_methods()
del register_namedarray_methods


@pytest.fixture
def x4():
    return Axis(4, "x")


FLOAT_OPERATORS = (
    (operator.add, ufuncs.add),
    (operator.sub, ufuncs.subtract),
    (operator.mul, ufuncs.multiply),
    (operator.truediv, ufuncs.true_divide),
    (operator.floordiv, ufuncs.floor_divide),
    (operator.mod, ufuncs.mod),
    (operator.pow, ufuncs.pow),
)
FLOAT_OPERATOR_IDS = ["add", "sub", "mul", "truediv", "floordiv", "mod", "pow"]

# `rshift`'s reflected form is deliberately excluded -- see
# `TestReflectedShiftOperators` below.
INT_FORWARD_OPERATORS = (
    (operator.and_, ufuncs.bitwise_and),
    (operator.or_, ufuncs.bitwise_or),
    (operator.xor, ufuncs.bitwise_xor),
    (operator.lshift, ufuncs.left_shift),
    (operator.rshift, ufuncs.right_shift),
)
INT_FORWARD_OPERATOR_IDS = ["and", "or", "xor", "lshift", "rshift"]

INT_REFLECTED_OPERATORS = (
    (operator.and_, ufuncs.bitwise_and),
    (operator.or_, ufuncs.bitwise_or),
    (operator.xor, ufuncs.bitwise_xor),
    (operator.lshift, ufuncs.left_shift),
)
INT_REFLECTED_OPERATOR_IDS = ["and", "or", "xor", "lshift"]

COMPARISON_OPERATORS = (
    (operator.eq, ufuncs.equal),
    (operator.ne, ufuncs.not_equal),
    (operator.lt, ufuncs.less),
    (operator.le, ufuncs.less_equal),
    (operator.gt, ufuncs.greater),
    (operator.ge, ufuncs.greater_equal),
)
COMPARISON_OPERATOR_IDS = ["eq", "ne", "lt", "le", "gt", "ge"]

UNARY_OPERATORS = (
    (operator.neg, ufuncs.negative),
    (operator.pos, ufuncs.positive),
    (operator.abs, ufuncs.abs),
)
UNARY_OPERATOR_IDS = ["neg", "pos", "abs"]


class TestFloatBinaryOperatorDispatch:
    @pytest.mark.parametrize("py_op,ufunc", FLOAT_OPERATORS, ids=FLOAT_OPERATOR_IDS)
    def test_forward(self, x4, py_op, ufunc):
        a = array(jnp.array([4.0, 9.0, 16.0, 25.0]), (x4,))
        b = array(jnp.array([2.0, 3.0, 4.0, 5.0]), (x4,))

        got = py_op(a, b)
        want = ufunc(a, b)

        assert jnp.array_equal(got.array, want.array)
        assert got.axes == want.axes

    @pytest.mark.parametrize("py_op,ufunc", FLOAT_OPERATORS, ids=FLOAT_OPERATOR_IDS)
    def test_reflected_when_lhs_is_a_bare_python_scalar(self, x4, py_op, ufunc):
        a = array(jnp.array([4.0, 9.0, 16.0, 25.0]), (x4,))

        # `2.0.__op__(a)` returns NotImplemented (a `float` doesn't know
        # about NamedArray), so Python falls back to `a.__rop__(2.0)`.
        got = py_op(2.0, a)
        want = ufunc(2.0, a)

        assert jnp.array_equal(got.array, want.array)
        assert got.axes == want.axes


class TestIntBinaryOperatorDispatch:
    @pytest.mark.parametrize("py_op,ufunc", INT_FORWARD_OPERATORS, ids=INT_FORWARD_OPERATOR_IDS)
    def test_forward(self, x4, py_op, ufunc):
        a = array(jnp.array([12, 9, 20, 5], dtype=jnp.int32), (x4,))
        b = array(jnp.array([3, 2, 1, 4], dtype=jnp.int32), (x4,))

        got = py_op(a, b)
        want = ufunc(a, b)

        assert jnp.array_equal(got.array, want.array)

    @pytest.mark.parametrize("py_op,ufunc", INT_REFLECTED_OPERATORS, ids=INT_REFLECTED_OPERATOR_IDS)
    def test_reflected_when_lhs_is_a_bare_python_scalar(self, x4, py_op, ufunc):
        a = array(jnp.array([12, 9, 20, 5], dtype=jnp.int32), (x4,))

        got = py_op(3, a)
        want = ufunc(3, a)

        assert jnp.array_equal(got.array, want.array)


class TestReflectedShiftOperators:
    """The reflected shift operators are the one place `_operator_r*` doesn't
    just mirror its forward counterpart with swapped arguments, so they get
    dedicated (rather than table-driven) tests."""

    def test_rlshift_reverses_argument_order(self, x4):
        a = array(jnp.array([1, 2, 3, 4], dtype=jnp.int32), (x4,))

        got = 3 << a  # dispatches to `a.__rlshift__(3)`
        want = ufuncs.left_shift(3, a)

        assert jnp.array_equal(got.array, want.array)

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "_operator_rrshift computes `ufuncs.right_shift(self, other)` instead "
            "of `ufuncs.right_shift(other, self)` -- argument order isn't reversed, "
            "unlike every other reflected operator in methods.py (compare against "
            "_operator_rlshift, which reverses correctly)."
        ),
    )
    def test_rrshift_reverses_argument_order(self, x4):
        a = array(jnp.array([1, 2, 3, 4], dtype=jnp.int32), (x4,))

        got = 64 >> a  # dispatches to `a.__rrshift__(64)`
        want = ufuncs.right_shift(64, a)

        assert jnp.array_equal(got.array, want.array)


class TestComparisonOperatorDispatch:
    @pytest.mark.parametrize("py_op,ufunc", COMPARISON_OPERATORS, ids=COMPARISON_OPERATOR_IDS)
    def test_dispatch(self, x4, py_op, ufunc):
        a = array(jnp.array([1.0, 2.0, 3.0, 4.0]), (x4,))
        b = array(jnp.array([4.0, 3.0, 3.0, 1.0]), (x4,))

        got = py_op(a, b)
        want = ufunc(a, b)

        assert jnp.array_equal(got.array, want.array)


class TestUnaryOperatorDispatch:
    @pytest.mark.parametrize("py_op,ufunc", UNARY_OPERATORS, ids=UNARY_OPERATOR_IDS)
    def test_dispatch(self, x4, py_op, ufunc):
        a = array(jnp.array([-2.0, -1.0, 0.0, 3.0]), (x4,))

        got = py_op(a)
        want = ufunc(a)

        assert jnp.array_equal(got.array, want.array)
        assert got.axes == want.axes

    def test_invert_dispatches_for_bool_arrays(self, x4):
        a = array(jnp.array([True, False, True, False]), (x4,))

        got = ~a
        want = ufuncs.invert(a)

        assert jnp.array_equal(got.array, want.array)


class TestUnsupportedOperators:
    def test_matmul_raises_type_error(self, x4):
        a = array(jnp.ones((4,)), (x4,))
        b = array(jnp.ones((4,)), (x4,))

        with pytest.raises(TypeError, match="matrix multiplication"):
            a @ b

    def test_rmatmul_raises_type_error(self, x4):
        # Called directly rather than via `bare_array @ a`: whether the
        # bare-array LHS's own `__matmul__` returns NotImplemented (routing
        # to `__rmatmul__`) or raises some other error first isn't something
        # this test should have to pin down -- the thing actually owned by
        # this module is `_unimplemented_matmul` itself.
        a = array(jnp.ones((4,)), (x4,))
        with pytest.raises(TypeError, match="matrix multiplication"):
            a.__rmatmul__(jnp.ones((4,)))

    def test_setitem_raises_type_error(self, x4):
        a = array(jnp.ones((4,)), (x4,))
        with pytest.raises(TypeError, match="immutable"):
            a[0] = 1.0


class TestAtProperty:
    def test_returns_named_index_update_helper(self, x4):
        a = array(jnp.ones((4,)), (x4,))
        assert isinstance(a.at, NamedIndexUpdateHelper)


class TestSetNamedArrayAttributes:
    def test_include_restricts_to_the_given_attributes(self):
        class Dummy:
            pass

        _set_namedarray_attributes(Dummy, include={"__add__"})

        assert hasattr(Dummy, "__add__")
        assert not hasattr(Dummy, "__sub__")

    def test_exclude_skips_the_given_attributes(self):
        class Dummy:
            pass

        _set_namedarray_attributes(Dummy, exclude={"__add__"})

        assert not hasattr(Dummy, "__add__")
        assert hasattr(Dummy, "__sub__")
