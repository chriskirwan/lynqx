import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from lynqx._src.array import NamedArrayImpl
from lynqx._src.axis import Axis
from lynqx._src.filters import (
    assert_single_array_leaf,
    is_arrayish,
    is_inexact_arrayish,
    is_inexact_jax_array,
    is_inexact_jax_array_like,
    is_inexact_named_array,
    is_jax_array,
    is_jax_array_like,
    is_named_array,
    is_scalar,
)


# name, value, expected (is_array, is_array_like, is_inexact_array, is_inexact_array_like)
_VALUES = (
    ("python_int", 5, (False, True, False, False)),
    ("python_float", 5.0, (False, True, False, True)),
    ("python_bool", True, (False, True, False, False)),
    ("python_complex", 1 + 2j, (False, True, False, True)),
    ("numpy_int_array", np.array([1, 2, 3]), (True, True, False, False)),
    ("numpy_float_array", np.array([1.0, 2.0]), (True, True, True, True)),
    ("jax_int_array", jnp.array([1, 2, 3]), (True, True, False, False)),
    ("jax_float_array", jnp.array([1.0, 2.0]), (True, True, True, True)),
    ("string", "hello", (False, False, False, False)),
    ("list", [1, 2, 3], (False, False, False, False)),
    ("none", None, (False, False, False, False)),
)
_VALUE_IDS = [v[0] for v in _VALUES]


class TestBasicArrayPredicates:
    @pytest.mark.parametrize("_name,value,expected", _VALUES, ids=_VALUE_IDS)
    def test_is_jax_array(self, _name, value, expected):
        assert is_jax_array(value) is expected[0]

    @pytest.mark.parametrize("_name,value,expected", _VALUES, ids=_VALUE_IDS)
    def test_is_jax_array_like(self, _name, value, expected):
        assert is_jax_array_like(value) is expected[1]

    @pytest.mark.parametrize("_name,value,expected", _VALUES, ids=_VALUE_IDS)
    def test_is_inexact_jax_array(self, _name, value, expected):
        assert is_inexact_jax_array(value) is expected[2]

    @pytest.mark.parametrize("_name,value,expected", _VALUES, ids=_VALUE_IDS)
    def test_is_inexact_jax_array_like(self, _name, value, expected):
        assert is_inexact_jax_array_like(value) is expected[3]


@pytest.fixture
def float_named():
    return NamedArrayImpl(jnp.array([1.0, 2.0, 3.0]), (Axis(3, "x"),))


@pytest.fixture
def int_named():
    return NamedArrayImpl(jnp.array([1, 2, 3]), (Axis(3, "x"),))


class TestNamedArrayPredicates:
    def test_is_named_array_true_for_a_named_array(self, float_named):
        assert is_named_array(float_named) is True

    @pytest.mark.parametrize("value", [5, 5.0, jnp.array([1.0]), np.array([1.0]), "x", None])
    def test_is_named_array_false_for_everything_else(self, value):
        assert is_named_array(value) is False

    def test_is_inexact_named_array_true_for_float_named_array(self, float_named):
        assert is_inexact_named_array(float_named) is True

    def test_is_inexact_named_array_false_for_int_named_array(self, int_named):
        assert is_inexact_named_array(int_named) is False

    def test_is_inexact_named_array_false_for_a_bare_array(self):
        assert is_inexact_named_array(jnp.array([1.0, 2.0])) is False

    def test_is_arrayish_true_for_named_array(self, float_named):
        assert is_arrayish(float_named) is True

    def test_is_arrayish_true_for_a_bare_jax_array(self):
        assert is_arrayish(jnp.array([1.0])) is True

    def test_is_arrayish_false_for_a_python_scalar(self):
        assert is_arrayish(5.0) is False

    def test_is_inexact_arrayish_unwraps_named_array(self, float_named, int_named):
        assert is_inexact_arrayish(float_named) is True
        assert is_inexact_arrayish(int_named) is False

    def test_is_inexact_arrayish_on_a_bare_array(self):
        assert is_inexact_arrayish(jnp.array([1.0])) is True
        assert is_inexact_arrayish(jnp.array([1])) is False


class TestIsScalar:
    def test_true_for_a_zero_dim_named_array(self):
        assert is_scalar(NamedArrayImpl(jnp.array(1.0), ())) is True

    def test_false_for_a_one_dim_named_array(self, float_named):
        assert is_scalar(float_named) is False

    def test_true_for_python_scalars(self):
        assert is_scalar(5.0) is True
        assert is_scalar(5) is True

    def test_false_for_a_bare_array_with_shape(self):
        assert is_scalar(jnp.array([1.0, 2.0])) is False

    def test_true_for_a_zero_dim_bare_array(self):
        assert is_scalar(jnp.array(1.0)) is True


class TestAssertSingleArrayLeaf:
    def test_passes_for_an_ordinary_named_array(self, float_named):
        assert_single_array_leaf(float_named, "some_fn")  # should not raise

    def test_ignores_non_named_array_input(self):
        assert_single_array_leaf(jnp.array([1.0, 2.0]), "some_fn")  # should not raise
        assert_single_array_leaf(5.0, "some_fn")  # should not raise

    def test_raises_when_the_array_leaf_has_been_filtered_out(self, float_named):
        # `eqx.partition` splits `float_named` into a "dynamic" half (the
        # array leaf, kept because it matches `eqx.is_array`) and a "static"
        # half (`array=None`). `None` flattens to zero pytree leaves, which
        # is exactly the array-leaf-count mismatch this function checks for.
        _dynamic, static = eqx.partition(float_named, eqx.is_array)

        with pytest.raises(AssertionError, match="some_fn"):
            assert_single_array_leaf(static, "some_fn")

    def test_dynamic_half_of_a_partition_still_passes(self, float_named):
        dynamic, _static = eqx.partition(float_named, eqx.is_array)
        assert_single_array_leaf(dynamic, "some_fn")  # should not raise
