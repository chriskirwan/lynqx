import jax
import jax.numpy as jnp
import pytest
from hypothesis import given
from lynqx._src.axis import Axis
from lynqx._src.named.constructors import array
from lynqx._src.named.creation import (
    arange,
    empty,
    empty_like,
    full,
    full_like,
    iota,
    ones,
    ones_like,
    zeros,
    zeros_like,
)
from lynqx._src.partition import PM
from strategies import disjoint_named_axis_tuples


@pytest.fixture
def x4():
    return Axis(4, "x")


@pytest.fixture
def y8():
    return Axis(8, "y")


@pytest.fixture
def z16():
    return Axis(16, "z")


class TestZeros:
    def test_shape_and_axes(self, x4, y8):
        got = zeros({"x": 4, "y": 8})
        assert got.axes == (x4, y8)
        assert got.array.shape == (4, 8)
        assert jnp.all(got.array == 0)

    def test_single_axis_not_wrapped_in_a_sequence(self, x4):
        got = zeros(x4)
        assert got.axes == (x4,)

    def test_explicit_dtype(self, x4):
        got = zeros(x4, dtype=jnp.int32)
        assert got.dtype == jnp.int32

    def test_default_dtype_matches_jnp(self, x4):
        got = zeros(x4)
        assert got.array.dtype == jnp.zeros((4,)).dtype


class TestOnes:
    def test_shape_and_values(self, x4, y8):
        got = ones({"x": 4, "y": 8})
        assert got.axes == (x4, y8)
        assert jnp.all(got.array == 1)

    def test_explicit_dtype(self, x4):
        got = ones(x4, dtype=jnp.int32)
        assert got.dtype == jnp.int32


class TestEmpty:
    def test_is_zero_initialized(self, x4):
        got = empty(x4)
        assert jnp.all(got.array == 0)


class TestFull:
    def test_fills_with_scalar_value(self, x4, y8):
        got = full({"x": 4, "y": 8}, 3.5)
        assert got.axes == (x4, y8)
        assert jnp.array_equal(got.array, jnp.full((4, 8), 3.5))

    def test_fills_with_named_array_scalar(self, x4):
        fill = array(jnp.array(9.0), ())
        got = full(x4, fill)
        assert jnp.array_equal(got.array, jnp.full((4,), 9.0))

    def test_duplicate_axis_names_raise(self):
        with pytest.raises(ValueError, match="Duplicate axis names"):
            full((Axis(4, "x"), Axis(8, "x")), 0.0)


LIKE_FUNCTIONS = (
    (zeros_like, 0.0),
    (ones_like, 1.0),
    (empty_like, 0.0),
)
LIKE_FUNCTION_IDS = ["zeros_like", "ones_like", "empty_like"]


class TestLikeFunctions:
    @pytest.mark.parametrize("fn,expected_fill", LIKE_FUNCTIONS, ids=LIKE_FUNCTION_IDS)
    def test_shape_and_axes_match_source(self, fn, expected_fill, x4, y8):
        a = array(jnp.arange(32.0).reshape(4, 8), (x4, y8))
        got = fn(a)
        assert got.axes == a.axes
        assert jnp.array_equal(got.array, jnp.full((4, 8), expected_fill))

    @pytest.mark.parametrize("fn,_expected_fill", LIKE_FUNCTIONS, ids=LIKE_FUNCTION_IDS)
    def test_shape_argument_overrides_source_axes(self, fn, _expected_fill, x4, y8, z16):
        a = array(jnp.ones((4, 8)), (x4, y8))
        got = fn(a, shape=(z16,))
        assert got.axes == (z16,)

    @pytest.mark.parametrize("fn,_expected_fill", LIKE_FUNCTIONS, ids=LIKE_FUNCTION_IDS)
    def test_dtype_defaults_to_source_array_dtype(self, fn, _expected_fill, x4):
        a = array(jnp.arange(4, dtype=jnp.int32), (x4,))
        got = fn(a)
        assert got.dtype == jnp.int32


class TestFullLike:
    def test_uses_given_fill_value_and_source_axes(self, x4, y8):
        a = array(jnp.zeros((4, 8)), (x4, y8))
        got = full_like(a, 3.0)
        assert got.axes == a.axes
        assert jnp.array_equal(got.array, jnp.full((4, 8), 3.0))

    def test_shape_argument_overrides_source_axes(self, x4, y8, z16):
        a = array(jnp.zeros((4, 8)), (x4, y8))
        got = full_like(a, 3.0, shape=(z16,))
        assert got.axes == (z16,)

    def test_dtype_defaults_to_source_array_dtype(self, x4):
        a = array(jnp.arange(4, dtype=jnp.int32), (x4,))
        got = full_like(a, 5)
        assert got.dtype == jnp.int32


class TestIota:
    def test_1d_matches_lax_iota(self, x4):
        got = iota(x4, dtype=jnp.int32)
        assert got.axes == (x4,)
        assert jnp.array_equal(got.array, jax.lax.iota(jnp.int32, 4))

    def test_1d_rejects_an_explicit_axis(self, x4):
        with pytest.raises(ValueError, match="must not be provided"):
            iota(x4, axis="x", dtype=jnp.int32)

    def test_multi_axis_requires_an_axis_argument(self, x4, y8):
        with pytest.raises(ValueError):
            iota((x4, y8), dtype=jnp.int32)

    def test_multi_axis_increments_along_the_named_axis(self, x4, y8):
        got = iota((x4, y8), axis="y", dtype=jnp.int32)
        want = jax.lax.broadcasted_iota(jnp.int32, (4, 8), 1)
        assert got.axes == (x4, y8)
        assert jnp.array_equal(got.array, want)

    def test_multi_axis_accepts_a_positional_axis_selector(self, x4, y8):
        got = iota((x4, y8), axis=0, dtype=jnp.int32)
        want = jax.lax.broadcasted_iota(jnp.int32, (4, 8), 0)
        assert jnp.array_equal(got.array, want)

    def test_unresolvable_axis_raises(self, x4, y8):
        with pytest.raises(ValueError):
            iota((x4, y8), axis="nope", dtype=jnp.int32)

    def test_default_dtype_matches_canonicalized_none(self, x4):
        got = iota(x4)
        assert got.array.dtype == jax.dtypes.canonicalize_dtype(None)


class TestArange:
    def test_default_start_and_step(self, x4):
        got = arange(x4, dtype=jnp.float32)
        assert got.axes == (x4,)
        assert jnp.array_equal(got.array, jnp.arange(4, dtype=jnp.float32))

    def test_explicit_start_and_step(self, x4):
        got = arange(x4, start=2, step=3, dtype=jnp.float32)
        want = 2 + 3 * jnp.arange(4, dtype=jnp.float32)
        assert jnp.array_equal(got.array, want)

    def test_dtype_inferred_as_floating_from_float_start_and_step(self, x4):
        got = arange(x4, start=1.0, step=0.5)
        assert jnp.issubdtype(got.array.dtype, jnp.floating)

    def test_dtype_inferred_as_integer_from_int_start_and_step(self, x4):
        got = arange(x4, start=1, step=2)
        assert jnp.issubdtype(got.array.dtype, jnp.integer)

    def test_complex_start_raises(self, x4):
        with pytest.raises(ValueError, match="complex"):
            arange(x4, start=1 + 2j)

    def test_complex_step_raises(self, x4):
        with pytest.raises(ValueError, match="complex"):
            arange(x4, step=1j)


class TestCreationProperties:
    @given(disjoint_named_axis_tuples(min_size=1, max_size=3))
    def test_zeros_axes_and_shape_match_spec(self, axes):
        got = zeros(axes)
        assert got.axes == axes
        assert got.array.shape == tuple(ax.size for ax in axes)
        assert jnp.all(got.array == 0)

    @given(disjoint_named_axis_tuples(min_size=1, max_size=3))
    def test_full_matches_jnp_full(self, axes):
        got = full(axes, 3.5)
        want = jnp.full(tuple(ax.size for ax in axes), 3.5)
        assert got.axes == axes
        assert jnp.array_equal(got.array, want)


class TestOutSharding:
    def test_zeros_respects_out_sharding_inside_a_mesh(self, mesh_1d):
        pm = PM({"batch": "x"})
        with jax.set_mesh(mesh_1d):
            got = zeros({"batch": 8}, out_sharding=pm)
        assert got.axes == (Axis(8, "batch"),)
