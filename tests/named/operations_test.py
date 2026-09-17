import jax.numpy as jnp
import pytest
from hypothesis import given
from lynqx._src.axis import Axis
from lynqx._src.named.constructors import array
from lynqx._src.named.operations import (
    broadcast_arrays,
    broadcast_to,
)
from strategies import (
    disjoint_named_axis_tuples,
)


@pytest.fixture
def x4():
    return Axis(4, "x")


@pytest.fixture
def y8():
    return Axis(8, "y")


@pytest.fixture
def z16():
    return Axis(16, "z")


@pytest.fixture
def named_3d(x4, y8, z16):
    data = jnp.arange(4 * 8 * 16).reshape((4, 8, 16))
    return array(data, (x4, y8, z16))


@pytest.fixture
def named_with_anon():
    data = jnp.arange(2 * 3 * 4).reshape((2, 3, 4))
    return array(data, (Axis(2, "batch"), Axis(3), Axis(4, "channel")))


class TestBroadcastTo:
    def test_broadcast_expands_unit_axes(self):
        a = array(jnp.ones((1, 8)), (Axis(1, "x"), Axis(8, "y")))
        target_shape = (Axis(4, "x"), Axis(8, "y"))

        res = broadcast_to(a, target_shape)

        assert res.axes == target_shape
        assert res.array.shape == (4, 8)
        assert jnp.array_equal(res.array, jnp.ones((4, 8)))

    def test_broadcast_adds_new_leading_or_trailing_axes(self):
        a = array(jnp.ones((8,)), (Axis(8, "y"),))
        target_shape = (Axis(4, "x"), Axis(8, "y"), Axis(16, "z"))

        res = broadcast_to(a, target_shape)

        assert res.axes == target_shape
        assert res.array.shape == (4, 8, 16)

    def test_broadcast_dict_shape_specifier(self):
        a = array(jnp.ones((1, 8)), (Axis(1, "x"), Axis(8, "y")))
        res = broadcast_to(a, {"x": 4, "y": 8, "z": 16})

        assert res.axes == (Axis(4, "x"), Axis(8, "y"), Axis(16, "z"))
        assert res.array.shape == (4, 8, 16)

    def test_broadcast_dropping_existing_axis_raises(self, named_3d):
        # Target shape missing 'z'
        target_shape = (Axis(4, "x"), Axis(8, "y"))
        with pytest.raises(ValueError, match="broadcast_to cannot drop axes"):
            broadcast_to(named_3d, target_shape)

    def test_broadcast_incompatible_non_unit_size_raises(self):
        a = array(jnp.ones((4, 8)), (Axis(4, "x"), Axis(8, "y")))
        target_shape = (Axis(8, "x"), Axis(8, "y"))

        with pytest.raises(ValueError, match="Cannot broadcast axis 'x' of size 4 to 8"):
            broadcast_to(a, target_shape)

    def test_broadcast_preserves_array_values(self):
        data = jnp.array([1.0, 2.0, 3.0])
        a = array(data, (Axis(3, "x"),))
        res = broadcast_to(a, (Axis(2, "batch"), Axis(3, "x")))

        expected = jnp.broadcast_to(data[None, :], (2, 3))
        assert jnp.array_equal(res.array, expected)

    def test_broadcast_reorders_source_and_inserts_axis_between_matched_axes(self, x4, z16):
        data = jnp.arange(16 * 4).reshape(16, 4)  # axes (z, x), in that order
        a = array(data, (z16, x4))

        target_shape = (x4, Axis(1, "w"), z16)
        res = broadcast_to(a, target_shape)

        assert res.axes == target_shape
        assert res.array.shape == (4, 1, 16)

        # Not just shape: confirm z's data actually landed on the z axis
        # (position 2) and not on w (position 1) as the bug would have it.
        expected = jnp.transpose(data)[:, None, :]  # (x, w, z)
        assert jnp.array_equal(res.array, expected)

    def test_broadcast_dropping_unit_axis_raises(self):
        a = array(jnp.ones((1, 3)), (Axis(1, "x"), Axis(3, "y")))
        with pytest.raises(ValueError, match="broadcast_to cannot drop axes"):
            broadcast_to(a, (Axis(3, "y"),))

    def test_broadcast_matches_anonymous_axis_positionally(self, named_with_anon):
        target_shape = (Axis(2, "batch"), Axis(3), Axis(4, "channel"))
        res = broadcast_to(named_with_anon, target_shape)
        assert res.axes == target_shape

    def test_broadcast_duplicate_axis_name_in_shape_raises(self):
        a = array(jnp.ones((8,)), (Axis(8, "x"),))
        with pytest.raises(ValueError, match="Duplicate axis names"):
            broadcast_to(a, (Axis(8, "x"), Axis(5, "x")))

    def test_broadcast_anonymous_trailing_alignment(self):
        a = array(jnp.ones((3,)), (Axis(3),))
        target_shape = (Axis(2, "batch"), Axis(3))
        res = broadcast_to(a, target_shape)

        assert res.axes == target_shape
        assert res.array.shape == (2, 3)

    def test_broadcast_anonymous_cannot_match_named_target(self):
        a = array(jnp.ones((4, 3)), (Axis(4), Axis(3, "x")))
        target_shape = (Axis(3, "x"), Axis(4, "y"))

        with pytest.raises(ValueError, match="broadcast_to cannot drop axes"):
            broadcast_to(a, target_shape)


class TestBroadcastArrays:
    def test_empty_input_returns_empty_tuple(self):
        assert broadcast_arrays() == ()

    def test_single_array_returns_unchanged(self, x4):
        a = array(jnp.ones((4,)), (x4,))
        (res,) = broadcast_arrays(a)

        assert res.axes == (x4,)
        assert jnp.array_equal(res.array, a.array)

    def test_broadcasts_two_disjoint_named_arrays(self, x4, y8):
        a = array(jnp.ones((4,)), (x4,))
        b = array(jnp.ones((8,)), (y8,))

        res_a, res_b = broadcast_arrays(a, b)

        expected_axes = (x4, y8)
        assert res_a.axes == expected_axes
        assert res_b.axes == expected_axes
        assert res_a.array.shape == (4, 8)
        assert res_b.array.shape == (4, 8)

    def test_broadcasts_three_arrays_with_mixed_and_unit_axes(self):
        ax_x1 = Axis(1, "x")
        ax_x4 = Axis(4, "x")
        ax_y8 = Axis(8, "y")
        ax_z16 = Axis(16, "z")

        a = array(jnp.ones((1, 8)), (ax_x1, ax_y8))
        b = array(jnp.ones((4, 8)), (ax_x4, ax_y8))
        c = array(jnp.ones((16,)), (ax_z16,))

        res_a, res_b, res_c = broadcast_arrays(a, b, c)

        expected_axes = (ax_x4, ax_y8, ax_z16)
        assert res_a.axes == expected_axes
        assert res_b.axes == expected_axes
        assert res_c.axes == expected_axes

        assert res_a.array.shape == (4, 8, 16)
        assert res_b.array.shape == (4, 8, 16)
        assert res_c.array.shape == (4, 8, 16)

    def test_broadcasts_values_correctly(self):
        a = array(jnp.array([1.0, 2.0, 3.0]), (Axis(3, "x"),))
        b = array(jnp.array([10.0, 20.0]), (Axis(2, "y"),))

        res_a, res_b = broadcast_arrays(a, b)

        expected_a = jnp.tile(jnp.array([1.0, 2.0, 3.0])[:, None], (1, 2))
        expected_b = jnp.tile(jnp.array([10.0, 20.0])[None, :], (3, 1))

        assert jnp.array_equal(res_a.array, expected_a)
        assert jnp.array_equal(res_b.array, expected_b)

    def test_incompatible_axis_sizes_raises_value_error(self):
        a = array(jnp.ones((4,)), (Axis(4, "x"),))
        b = array(jnp.ones((8,)), (Axis(8, "x"),))

        with pytest.raises(ValueError, match="Incompatible sizes for axis 'x'"):
            broadcast_arrays(a, b)

    def test_incompatible_anonymous_axis_sizes_raises_value_error(self):
        a = array(jnp.ones((4,)), (Axis(4),))
        b = array(jnp.ones((8,)), (Axis(8),))

        with pytest.raises(ValueError, match="Cannot align anonymous axes"):
            broadcast_arrays(a, b)

    @given(disjoint_named_axis_tuples(min_size=1, max_size=3))
    def test_broadcast_arrays_idempotent_shapes(self, axes):
        shape = tuple(ax.size for ax in axes)
        arr = jnp.ones(shape)
        named_arr = array(arr, axes)

        (res,) = broadcast_arrays(named_arr)
        assert res.axes == named_arr.axes
        assert res.array.shape == named_arr.array.shape

    def test_broadcast_arrays_preserves_left_hand_axis_order(self):
        a = array(jnp.ones((4, 8, 3)), (Axis(4, "x"), Axis(8, "y"), Axis(3, "i")))
        b = array(jnp.ones((3,)), (Axis(3, "i"),))

        res_a, res_b = broadcast_arrays(a, b)

        expected_axes = (Axis(4, "x"), Axis(8, "y"), Axis(3, "i"))  # NOT (i, x, y)
        assert res_a.axes == expected_axes
        assert res_b.axes == expected_axes
