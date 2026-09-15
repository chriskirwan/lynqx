import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, strategies as st
from lynqx._src.axis import Axis
from lynqx._src.named.constructors import array
from lynqx._src.named.indexing import (
    dynamic_slice,
    dynamic_update_slice,
    NamedIndexUpdateHelper,
    take,
    take_along_axis,
)
from lynqx._src.named.methods import register_namedarray_methods


register_namedarray_methods()
del register_namedarray_methods


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
    data = jnp.arange(4 * 8 * 16, dtype=jnp.float32).reshape((4, 8, 16))
    return array(data, (x4, y8, z16))


class TestBasicGetitem:
    def test_dict_form_by_name(self, named_3d, y8, z16):
        res = named_3d[{"x": 2}]
        assert res.axes == (y8, z16)
        assert jnp.array_equal(res.array, named_3d.array[2])

    def test_dict_form_slice_resizes_axis(self, named_3d, x4, z16):
        res = named_3d[{"y": slice(1, 4)}]
        assert res.axes == (x4, Axis(3, "y"), z16)
        assert jnp.array_equal(res.array, named_3d.array[:, 1:4, :])

    def test_dict_form_unmentioned_axes_pass_through(self, named_3d):
        res = named_3d[{"x": 1}]
        assert jnp.array_equal(res.array, named_3d.array[1, :, :])

    def test_dict_form_duplicate_target_raises(self, named_3d, x4):
        with pytest.raises(ValueError, match="only one index"):
            named_3d[{"x": 1, x4: 2}]

    def test_dict_form_bare_array_raises(self, named_3d):
        with pytest.raises(TypeError, match="NamedArray"):
            named_3d[{"x": jnp.array([0, 1])}]

    def test_dict_form_unresolved_axis_raises(self, named_3d):
        with pytest.raises(ValueError, match="not found"):
            named_3d[{"nope": 0}]

    def test_positional_tuple_matches_numpy_semantics(self, named_3d):
        res = named_3d[1, :, 2:5]
        assert jnp.array_equal(res.array, named_3d.array[1, :, 2:5])
        assert res.axes == (Axis(8, "y"), Axis(3, "z"))

    def test_bare_int_pads_remaining_axes_implicitly(self, named_3d, y8, z16):
        res = named_3d[1]
        assert res.axes == (y8, z16)
        assert jnp.array_equal(res.array, named_3d.array[1])

    def test_ellipsis_expands_to_fill_middle(self, named_3d, y8):
        res = named_3d[1, ..., 3]
        assert res.axes == (y8,)
        assert jnp.array_equal(res.array, named_3d.array[1, :, 3])

    def test_too_many_ellipses_raises(self, named_3d):
        with pytest.raises(IndexError, match="Ellipsis"):
            named_3d[..., ...]

    def test_too_many_indices_raises(self, named_3d):
        with pytest.raises(IndexError, match="too many"):
            named_3d[0, 0, 0, 0]

    def test_negative_int_index_consumes_axis(self, named_3d, x4, y8):
        res = named_3d[:, :, -1]
        assert res.axes == (x4, y8)
        assert jnp.array_equal(res.array, named_3d.array[:, :, -1])

    def test_traced_slice_bound_redirects_to_dynamic_slice_in_error(self, x4):
        a = array(jnp.arange(4.0), (x4,))

        @jax.jit
        def f(start):
            return a[slice(start, start + 2)].array

        with pytest.raises(TypeError, match="dynamic_slice"):
            f(1)


class TestAt:
    def test_get_matches_getitem(self, named_3d):
        via_at = NamedIndexUpdateHelper(named_3d)[{"x": 1}].get()
        via_getitem = named_3d[{"x": 1}]
        assert via_at.axes == via_getitem.axes
        assert jnp.array_equal(via_at.array, via_getitem.array)

    def test_set_returns_full_source_axes(self, named_3d):
        ref = NamedIndexUpdateHelper(named_3d)[{"x": 1}]
        updated = ref.set(array(jnp.zeros((8, 16)), ref.output_axes))
        assert updated.axes == named_3d.axes
        assert jnp.array_equal(updated.array[1], jnp.zeros((8, 16)))
        assert jnp.array_equal(updated.array[0], named_3d.array[0])

    def test_set_broadcasts_a_smaller_named_value(self, x4, y8):
        a = array(jnp.ones((4, 8)), (x4, y8))
        ref = NamedIndexUpdateHelper(a)[{"x": slice(1, 3)}]
        # value has only the "y" axis -- must broadcast across the sliced "x" axis
        value = array(jnp.arange(8.0), (y8,))

        updated = ref.set(value)

        assert jnp.array_equal(updated.array[1], jnp.arange(8.0))
        assert jnp.array_equal(updated.array[2], jnp.arange(8.0))
        assert jnp.array_equal(updated.array[0], jnp.ones(8))

    def test_add_accumulates_rather_than_overwrites(self, x4):
        a = array(jnp.ones((4,)), (x4,))
        ref = NamedIndexUpdateHelper(a)[{"x": slice(0, 2)}]
        updated = ref.add(array(jnp.array([1.0, 1.0]), (Axis(2, "x"),)))
        assert jnp.array_equal(updated.array, jnp.array([2.0, 2.0, 1.0, 1.0]))


class TestAdvancedIndexing:
    def test_bare_array_positionally_gives_anonymous_axis(self, x4, y8):
        a = array(jnp.arange(32.0).reshape(4, 8), (x4, y8))
        res = a[jnp.array([0, 2]), :]
        assert res.axes == (Axis(2), y8)
        assert jnp.array_equal(res.array, a.array[jnp.array([0, 2]), :])

    def test_named_index_positionally_keeps_its_name(self, x4, y8):
        a = array(jnp.arange(32.0).reshape(4, 8), (x4, y8))
        idx = array(jnp.array([0, 2]), (Axis(2, "k"),))
        res = a[idx, :]
        assert res.axes == (Axis(2, "k"), y8)

    def test_contiguous_consumed_axes_insert_broadcast_axes_in_place(self, x4, y8, z16):
        a = array(jnp.arange(4 * 8 * 16).reshape(4, 8, 16), (x4, y8, z16))
        idx = array(jnp.array([0, 1]), (Axis(2, "k"),))
        # consumed positions = {1} (just "y") -- trivially contiguous
        res = a[:, idx, :]
        assert res.axes == (x4, Axis(2, "k"), z16)

    def test_noncontiguous_consumed_axes_move_broadcast_axes_to_front(self, x4, y8, z16):
        a = array(jnp.arange(4 * 8 * 16).reshape(4, 8, 16), (x4, y8, z16))
        idx_x = array(jnp.array([0, 1]), (Axis(2, "k"),))
        idx_z = array(jnp.array([0, 1]), (Axis(2, "k"),))
        # consumed positions = {0, 2}, interleaved with the kept "y" axis at 1
        res = a[idx_x, :, idx_z]
        assert res.axes == (Axis(2, "k"), y8)
        expected = a.array[jnp.array([0, 1]), :, jnp.array([0, 1])]
        assert jnp.array_equal(res.array, expected)

    def test_named_batched_gather_injects_implicit_arange(self):
        # "batch" is shared between `weights`'s own axis and `idx`'s own axis,
        # and is left fully unconstrained on `weights` -- so it should be
        # promoted into an aligned gather, not broadcast as an independent axis.
        weights = array(jnp.arange(12.0).reshape(3, 4), (Axis(3, "batch"), Axis(4, "vocab")))
        idx = array(jnp.array([[0, 1], [1, 2], [3, 0]]), (Axis(3, "batch"), Axis(2, "k")))

        res = weights[{"vocab": idx}]

        expected = weights.array[jnp.arange(3)[:, None], idx.array]
        assert jnp.array_equal(res.array, expected)
        assert [ax.name for ax in res.axes] == ["batch", "k"]

    def test_explicit_bound_on_a_colliding_name_raises_instead_of_overriding(self):
        # "batch" is shared, but the caller explicitly bounded `weights`'s own
        # "batch" axis -- that's a real conflict, not something to silently
        # resolve one way or the other.
        weights = array(jnp.arange(12.0).reshape(3, 4), (Axis(3, "batch"), Axis(4, "vocab")))
        idx = array(jnp.array([[0, 1], [1, 0]]), (Axis(2, "batch"), Axis(2, "k")))
        with pytest.raises(ValueError):
            weights[{"batch": slice(0, 2), "vocab": idx}]

    def test_dict_form_advanced_index_must_be_named_array(self, x4, y8):
        a = array(jnp.arange(32.0).reshape(4, 8), (x4, y8))
        with pytest.raises(TypeError, match="NamedArray"):
            a[{"x": jnp.array([0, 1])}]


class TestGetitemAtParity:
    @pytest.mark.parametrize(
        "key",
        [
            {"x": 1},
            {"y": slice(1, 4)},
            (1, slice(None), slice(2, 5)),
        ],
    )
    def test_getitem_and_at_get_agree(self, named_3d, key):
        via_getitem = named_3d[key]
        via_at = NamedIndexUpdateHelper(named_3d)[key].get()
        assert via_getitem.axes == via_at.axes
        assert jnp.array_equal(via_getitem.array, via_at.array)


class TestDynamicSlice:
    def test_matches_lax_dynamic_slice(self, x4, y8):
        a = array(jnp.arange(32.0).reshape(4, 8), (x4, y8))
        res = dynamic_slice(a, start={"y": 2}, length={"y": 3})
        assert res.axes == (x4, Axis(3, "y"))
        assert jnp.array_equal(res.array, jax.lax.dynamic_slice(a.array, (0, 2), (4, 3)))

    def test_unspecified_axes_read_in_full(self, x4, y8):
        a = array(jnp.arange(32.0).reshape(4, 8), (x4, y8))
        res = dynamic_slice(a, start={"y": 0}, length={"y": 8})
        assert res.axes == (x4, y8)

    def test_mismatched_start_and_length_axes_raises(self, x4):
        a = array(jnp.arange(4.0), (x4,))
        with pytest.raises(ValueError, match="same axes"):
            dynamic_slice(a, start={"x": 0}, length={})

    def test_out_of_bounds_start_is_clamped_not_raised(self, x4):
        a = array(jnp.arange(4.0), (x4,))
        res = dynamic_slice(a, start={"x": 100}, length={"x": 2})
        # clamped to the last valid start (2), matching lax.dynamic_slice's own contract
        assert jnp.array_equal(res.array, jnp.array([2.0, 3.0]))

    def test_traced_start_works_under_jit(self, x4):
        a = array(jnp.arange(4.0), (x4,))

        @jax.jit
        def f(start):
            return dynamic_slice(a, start={"x": start}, length={"x": 2}).array

        assert jnp.array_equal(f(1), jnp.array([1.0, 2.0]))


class TestDynamicUpdateSlice:
    def test_matches_lax_dynamic_update_slice(self, x4):
        a = array(jnp.zeros((4,)), (x4,))
        update = array(jnp.array([9.0, 9.0]), (Axis(2, "x"),))
        res = dynamic_update_slice(a, update, start={"x": 1})
        assert jnp.array_equal(res.array, jnp.array([0.0, 9.0, 9.0, 0.0]))
        assert res.axes == a.axes

    def test_axis_name_mismatch_raises(self, x4):
        a = array(jnp.zeros((4,)), (x4,))
        update = array(jnp.array([9.0, 9.0]), (Axis(2, "not_x"),))
        with pytest.raises(ValueError, match="names disagree"):
            dynamic_update_slice(a, update, start={"x": 0})


class TestTake:
    def test_named_index_axes_replace_the_gathered_axis(self, x4, y8):
        a = array(jnp.arange(32.0).reshape(4, 8), (x4, y8))
        idx = array(jnp.array([[0, 1], [2, 3]]), (Axis(2, "batch"), Axis(2, "k")))

        res = take(a, "x", idx)

        assert res.axes == (Axis(2, "batch"), Axis(2, "k"), y8)
        assert jnp.array_equal(res.array, jnp.take(a.array, idx.array, axis=0))

    def test_anonymous_index_gives_anonymous_output_axis(self, x4, y8):
        a = array(jnp.arange(32.0).reshape(4, 8), (x4, y8))
        idx = array(jnp.array([0, 2]), (Axis(2),))

        res = take(a, "x", idx)

        assert res.axes == (Axis(2), y8)

    def test_name_collision_with_surviving_axis_raises(self, x4, y8):
        a = array(jnp.arange(32.0).reshape(4, 8), (x4, y8))
        idx = array(jnp.array([0, 1]), (y8.resize(2),))  # named "y", collides with a's own "y"

        with pytest.raises(ValueError):
            take(a, "x", idx)


class TestTakeAlongAxis:
    def test_matches_jnp_reference(self, x4, y8):
        a = array(jnp.array([[3.0, 1.0, 2.0], [6.0, 5.0, 4.0]]), (Axis(2, "batch"), Axis(3, "t")))
        order = array(jnp.argsort(a.array, axis=1), (Axis(2, "batch"), Axis(3, "t")))

        res = take_along_axis(a, order, axis="t")

        assert jnp.array_equal(res.array, jnp.take_along_axis(a.array, order.array, axis=1))
        assert res.axes == a.axes

    def test_axis_name_mismatch_at_a_shared_position_raises(self):
        a = array(jnp.ones((2, 3)), (Axis(2, "batch"), Axis(3, "t")))
        bad_indices = array(jnp.zeros((2, 3), dtype=jnp.int32), (Axis(2, "not_batch"), Axis(3, "t")))
        with pytest.raises(ValueError, match="names disagree"):
            take_along_axis(a, bad_indices, axis="t")

    def test_gathered_axis_may_resize(self):
        a = array(jnp.arange(6.0).reshape(2, 3), (Axis(2, "batch"), Axis(3, "t")))
        indices = array(jnp.zeros((2, 1), dtype=jnp.int32), (Axis(2, "batch"), Axis(1, "t")))
        res = take_along_axis(a, indices, axis="t")
        assert res.axes == (Axis(2, "batch"), Axis(1, "t"))


@st.composite
def _basic_or_advanced_3d_key(draw):
    """A random 3-axis array plus a key that mixes an advanced (array) index
    on one axis with plain slices/ints elsewhere -- exercising both the
    contiguous and non-contiguous placement branches across random shapes."""
    sizes = draw(st.lists(st.integers(min_value=1, max_value=6), min_size=3, max_size=3))
    axes = tuple(Axis(s, n) for s, n in zip(sizes, ("a", "b", "c")))
    adv_position = draw(st.integers(min_value=0, max_value=2))
    idx_len = draw(st.integers(min_value=1, max_value=4))
    idx_values = draw(
        st.lists(st.integers(min_value=0, max_value=sizes[adv_position] - 1), min_size=idx_len, max_size=idx_len)
    )

    key = []
    for i in range(3):
        if i == adv_position:
            key.append(jnp.array(idx_values))
        else:
            key.append(draw(st.sampled_from([slice(None), 0])))
    return axes, tuple(key)


class TestAdvancedIndexingShapeAgreement:
    @given(_basic_or_advanced_3d_key())
    def test_output_axes_shape_matches_real_array_indexing(self, case):
        axes, key = case
        data = jnp.arange(int(np.prod([ax.size for ax in axes]))).reshape(tuple(ax.size for ax in axes))
        a = array(data, axes)

        result = a[key]
        real = data[key]

        assert result.array.shape == real.shape
        assert tuple(ax.size for ax in result.axes) == real.shape
