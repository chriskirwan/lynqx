from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import pytest
from lynqx._src.array import NamedArrayImpl
from lynqx._src.axis import Axis
from lynqx._src.named import reductions
from lynqx._src.named.constructors import array


# Fixed axis shape shared by every table-driven test below, so a failure
# always reports against the same, easy-to-reason-about geometry: a
# 3x4x5 array over axes "batch", "x", "y".
AXIS_SHAPE = (("batch", 3), ("x", 4), ("y", 5))


@pytest.fixture
def batch3():
    return Axis(3, "batch")


@pytest.fixture
def x4():
    return Axis(4, "x")


@pytest.fixture
def y5():
    return Axis(5, "y")


def _shape():
    return tuple(size for _, size in AXIS_SHAPE)


def _axes():
    return tuple(Axis(size, name) for name, size in AXIS_SHAPE)


def _make_named(*, bool_input: bool = False) -> NamedArrayImpl:
    shape = _shape()
    if bool_input:
        data = (jnp.arange(int(np.prod(shape))).reshape(shape) % 3) == 0
    else:
        data = jnp.arange(1, int(np.prod(shape)) + 1, dtype=jnp.float32).reshape(shape)
    return NamedArrayImpl(data, _axes())


def _make_named_with_nan() -> NamedArrayImpl:
    shape = _shape()
    data = jnp.arange(1, int(np.prod(shape)) + 1, dtype=jnp.float32).reshape(shape)
    data = data.at[0, 0, 0].set(jnp.nan)
    return NamedArrayImpl(data, _axes())


@dataclass(frozen=True)
class ReductionRecord:
    """One row of the reduction op table: everything needed to test a single
    `reductions` function generically against its `jnp` reference."""

    name: str  # attribute shared by `reductions` and `jnp`
    supports_initial: bool = False
    bool_input: bool = False
    supports_multi_axis: bool = True


REDUCTION_RECORDS = [
    ReductionRecord("any", bool_input=True),
    ReductionRecord("all", bool_input=True),
    ReductionRecord("sum", supports_initial=True),
    ReductionRecord("prod", supports_initial=True),
    ReductionRecord("mean"),
    ReductionRecord("std"),
    ReductionRecord("var"),
    ReductionRecord("min", supports_initial=True),
    ReductionRecord("max", supports_initial=True),
    ReductionRecord("amin", supports_initial=True),
    ReductionRecord("amax", supports_initial=True),
    ReductionRecord("ptp"),
    ReductionRecord("nanmax", supports_initial=True),
    ReductionRecord("nanmin", supports_initial=True),
    ReductionRecord("nanprod"),
    ReductionRecord("nanstd"),
    ReductionRecord("nansum"),
    ReductionRecord("nanvar"),
]

# `argmin`/`argmax`/`nanargmin`/`nanargmax` only accept a single axis (or
# `None`) at the `jnp` level, so they're tabulated separately from the
# multi-axis-capable reductions above.
ARG_RECORDS = [
    ReductionRecord("argmin", supports_multi_axis=False),
    ReductionRecord("argmax", supports_multi_axis=False),
    ReductionRecord("nanargmax", supports_multi_axis=False),
    ReductionRecord("nanargmin", supports_multi_axis=False),
]

ALL_RECORDS = REDUCTION_RECORDS + ARG_RECORDS

NAN_FUNCS = ["nansum", "nanprod", "nanstd", "nanvar", "nanmax", "nanmin", "nanargmax", "nanargmin"]

AXISWISE_RECORDS = ["cumsum", "cumprod", "nancumsum", "nancumprod"]


class TestReductionsMatchReference:
    @pytest.mark.parametrize("record", ALL_RECORDS, ids=lambda r: r.name)
    def test_full_reduction_over_all_axes(self, record):
        lynqx_fn = getattr(reductions, record.name)
        ref_fn = getattr(jnp, record.name)
        x = _make_named(bool_input=record.bool_input)

        got = lynqx_fn(x, axis=None)
        want = ref_fn(x.array, axis=None)

        assert got.axes == ()
        np.testing.assert_allclose(got.array, want)

    @pytest.mark.parametrize("record", [r for r in ALL_RECORDS], ids=lambda r: r.name)
    def test_single_named_axis_reduction(self, record):
        lynqx_fn = getattr(reductions, record.name)
        ref_fn = getattr(jnp, record.name)
        x = _make_named(bool_input=record.bool_input)

        got = lynqx_fn(x, axis="x")
        want = ref_fn(x.array, axis=1)

        assert [ax.name for ax in got.axes] == ["batch", "y"]
        np.testing.assert_allclose(got.array, want)

    @pytest.mark.parametrize("record", [r for r in REDUCTION_RECORDS if r.supports_multi_axis], ids=lambda r: r.name)
    def test_multi_axis_reduction_by_name(self, record):
        lynqx_fn = getattr(reductions, record.name)
        ref_fn = getattr(jnp, record.name)
        x = _make_named(bool_input=record.bool_input)

        got = lynqx_fn(x, axis=["batch", "y"])
        want = ref_fn(x.array, axis=(0, 2))

        assert [ax.name for ax in got.axes] == ["x"]
        np.testing.assert_allclose(got.array, want)

    @pytest.mark.parametrize("record", [r for r in ALL_RECORDS], ids=lambda r: r.name)
    def test_keepdims_resizes_reduced_axis_to_one(self, record):
        lynqx_fn = getattr(reductions, record.name)
        x = _make_named(bool_input=record.bool_input)

        got = lynqx_fn(x, axis="x", keepdims=True)

        assert [ax.name for ax in got.axes] == ["batch", "x", "y"]
        assert [ax.size for ax in got.axes] == [3, 1, 5]

    @pytest.mark.parametrize("record", ALL_RECORDS, ids=lambda r: r.name)
    def test_keepdims_with_full_reduction(self, record):
        lynqx_fn = getattr(reductions, record.name)
        x = _make_named(bool_input=record.bool_input)

        got = lynqx_fn(x, axis=None, keepdims=True)

        assert [ax.size for ax in got.axes] == [1, 1, 1]
        assert [ax.name for ax in got.axes] == ["batch", "x", "y"]


class TestNanVariantsIgnoreNaNs:
    """`nan*` functions should skip NaNs rather than propagating them, and
    otherwise agree with their `jnp` counterpart bit-for-bit."""

    @pytest.mark.parametrize("name", NAN_FUNCS)
    def test_matches_reference_with_nans_present(self, name):
        lynqx_fn = getattr(reductions, name)
        ref_fn = getattr(jnp, name)
        x = _make_named_with_nan()

        got = lynqx_fn(x, axis="x")
        want = ref_fn(x.array, axis=1)

        np.testing.assert_allclose(got.array, want)


class TestWhereBroadcasting:
    def test_where_mask_on_subset_axis_is_broadcast_before_reducing(self, x4):
        x = _make_named()
        mask = array(jnp.array([True, False, True, False]), (x4,))

        got = reductions.sum(x, axis="x", where=mask)
        want = jnp.sum(x.array, axis=1, where=mask.array[None, :, None])

        np.testing.assert_allclose(got.array, want)

    def test_where_mask_supported_by_any_and_all(self, x4):
        x = _make_named(bool_input=True)
        mask = array(jnp.array([True, True, False, False]), (x4,))

        got = reductions.any(x, axis="x", where=mask)
        want = jnp.any(x.array, axis=1, where=mask.array[None, :, None])

        assert jnp.array_equal(got.array, want)


class TestInitialValue:
    def test_sum_initial_is_added_to_the_reduction(self):
        x = _make_named()

        got = reductions.sum(x, axis="x", initial=100.0)
        want = jnp.sum(x.array, axis=1, initial=100.0)

        np.testing.assert_allclose(got.array, want)

    def test_max_initial_lower_bounds_the_result(self):
        x = _make_named()

        got = reductions.max(x, axis="x", initial=1000.0)
        want = jnp.max(x.array, axis=1, initial=1000.0)

        np.testing.assert_allclose(got.array, want)


class TestReductionErrorHandling:
    def test_unresolved_axis_name_raises(self):
        # `axis_indices` is called with its default `strict=True`, so it raises
        # its own "could not be resolved" error before `_reduce_one_leaf`'s own
        # (otherwise unreachable) "is not in" check would ever fire.
        x = _make_named()
        with pytest.raises(ValueError, match="could not be resolved"):
            reductions.sum(x, axis="nope")

    def test_nanargmax_multiple_axes_raises_single_axis_only(self):
        x = _make_named()
        with pytest.raises(ValueError, match="single axis"):
            reductions.nanargmax(x, axis=["batch", "y"])

    def test_nanargmin_multiple_axes_raises_single_axis_only(self):
        x = _make_named()
        with pytest.raises(ValueError, match="single axis"):
            reductions.nanargmin(x, axis=["batch", "y"])


class TestWrapReductionCallRejectsNonNamedArrayPytrees:
    """`wrap_reduction_call` calls `util.ensure_named` on the raw top-level
    input *before* the `jax.tree.map` call, so -- despite the `map` call --
    a plain pytree of `NamedArray`s (e.g. a dict) is not actually supported;
    it's rejected the same way a bare non-array/scalar input would be."""

    def test_dict_of_named_arrays_raises_type_error(self):
        x = _make_named()
        tree = {"a": x, "b": x}

        with pytest.raises(TypeError, match="NamedArray"):
            reductions.sum(tree, axis="x")


class TestCumulativeOps:
    @pytest.mark.parametrize("name", AXISWISE_RECORDS)
    def test_matches_reference_and_preserves_axes(self, name):
        lynqx_fn = getattr(reductions, name)
        ref_fn = getattr(jnp, name)
        x = _make_named()

        got = lynqx_fn(x, axis="x")
        want = ref_fn(x.array, axis=1)

        assert got.axes == x.axes
        np.testing.assert_allclose(got.array, want)

    @pytest.mark.parametrize("name", AXISWISE_RECORDS)
    def test_multiple_axes_raises_single_axis_only(self, name):
        lynqx_fn = getattr(reductions, name)
        x = _make_named()

        with pytest.raises(ValueError, match="single axis"):
            lynqx_fn(x, axis=["batch", "y"])

    @pytest.mark.parametrize("name", AXISWISE_RECORDS)
    def test_unresolved_axis_raises(self, name):
        lynqx_fn = getattr(reductions, name)
        x = _make_named()

        with pytest.raises(ValueError, match="could not be resolved"):
            lynqx_fn(x, axis="nope")


class TestSortingOps:
    def test_sort_matches_reference_and_preserves_axes(self):
        x = _make_named()

        got = reductions.sort(x, axis="x")
        want = jnp.sort(x.array, axis=1)

        assert got.axes == x.axes
        np.testing.assert_allclose(got.array, want)

    def test_argsort_matches_reference_and_preserves_axes(self):
        x = _make_named()

        got = reductions.argsort(x, axis="x")
        want = jnp.argsort(x.array, axis=1)

        assert got.axes == x.axes
        assert jnp.array_equal(got.array, want)

    def test_sort_multiple_axes_raises_single_axis_only(self):
        x = _make_named()
        with pytest.raises(ValueError, match="single axis"):
            reductions.sort(x, axis=["batch", "y"])

    def test_partition_matches_reference_and_preserves_axes(self):
        x = _make_named()

        got = reductions.partition(x, kth=2, axis="x")
        want = jnp.partition(x.array, kth=2, axis=1)

        assert got.axes == x.axes
        np.testing.assert_allclose(got.array, want)

    def test_argpartition_matches_reference_and_preserves_axes(self):
        x = _make_named()

        got = reductions.argpartition(x, kth=2, axis="x")
        want = jnp.argpartition(x.array, kth=2, axis=1)

        assert got.axes == x.axes
        assert jnp.array_equal(got.array, want)


class TestWrapReductionCallDirectly:
    """Exercises validation branches of `wrap_reduction_call` that aren't
    reachable through any of the public `reductions` functions (none of them
    expose `out`, and every function that exposes `where`/`initial` already
    supports it)."""

    def test_out_kwarg_raises(self):
        x = _make_named()
        wrapped = reductions.wrap_reduction_call(jnp.sum)

        with pytest.raises(ValueError, match="out is not supported"):
            wrapped(x, axis="x", out=jnp.zeros((3, 5)))

    def test_where_raises_when_unsupported(self, x4):
        x = _make_named()
        mask = array(jnp.ones((4,), dtype=bool), (x4,))
        wrapped = reductions.wrap_reduction_call(jnp.sum, supports_where=False)

        with pytest.raises(ValueError, match="where is not supported"):
            wrapped(x, axis="x", where=mask)

    def test_initial_raises_when_unsupported(self):
        x = _make_named()
        wrapped = reductions.wrap_reduction_call(jnp.sum, supports_initial=False)

        with pytest.raises(ValueError, match="initial is not supported"):
            wrapped(x, axis="x", initial=1.0)

    def test_single_axis_only_restriction(self):
        x = _make_named()
        wrapped = reductions.wrap_reduction_call(jnp.sum, single_axis_only=True)

        with pytest.raises(ValueError, match="single axis"):
            wrapped(x, axis=["batch", "y"])


class TestWrapAxiswiseCallDirectly:
    def test_axis_none_passes_through_to_the_wrapped_function(self, x4):
        a = array(jnp.arange(4.0), (x4,))
        wrapped = reductions.wrap_axiswise_call(jnp.sort)

        got = wrapped(a, axis=None)

        assert got.axes == a.axes
        assert jnp.array_equal(got.array, jnp.sort(a.array, axis=None))

    def test_single_axis_only_restriction(self):
        x = _make_named()
        wrapped = reductions.wrap_axiswise_call(jnp.cumsum, single_axis_only=True)

        with pytest.raises(ValueError, match="single axis"):
            wrapped(x, axis=["batch", "y"])

    def test_unresolved_axis_raises(self):
        x = _make_named()
        wrapped = reductions.wrap_axiswise_call(jnp.cumsum)

        with pytest.raises(ValueError, match="could not be resolved"):
            wrapped(x, axis="nope")
