import jax
import jax.numpy as jnp
import jax.test_util as jtu
import numpy as np
import pytest
from hypothesis import given, strategies as st
from jax import P
from lynqx._src.axis import Axis
from lynqx._src.named.constructors import array
from lynqx._src.named.tensor_contractions import (
    dot,
    matmul,
    outer,
    trace,
    vecdot,
)
from lynqx._src.typing import PM


@pytest.fixture
def x4():
    return Axis(4, "x")


@pytest.fixture
def y8():
    return Axis(8, "y")


@pytest.fixture
def z16():
    return Axis(16, "z")


def _rand(seed, shape, dtype=jnp.float32):
    return jax.random.normal(jax.random.key(seed), shape, dtype=dtype)


def _named_rand(seed, *axes):
    return array(_rand(seed, tuple(ax.size for ax in axes)), axes)


def _assert_close(got, want, tol=1e-5):
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=tol, atol=tol)


# ---------------------------------------------------------------------------
# dot
# ---------------------------------------------------------------------------


class TestDot:
    def test_matrix_product_matches_jnp(self):
        i, j, k = Axis(3, "i"), Axis(4, "j"), Axis(5, "k")
        a = _named_rand(0, i, j)
        b = _named_rand(1, j, k)

        res = dot(a, b, ((("j",), ("j",)), ((), ())))

        assert res.axes == (i, k)
        _assert_close(res.array, a.array @ b.array)

    def test_contracted_axes_may_have_different_names(self):
        i, j, l, k = Axis(3, "i"), Axis(4, "j"), Axis(4, "l"), Axis(5, "k")
        a = _named_rand(0, i, j)
        b = _named_rand(1, l, k)

        res = dot(a, b, ((("j",), ("l",)), ((), ())))

        assert res.axes == (i, k)
        _assert_close(res.array, a.array @ b.array)

    def test_batch_axes_come_first_then_free_lhs_then_free_rhs(self):
        b_ax, i, j, k = Axis(2, "b"), Axis(3, "i"), Axis(4, "j"), Axis(5, "k")
        # deliberately scramble the input layouts
        a = _named_rand(0, i, b_ax, j)
        c = _named_rand(1, j, k, b_ax)

        res = dot(a, c, ((("j",), ("j",)), (("b",), ("b",))))

        assert res.axes == (b_ax, i, k)
        _assert_close(res.array, jnp.einsum("ibj,jkb->bik", a.array, c.array))

    def test_multiple_contracting_axes(self):
        i, j, k, l = Axis(2, "i"), Axis(3, "j"), Axis(4, "k"), Axis(5, "l")
        a = _named_rand(0, i, j, k)
        b = _named_rand(1, j, k, l)

        res = dot(a, b, ((("j", "k"), ("j", "k")), ((), ())))

        assert res.axes == (i, l)
        _assert_close(res.array, jnp.einsum("ijk,jkl->il", a.array, b.array))

    def test_integer_selectors_resolve_by_position(self):
        i, j, k = Axis(3, "i"), Axis(4, "j"), Axis(5, "k")
        a = _named_rand(0, i, j)
        b = _named_rand(1, j, k)

        res = dot(a, b, (((1,), (0,)), ((), ())))

        assert res.axes == (i, k)
        _assert_close(res.array, a.array @ b.array)

    def test_negative_integer_selectors(self):
        i, j, k = Axis(3, "i"), Axis(4, "j"), Axis(5, "k")
        a = _named_rand(0, i, j)
        b = _named_rand(1, j, k)

        res = dot(a, b, (((-1,), (-2,)), ((), ())))

        assert res.axes == (i, k)

    def test_anonymous_axes_can_be_contracted_by_position(self):
        a = array(_rand(0, (3, 4)), (Axis(3), Axis(4)))
        b = array(_rand(1, (4, 5)), (Axis(4), Axis(5)))

        res = dot(a, b, (((1,), (0,)), ((), ())))

        assert res.axes == (Axis(3), Axis(5))
        _assert_close(res.array, a.array @ b.array)

    def test_list_specs_are_accepted_like_tuple_specs(self):
        # The `spec` annotation is Sequence-based, so lists must work too, even
        # though the jit static-arg cache key needs to be hashable.
        i, j, k = Axis(3, "i"), Axis(4, "j"), Axis(5, "k")
        a = _named_rand(0, i, j)
        b = _named_rand(1, j, k)

        res = dot(a, b, ((["j"], ["j"]), ([], [])))

        assert res.axes == (i, k)
        _assert_close(res.array, a.array @ b.array)

    def test_scalar_operand_with_empty_spec_scales_the_array(self, x4):
        a = _named_rand(0, x4)

        res = dot(2.0, a, (((), ()), ((), ())))

        assert res.axes == (x4,)
        _assert_close(res.array, 2.0 * a.array)

    def test_preferred_element_type_controls_output_dtype(self):
        i, j, k = Axis(3, "i"), Axis(4, "j"), Axis(5, "k")
        a = array(jnp.ones((3, 4), dtype=jnp.int8), (i, j))
        b = array(jnp.ones((4, 5), dtype=jnp.int8), (j, k))

        res = dot(a, b, ((("j",), ("j",)), ((), ())), preferred_element_type=jnp.int32)

        assert res.dtype == jnp.int32
        assert jnp.array_equal(res.array, jnp.full((3, 5), 4, dtype=jnp.int32))

    def test_precision_is_forwarded(self):
        i, j, k = Axis(3, "i"), Axis(4, "j"), Axis(5, "k")
        a = _named_rand(0, i, j)
        b = _named_rand(1, j, k)

        res = dot(a, b, ((("j",), ("j",)), ((), ())), precision=jax.lax.Precision.HIGHEST)

        _assert_close(res.array, jnp.matmul(a.array, b.array, precision=jax.lax.Precision.HIGHEST))

    # -- validation ---------------------------------------------------------

    def test_contract_length_mismatch_raises(self, x4, y8):
        a = _named_rand(0, x4, y8)
        with pytest.raises(ValueError, match="contracting axes must be the same length"):
            dot(a, a, ((("x", "y"), ("x",)), ((), ())))

    def test_batch_length_mismatch_raises(self, x4, y8):
        a = _named_rand(0, x4, y8)
        with pytest.raises(ValueError, match="batch axes must be the same length"):
            dot(a, a, (((), ()), (("x",), ())))

    def test_contract_size_mismatch_raises(self, x4, y8):
        a = _named_rand(0, x4)
        b = _named_rand(1, y8)
        with pytest.raises(ValueError, match="Contracting axis size mismatch"):
            dot(a, b, ((("x",), ("y",)), ((), ())))

    def test_batch_size_mismatch_raises(self):
        a = _named_rand(0, Axis(4, "b"), Axis(3, "i"))
        b = _named_rand(1, Axis(8, "b"), Axis(3, "i"))
        with pytest.raises(ValueError, match="Batch axis size mismatch"):
            dot(a, b, (((), ()), (("b",), ("b",))))

    def test_batch_name_mismatch_raises(self):
        a = _named_rand(0, Axis(4, "b"))
        b = _named_rand(1, Axis(4, "c"))
        with pytest.raises(ValueError, match="Batch axis names must match"):
            dot(a, b, (((), ()), (("b",), ("c",))))

    def test_axis_in_both_contract_and_batch_raises(self, x4):
        a = _named_rand(0, x4)
        b = array(_rand(1, (4,)), (Axis(4),))
        with pytest.raises(ValueError, match="both contracting and batch"):
            dot(a, b, ((("x",), (0,)), (("x",), (0,))))

    def test_output_name_collision_raises(self):
        i, j = Axis(3, "i"), Axis(4, "j")
        a = _named_rand(0, i, j)
        b = _named_rand(1, j, i)
        with pytest.raises(ValueError, match="Output axis name collision"):
            dot(a, b, ((("j",), ("j",)), ((), ())))

    def test_unresolved_selector_raises(self, x4):
        a = _named_rand(0, x4)
        with pytest.raises(ValueError, match="could not be resolved"):
            dot(a, a, ((("nope",), ("x",)), ((), ())))

    def test_non_namedarray_operand_raises_type_error(self, x4):
        a = _named_rand(0, x4)
        with pytest.raises(TypeError, match="dot"):
            dot(a, object(), (((), ()), ((), ())))

    # -- JAX composability --------------------------------------------------

    def test_composes_with_outer_jit(self):
        i, j, k = Axis(3, "i"), Axis(4, "j"), Axis(5, "k")
        a = _named_rand(0, i, j)
        b = _named_rand(1, j, k)

        @jax.jit
        def f(a, b):
            return dot(a, b, ((("j",), ("j",)), ((), ())))

        res = f(a, b)

        assert res.axes == (i, k)
        _assert_close(res.array, a.array @ b.array)

    def test_composes_with_vmap(self):
        i, j, k = Axis(3, "i"), Axis(4, "j"), Axis(5, "k")
        a = _named_rand(0, i, j)
        bs = _rand(1, (6, 4, 5))

        def f(b):
            return dot(a, array(b, (j, k)), ((("j",), ("j",)), ((), ()))).array

        _assert_close(jax.vmap(f)(bs), jnp.einsum("ij,njk->nik", a.array, bs))

    def test_gradients_match_numerical_gradients(self):
        i, j, k = Axis(3, "i"), Axis(4, "j"), Axis(5, "k")
        a = _rand(0, (3, 4))
        b = _rand(1, (4, 5))

        def f(a, b):
            return dot(array(a, (i, j)), array(b, (j, k)), ((("j",), ("j",)), ((), ()))).array.sum()

        jtu.check_grads(f, (a, b), order=1, modes=["rev"], atol=1e-2, rtol=1e-2)

    def test_pm_out_sharding_resolves_inside_a_mesh(self, mesh_1d):
        n = len(jax.devices())
        b_ax, i, j = Axis(n, "batch"), Axis(3, "i"), Axis(4, "j")
        a = _named_rand(0, b_ax, i, j)
        c = _named_rand(1, b_ax, j)
        pm = PM({"batch": "x"})

        with jax.set_mesh(mesh_1d):
            res = dot(a, c, ((("j",), ("j",)), (("batch",), ("batch",))), out_sharding=pm)

        assert res.axes == (b_ax, i)
        _assert_close(res.array, jnp.einsum("bij,bj->bi", a.array, c.array))

    def test_partition_spec_out_sharding_accepted(self, mesh_1d):
        n = len(jax.devices())
        b_ax, j = Axis(n, "batch"), Axis(4, "j")
        a = _named_rand(0, b_ax, j)
        c = _named_rand(1, b_ax, j)

        with jax.set_mesh(mesh_1d):
            res = dot(a, c, ((("j",), ("j",)), (("batch",), ("batch",))), out_sharding=P("x"))

        assert res.axes == (b_ax,)

    @given(
        st.integers(min_value=1, max_value=6),
        st.integers(min_value=1, max_value=6),
        st.integers(min_value=1, max_value=6),
    )
    def test_matches_matmul_for_random_sizes(self, m, n, p):
        a = _named_rand(0, Axis(m, "i"), Axis(n, "j"))
        b = _named_rand(1, Axis(n, "j"), Axis(p, "k"))

        res = dot(a, b, ((("j",), ("j",)), ((), ())))

        assert res.array.shape == (m, p)
        _assert_close(res.array, a.array @ b.array, tol=1e-4)


# ---------------------------------------------------------------------------
# vecdot
# ---------------------------------------------------------------------------


class TestVecdot:
    def test_one_dimensional_inputs_give_a_scalar(self, x4):
        a = _named_rand(0, x4)
        b = _named_rand(1, x4)

        res = vecdot(a, b, "x")

        assert res.axes == ()
        _assert_close(res.array, jnp.vdot(a.array, b.array))

    def test_shared_named_axis_is_inferred_as_batch(self):
        b_ax, j = Axis(3, "b"), Axis(4, "j")
        a = _named_rand(0, b_ax, j)
        c = _named_rand(1, b_ax, j)

        res = vecdot(a, c, "j")

        assert res.axes == (b_ax,)
        _assert_close(res.array, jnp.sum(a.array * c.array, axis=-1))

    def test_batch_inference_is_independent_of_axis_order(self):
        b_ax, j = Axis(3, "b"), Axis(4, "j")
        a = _named_rand(0, b_ax, j)
        c = _named_rand(1, j, b_ax)

        res = vecdot(a, c, "j")

        assert res.axes == (b_ax,)
        _assert_close(res.array, jnp.einsum("bj,jb->b", a.array, c.array))

    def test_unshared_named_axes_stay_free(self):
        i, j, k = Axis(3, "i"), Axis(4, "j"), Axis(5, "k")
        a = _named_rand(0, i, j)
        b = _named_rand(1, j, k)

        res = vecdot(a, b, "j")

        assert res.axes == (i, k)
        _assert_close(res.array, a.array @ b.array)

    def test_tuple_axis_contracts_different_names(self):
        i, j, l = Axis(3, "i"), Axis(4, "j"), Axis(4, "l")
        a = _named_rand(0, i, j)
        b = _named_rand(1, l)

        res = vecdot(a, b, ("j", "l"))

        assert res.axes == (i,)
        _assert_close(res.array, a.array @ b.array)

    def test_anonymous_trailing_axes_are_batched_positionally(self):
        j = Axis(4, "j")
        a = array(_rand(0, (3, 4)), (Axis(3), j))
        b = array(_rand(1, (3, 4)), (Axis(3), j))

        res = vecdot(a, b, "j")

        assert res.axes == (Axis(3),)
        _assert_close(res.array, jnp.sum(a.array * b.array, axis=-1))

    def test_multiple_shared_axes_are_all_batched(self):
        b1, b2, j = Axis(2, "b1"), Axis(3, "b2"), Axis(4, "j")
        a = _named_rand(0, b1, b2, j)
        c = _named_rand(1, b2, b1, j)

        res = vecdot(a, c, "j")

        assert set(ax.name for ax in res.axes) == {"b1", "b2"}
        assert res.axes == (b1, b2)
        _assert_close(res.array, jnp.einsum("abj,baj->ab", a.array, c.array))

    def test_missing_axis_raises(self, x4):
        a = _named_rand(0, x4)
        with pytest.raises(ValueError, match="could not be resolved"):
            vecdot(a, a, "nope")

    def test_contract_size_mismatch_raises(self, x4, y8):
        a = _named_rand(0, x4)
        b = _named_rand(1, y8)
        with pytest.raises(ValueError, match="Contracting axis size mismatch"):
            vecdot(a, b, ("x", "y"))

    def test_composes_with_jit(self):
        b_ax, j = Axis(3, "b"), Axis(4, "j")
        a = _named_rand(0, b_ax, j)
        c = _named_rand(1, b_ax, j)

        res = jax.jit(lambda a, c: vecdot(a, c, "j"))(a, c)

        _assert_close(res.array, jnp.sum(a.array * c.array, axis=-1))

    def test_gradient_matches_numerical_gradient(self):
        b_ax, j = Axis(3, "b"), Axis(4, "j")

        def f(a, c):
            return vecdot(array(a, (b_ax, j)), array(c, (b_ax, j)), "j").array.sum()

        jtu.check_grads(f, (_rand(0, (3, 4)), _rand(1, (3, 4))), order=1, modes=["rev"], atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# matmul
# ---------------------------------------------------------------------------


class TestMatmul:
    def test_two_dimensional_matches_jnp_matmul(self):
        i, j, k = Axis(3, "i"), Axis(4, "j"), Axis(5, "k")
        a = _named_rand(0, i, j)
        b = _named_rand(1, j, k)

        res = matmul(a, b, (("j",), ("j",)))

        assert res.axes == (i, k)
        _assert_close(res.array, jnp.matmul(a.array, b.array))

    def test_shared_named_axis_is_batched(self):
        b_ax, i, j, k = Axis(2, "b"), Axis(3, "i"), Axis(4, "j"), Axis(5, "k")
        a = _named_rand(0, b_ax, i, j)
        c = _named_rand(1, b_ax, j, k)

        res = matmul(a, c, (("j",), ("j",)))

        assert res.axes == (b_ax, i, k)
        _assert_close(res.array, jnp.matmul(a.array, c.array))

    def test_bare_string_contract_form(self):
        i, j, k = Axis(3, "i"), Axis(4, "j"), Axis(5, "k")
        a = _named_rand(0, i, j)
        b = _named_rand(1, j, k)

        res = matmul(a, b, ("j", "j"))

        assert res.axes == (i, k)

    def test_contract_different_names(self):
        i, j, l, k = Axis(3, "i"), Axis(4, "j"), Axis(4, "l"), Axis(5, "k")
        a = _named_rand(0, i, j)
        b = _named_rand(1, l, k)

        res = matmul(a, b, (("j",), ("l",)))

        assert res.axes == (i, k)
        _assert_close(res.array, a.array @ b.array)

    def test_matrix_vector(self):
        i, j = Axis(3, "i"), Axis(4, "j")
        a = _named_rand(0, i, j)
        v = _named_rand(1, j)

        res = matmul(a, v, (("j",), ("j",)))

        assert res.axes == (i,)
        _assert_close(res.array, a.array @ v.array)

    def test_shared_free_name_becomes_batch_instead_of_colliding(self):
        i, j = Axis(3, "i"), Axis(4, "j")
        a = _named_rand(0, i, j)
        b = _named_rand(1, j, i)
        # "i" is shared, so it is inferred as a batch axis and must NOT collide.
        res = matmul(a, b, (("j",), ("j",)))
        assert res.axes == (i,)
        _assert_close(res.array, jnp.einsum("ij,ji->i", a.array, b.array))

    def test_shared_name_with_different_sizes_raises(self):
        a = _named_rand(0, Axis(3, "b"), Axis(4, "j"))
        b = _named_rand(1, Axis(5, "b"), Axis(4, "j"))
        with pytest.raises(ValueError, match="Batch axis size mismatch"):
            matmul(a, b, (("j",), ("j",)))

    def test_missing_contract_axis_raises(self, x4):
        a = _named_rand(0, x4)
        with pytest.raises(ValueError, match="could not be resolved"):
            matmul(a, a, (("nope",), ("x",)))

    def test_composes_with_jit(self):
        i, j, k = Axis(3, "i"), Axis(4, "j"), Axis(5, "k")
        a = _named_rand(0, i, j)
        b = _named_rand(1, j, k)

        res = jax.jit(lambda a, b: matmul(a, b, (("j",), ("j",))))(a, b)

        _assert_close(res.array, a.array @ b.array)


# ---------------------------------------------------------------------------
# outer
# ---------------------------------------------------------------------------


class TestOuter:
    def test_vectors_match_jnp_outer(self, x4, y8):
        a = _named_rand(0, x4)
        b = _named_rand(1, y8)

        res = outer(a, b)

        assert res.axes == (x4, y8)
        _assert_close(res.array, jnp.outer(a.array, b.array))

    def test_higher_rank_concatenates_axes_in_order(self, x4, y8, z16):
        a = _named_rand(0, x4, y8)
        b = _named_rand(1, z16)

        res = outer(a, b)

        assert res.axes == (x4, y8, z16)
        _assert_close(res.array, jnp.einsum("ij,k->ijk", a.array, b.array))

    def test_anonymous_axes_never_collide(self):
        a = array(_rand(0, (3,)), (Axis(3),))
        b = array(_rand(1, (3,)), (Axis(3),))

        res = outer(a, b)

        assert res.axes == (Axis(3), Axis(3))
        _assert_close(res.array, jnp.outer(a.array, b.array))

    def test_scalar_operand_scales(self, x4):
        a = _named_rand(0, x4)

        res = outer(3.0, a)

        assert res.axes == (x4,)
        _assert_close(res.array, 3.0 * a.array)

    def test_shared_name_raises(self, x4):
        a = _named_rand(0, x4)
        with pytest.raises(ValueError, match="Output axis name collision"):
            outer(a, a)

    def test_preferred_element_type(self, x4, y8):
        a = array(jnp.ones((4,), dtype=jnp.int8), (x4,))
        b = array(jnp.ones((8,), dtype=jnp.int8), (y8,))

        res = outer(a, b, preferred_element_type=jnp.int32)

        assert res.dtype == jnp.int32

    def test_composes_with_jit(self, x4, y8):
        a = _named_rand(0, x4)
        b = _named_rand(1, y8)

        res = jax.jit(outer)(a, b)

        assert res.axes == (x4, y8)
        _assert_close(res.array, jnp.outer(a.array, b.array))

    def test_gradient_matches_numerical_gradient(self, x4, y8):
        def f(a, b):
            return outer(array(a, (x4,)), array(b, (y8,))).array.sum()

        jtu.check_grads(f, (_rand(0, (4,)), _rand(1, (8,))), order=1, modes=["rev"], atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# trace
# ---------------------------------------------------------------------------


class TestTrace:
    def test_default_axes_match_jnp_trace(self, x4, y8, z16):
        a = _named_rand(0, x4, x4.rename("w"), z16)

        res = trace(a)

        assert res.axes == (z16,)
        _assert_close(res.array, jnp.trace(a.array))

    def test_two_dimensional_gives_scalar(self):
        a = _named_rand(0, Axis(4, "i"), Axis(4, "j"))

        res = trace(a)

        assert res.axes == ()
        _assert_close(res.array, jnp.trace(a.array))

    def test_axes_selected_by_name(self, x4, y8):
        z = Axis(4, "z")
        a = _named_rand(0, x4, y8, z)

        res = trace(a, axis1="x", axis2="z")

        assert res.axes == (y8,)
        _assert_close(res.array, jnp.trace(a.array, axis1=0, axis2=2))

    def test_axis_order_selects_which_is_the_row_axis(self):
        # With a non-zero offset, swapping axis1/axis2 changes the diagonal.
        a = _named_rand(0, Axis(4, "i"), Axis(4, "j"))

        r1 = trace(a, offset=1, axis1="i", axis2="j")
        r2 = trace(a, offset=1, axis1="j", axis2="i")

        _assert_close(r1.array, jnp.trace(a.array, offset=1, axis1=0, axis2=1))
        _assert_close(r2.array, jnp.trace(a.array, offset=1, axis1=1, axis2=0))

    @pytest.mark.parametrize("offset", [-2, -1, 0, 1, 2])
    def test_offset_matches_jnp(self, offset):
        a = _named_rand(0, Axis(5, "i"), Axis(5, "j"))

        res = trace(a, offset=offset, axis1="i", axis2="j")

        _assert_close(res.array, jnp.trace(a.array, offset=offset))

    def test_rectangular_axes(self):
        a = _named_rand(0, Axis(3, "i"), Axis(5, "j"), Axis(2, "k"))

        res = trace(a, axis1="i", axis2="j")

        assert res.axes == (Axis(2, "k"),)
        _assert_close(res.array, jnp.trace(a.array, axis1=0, axis2=1))

    def test_negative_integer_selectors(self, z16):
        a = _named_rand(0, z16, Axis(4, "i"), Axis(4, "j"))

        res = trace(a, axis1=-2, axis2=-1)

        assert res.axes == (z16,)
        _assert_close(res.array, jnp.trace(a.array, axis1=-2, axis2=-1))

    def test_anonymous_axes_by_position(self):
        a = array(_rand(0, (4, 4, 3)), (Axis(4), Axis(4), Axis(3)))

        res = trace(a, axis1=0, axis2=1)

        assert res.axes == (Axis(3),)

    def test_dtype_is_forwarded(self):
        a = array(jnp.ones((3, 3), dtype=jnp.int8), (Axis(3, "i"), Axis(3, "j")))

        res = trace(a, dtype=jnp.int32)

        assert res.dtype == jnp.int32
        assert int(res.array) == 3

    def test_same_axis_twice_raises(self, x4):
        a = _named_rand(0, x4, Axis(4, "y"))
        with pytest.raises(ValueError, match="same named axes"):
            trace(a, axis1="x", axis2="x")

    def test_same_axis_by_name_and_position_raises(self, x4):
        a = _named_rand(0, x4, Axis(4, "y"))
        with pytest.raises(ValueError, match="same named axes"):
            trace(a, axis1="x", axis2=0)

    def test_unresolved_axis_raises(self, x4):
        a = _named_rand(0, x4, Axis(4, "y"))
        with pytest.raises(ValueError, match="could not be resolved"):
            trace(a, axis1="x", axis2="nope")

    def test_non_namedarray_raises_type_error(self):
        with pytest.raises(TypeError, match="trace"):
            trace(object())

    def test_composes_with_jit(self, z16):
        a = _named_rand(0, Axis(4, "i"), Axis(4, "j"), z16)

        res = jax.jit(lambda a: trace(a, axis1="i", axis2="j"))(a)

        assert res.axes == (z16,)
        _assert_close(res.array, jnp.trace(a.array))

    def test_gradient_is_identity_on_the_diagonal(self):
        i, j = Axis(4, "i"), Axis(4, "j")

        def f(a):
            return trace(array(a, (i, j))).array

        g = jax.grad(f)(_rand(0, (4, 4)))

        assert jnp.array_equal(g, jnp.eye(4))

    @given(st.integers(min_value=1, max_value=6))
    def test_trace_of_identity_is_its_size(self, n):
        a = array(jnp.eye(n), (Axis(n, "i"), Axis(n, "j")))
        assert float(trace(a).array) == n
