"""Tests for sharding.py.

A few notes on the approach, taken from how JAX itself tests sharding-aware
code:

- `mesh_1d` / `mesh_2d` (conftest.py) build a real `jax.sharding.Mesh` over
  several *simulated* CPU devices (`--xla_force_host_platform_device_count`),
  so the "resolve a `PM` against the current mesh, inside a mesh context"
  path is exercised for real rather than only mocked.
- The pure control-flow branches of `canonicalize_sharding` (unsupported
  types, the "you're not inside a mesh" error, an unresolved logical axis)
  don't need real devices at all, so those are tested by monkeypatching
  `jax.sharding.get_abstract_mesh` directly -- this is both faster and lets
  us assert on the "empty mesh" branch without depending on whichever mesh
  happens to be active in the rest of the session.
- `TestJitComposability` exercises the sharding utilities the way JAX's own
  docs test them (see https://docs.jax.dev/en/latest/201/sharding.html and
  https://docs.jax.dev/en/latest/301/index.html): a non-trivial function
  wrapped in `jax.jit`, with `in_shardings`/`out_shardings` derived from a
  `PM`, a `with_sharding_constraint` on an intermediate value, and
  `auto_mesh_axes` used to drop a subroutine into Auto-sharding mode inside
  an otherwise Explicit-sharded jit. `_dummy_contract` is a stand-in for a
  future real op (e.g. `lynqx.named.dot`) -- swap it out once that exists,
  the surrounding sharding-composability assertions should still hold.
"""

from typing import Any

import jax
import jax.numpy as jnp
import pytest
from jax import NamedSharding, P
from lynqx._src.array import NamedArrayImpl
from lynqx._src.axis_util import axis_shape_to_tuple, axis_sizes
from lynqx._src.sharding import (
    auto_mesh_axes,
    canonicalize_sharding,
    device_put,
    reshard,
    with_sharding_constraint,
)
from lynqx._src.typing import AxisShape, PM


# from lynqx._src.named import asnamed, zeros  # NOTE: adjust path if different in-repo


def named(a: Any, shape: AxisShape):
    shape = axis_shape_to_tuple(shape)
    return NamedArrayImpl(a, shape)


def zeros(shape: AxisShape):
    shape = axis_shape_to_tuple(shape)
    positional = axis_sizes(shape)

    return NamedArrayImpl(jnp.zeros(positional), shape)


def _dummy_contract(a, b):
    """Placeholder for a future non-trivial lynqx op (e.g. `named.dot`).

    Contracts the trailing axis of two equally-shaped arrays down to one
    value per row. Deliberately not a `NamedArray` op: it's here purely to
    give `jax.jit` something non-trivial to trace through so we can check
    that our sharding utilities survive that boundary, not to test any
    particular lynqx API surface.
    """
    return jnp.sum(a * b, axis=-1)


class _FakeMesh:
    """Minimal stand-in for `jax.sharding.AbstractMesh` -- only `.empty` is
    read by `canonicalize_sharding`."""

    def __init__(self, empty: bool):
        self.empty = empty


class TestCanonicalizeSharding:
    def test_none_passes_through(self):
        assert canonicalize_sharding(None, ("batch",), "some_fn") is None

    def test_partition_spec_passes_through_unchanged(self):
        spec = P("data")
        assert canonicalize_sharding(spec, ("batch",), "some_fn") is spec

    def test_named_sharding_passes_through_unchanged(self, mesh_1d):
        sharding = NamedSharding(mesh_1d, P("x"))
        assert canonicalize_sharding(sharding, ("batch",), "some_fn") is sharding

    def test_unsupported_type_raises_type_error_naming_the_caller(self):
        with pytest.raises(TypeError, match="some_fn"):
            canonicalize_sharding(object(), ("batch",), "some_fn")

    def test_unsupported_type_error_names_the_supported_types(self):
        with pytest.raises(TypeError, match="NamedSharding|PartitionSpec|PM"):
            canonicalize_sharding(42, ("batch",), "some_fn")

    def test_pm_outside_mesh_context_raises(self, monkeypatch):
        monkeypatch.setattr(jax.sharding, "get_abstract_mesh", lambda: _FakeMesh(empty=True))
        pm = PM({"batch": "data"})
        with pytest.raises(ValueError, match="mesh"):
            canonicalize_sharding(pm, ("batch",), "some_fn")

    def test_pm_with_unresolved_logical_axis_raises(self, monkeypatch):
        monkeypatch.setattr(jax.sharding, "get_abstract_mesh", lambda: _FakeMesh(empty=False))
        pm = PM({"batch": None})
        with pytest.raises(ValueError, match="batch"):
            canonicalize_sharding(pm, ("batch",), "some_fn")

    def test_fully_mapped_pm_resolves_to_partition_spec(self, monkeypatch):
        monkeypatch.setattr(jax.sharding, "get_abstract_mesh", lambda: _FakeMesh(empty=False))
        pm = PM({"batch": "data"})
        assert canonicalize_sharding(pm, ("batch",), "some_fn") == P("data")

    def test_pm_resolved_inside_a_real_mesh_context(self, mesh_1d):
        pm = PM({"batch": "x"})
        with jax.set_mesh(mesh_1d):
            result = canonicalize_sharding(pm, ("batch",), "some_fn")
        assert result == P("x")

    def test_pm_still_raises_inside_real_mesh_if_axis_unresolved(self, mesh_1d):
        pm = PM({"batch": None})
        with jax.set_mesh(mesh_1d):
            with pytest.raises(ValueError, match="batch"):
                canonicalize_sharding(pm, ("batch",), "some_fn")


class TestAutoMeshAxes:
    def test_none_out_sharding_never_touches_auto_axes(self, monkeypatch):
        auto_axes_calls = []
        monkeypatch.setattr(jax.sharding, "auto_axes", lambda *a, **k: auto_axes_calls.append((a, k)))

        def fn(x, *, scale):
            return x * scale

        wrapped = auto_mesh_axes(fn, ("batch",), None, scale=2)

        assert wrapped(3) == 6
        assert auto_axes_calls == []

    def test_hoist_kwargs_bound_before_auto_axes_is_applied(self, monkeypatch):
        captured = {}

        def fake_auto_axes(f, out_sharding):
            captured["fn"] = f
            captured["out_sharding"] = out_sharding
            return "AUTO_AXES_RESULT"

        monkeypatch.setattr(jax.sharding, "auto_axes", fake_auto_axes)
        monkeypatch.setattr(
            "lynqx._src.sharding.canonicalize_sharding",
            lambda sharding, axes, fn_name: sharding,
        )

        def fn(x, *, scale):
            return x * scale

        result = auto_mesh_axes(fn, ("batch",), P("data"), scale=5)

        assert result == "AUTO_AXES_RESULT"
        assert captured["out_sharding"] == P("data")
        assert captured["fn"](4) == 20  # scale=5 was already bound via partial

    def test_canonicalize_sharding_called_with_auto_mesh_fn_name(self, monkeypatch):
        seen = {}

        def fake_canonicalize(sharding, axes, fn_name):
            seen["fn_name"] = fn_name
            return sharding

        monkeypatch.setattr("lynqx._src.sharding.canonicalize_sharding", fake_canonicalize)
        monkeypatch.setattr(jax.sharding, "auto_axes", lambda f, out_sharding: out_sharding)

        auto_mesh_axes(lambda x: x, ("batch",), P("data"))

        assert seen["fn_name"] == "auto_mesh"

    def test_real_pm_out_sharding_wraps_and_executes_under_auto_axes(self, mesh_1d):
        pm = PM({"batch": "x"})

        def fn(x):
            return x

        with jax.set_mesh(mesh_1d):
            wrapped = auto_mesh_axes(fn, ("batch",), pm)
            x = zeros({"batch": 8})
            out = wrapped(x)

        assert out.axes == x.axes


class TestReshard:
    def test_reshards_a_single_named_array_leaf(self, mesh_1d):
        x = zeros({"batch": 8, "feature": 4})
        pm = PM({"batch": "x"})
        with jax.set_mesh(mesh_1d):
            out = reshard(x, pm)
        assert out.axes == x.axes

    def test_reshards_a_pytree_of_named_arrays(self, mesh_1d):
        tree = {"a": zeros({"batch": 8}), "b": zeros({"batch": 8, "feature": 4})}
        pm = PM({"batch": "x"})
        with jax.set_mesh(mesh_1d):
            out = reshard(tree, {"a": pm, "b": pm})
        assert set(out) == {"a", "b"}
        assert out["a"].axes == tree["a"].axes
        assert out["b"].axes == tree["b"].axes

    def test_single_sharding_broadcasts_prefix_style_over_pytree(self, mesh_1d):
        # `flatten_axes` (borrowed from vmap's `in_axes` handling) lets a
        # single spec stand in for "apply this to every leaf".
        tree = {"a": zeros({"batch": 8}), "b": zeros({"batch": 8})}
        pm = PM({"batch": "x"})
        with jax.set_mesh(mesh_1d):
            out = reshard(tree, pm)
        assert set(out) == {"a", "b"}

    def test_none_out_shardings_raises_without_a_concrete_target(self):
        # `reshard` delegates to `jax.reshard`, which requires a concrete,
        # non-empty-mesh sharding for every leaf -- there's no "leave it
        # alone" mode. This is worth a second look upstream: the signature
        # (`out_shardings: ShardingLike | None = None`) reads as if `None`
        # is a legal no-op, but it always raises in practice.
        x = zeros({"batch": 8})
        with pytest.raises(ValueError, match="non-empty mesh"):
            reshard(x, None)

    def test_pm_sharding_on_non_named_array_leaf_raises(self, mesh_1d):
        pm = PM({"batch": "x"})
        with jax.set_mesh(mesh_1d):
            with pytest.raises(TypeError, match="NamedArray"):
                reshard(jnp.zeros((8,)), pm)

    def test_plain_partition_spec_is_fine_for_non_named_array_leaf(self, mesh_1d):
        with jax.set_mesh(mesh_1d):
            out = reshard(jnp.zeros((8,)), P("x"))
        assert out.shape == (8,)

    def test_named_array_treated_as_a_single_leaf_not_decomposed(self, mesh_1d):
        # A NamedArray nested in a list should be matched one-for-one
        # against a single sharding, not flattened further.
        x = zeros({"batch": 8})
        pm = PM({"batch": "x"})
        with jax.set_mesh(mesh_1d):
            [out] = reshard([x], [pm])
        assert out.axes == x.axes


class TestDevicePut:
    def test_no_device_preserves_named_array_metadata(self):
        x = zeros({"batch": 8})
        out = device_put(x)
        assert out.axes == x.axes

    def test_no_device_on_plain_array_is_a_no_op_copy(self):
        arr = jnp.arange(4)
        out = device_put(arr)
        assert (out == arr).all()

    def test_pm_device_resolved_against_named_array_axes(self, mesh_1d):
        x = zeros({"batch": 8})
        pm = PM({"batch": "x"})
        with jax.set_mesh(mesh_1d):
            out = device_put(x, pm)
        assert out.axes == x.axes

    def test_pm_device_on_non_named_array_raises(self, mesh_1d):
        pm = PM({"batch": "x"})
        with jax.set_mesh(mesh_1d):
            with pytest.raises(TypeError, match="NamedArray"):
                device_put(jnp.zeros((8,)), pm)

    def test_src_may_alias_and_donate_are_forwarded_when_device_is_none(self, monkeypatch):
        seen = {}

        def fake_device_put(x, device=None, *, src=None, may_alias=False, donate=False):
            seen.update(device=device, src=src, may_alias=may_alias, donate=donate)
            return x

        monkeypatch.setattr(jax, "device_put", fake_device_put)

        device_put(jnp.arange(4), None, src="cpu-src", may_alias=True, donate=True)

        assert seen == {"device": None, "src": "cpu-src", "may_alias": True, "donate": True}


class TestWithShardingConstraint:
    """Under an all-`Explicit` mesh, `with_sharding_constraint` acts as an
    *assertion* against an array's actual current sharding, not as an
    operation that shards it for you -- that's what `reshard` is for (JAX
    says as much in its own error message). So these tests reshard first,
    and separately check the assertion actually fires on a mismatch.
    """

    def test_succeeds_when_actual_sharding_already_matches(self, mesh_1d):
        pm = PM({"batch": "x"})
        with jax.set_mesh(mesh_1d):
            x = reshard(zeros({"batch": 8}), pm)  # actually shard it first
            out = with_sharding_constraint(x, pm)
        assert out.axes == x.axes

    def test_raises_when_actual_sharding_does_not_match(self, mesh_1d):
        # A freshly created array is fully replicated (P(None, ...)); just
        # asserting a sharded spec at it doesn't make it so.
        pm = PM({"batch": "x"})
        with jax.set_mesh(mesh_1d):
            x = zeros({"batch": 8})
            with pytest.raises(AssertionError, match="with_sharding_constraint"):
                with_sharding_constraint(x, pm)

    def test_constrains_a_pytree_with_matching_sharding_structure(self, mesh_1d):
        pm = PM({"batch": "x"})
        with jax.set_mesh(mesh_1d):
            tree = {
                "a": reshard(zeros({"batch": 8}), pm),
                "b": reshard(zeros({"batch": 8}), pm),
            }
            out = with_sharding_constraint(tree, {"a": pm, "b": pm})
        assert set(out) == {"a", "b"}

    def test_pm_on_non_named_array_leaf_raises(self, mesh_1d):
        pm = PM({"batch": "x"})
        with jax.set_mesh(mesh_1d):
            with pytest.raises(TypeError, match="NamedArray"):
                with_sharding_constraint(jnp.zeros((8,)), pm)

    def test_mismatched_sharding_pytree_structure_raises(self, mesh_1d):
        tree = {"a": zeros({"batch": 8}), "b": zeros({"batch": 8})}
        pm = PM({"batch": "x"})
        with jax.set_mesh(mesh_1d):
            with pytest.raises(ValueError):
                with_sharding_constraint(tree, {"a": pm})  # missing "b"


class TestJitComposability:
    """Do the sharding utilities actually compose with `jax.jit`, or do they
    only work when called eagerly? This mirrors the patterns in JAX's own
    sharding docs: explicit `in_shardings`/`out_shardings` at a jit
    boundary, `with_sharding_constraint` pinning an intermediate, and
    `auto_axes` (via `auto_mesh_axes`) temporarily relaxing a subroutine to
    Auto sharding inside an Explicit-sharded jit.
    """

    def test_pm_derived_sharding_used_as_jit_in_and_out_shardings(self, mesh_1d):
        # https://docs.jax.dev/en/latest/201/sharding.html -- explicit
        # in_shardings/out_shardings at a jit boundary.
        pm = PM({"batch": "x"})
        with jax.set_mesh(mesh_1d):
            in_sharding = NamedSharding(mesh_1d, canonicalize_sharding(pm, ("batch", "feature"), "test"))
            out_sharding = NamedSharding(mesh_1d, canonicalize_sharding(pm, ("batch",), "test"))

            jit_contract = jax.jit(
                _dummy_contract,
                in_shardings=(in_sharding, in_sharding),
                out_shardings=out_sharding,
            )

            a = jax.device_put(jnp.arange(32.0).reshape(8, 4), in_sharding)
            b = jax.device_put(jnp.ones((8, 4)), in_sharding)
            out = jit_contract(a, b)

        assert jnp.allclose(out, jnp.sum(a * b, axis=-1))
        assert out.sharding.is_equivalent_to(out_sharding, out.ndim)

    def test_with_sharding_constraint_composes_inside_jit(self, mesh_1d):
        pm = PM({"batch": "x"})
        # `with_sharding_constraint` on plain (non-NamedArray) leaves takes a
        # bare P/NamedSharding rather than a PM -- see
        # test_pm_on_non_named_array_leaf_raises above -- so we resolve the
        # PM to a P up front via canonicalize_sharding, exactly as
        # `named`-level ops built on top of it would.
        with jax.set_mesh(mesh_1d):
            batch_spec = canonicalize_sharding(pm, ("batch",), "test")

            @jax.jit
            def jitted_contract(a, b):
                a = with_sharding_constraint(a, batch_spec)
                b = with_sharding_constraint(b, batch_spec)
                return _dummy_contract(a, b)

            # Under Explicit mesh axes, with_sharding_constraint asserts
            # rather than reshards, so the inputs must already carry the
            # target sharding *before* they're traced into the jit.
            a = reshard(jnp.arange(32.0).reshape(8, 4), batch_spec)
            b = reshard(jnp.ones((8, 4)), batch_spec)
            out = jitted_contract(a, b)

        assert jnp.allclose(out, jnp.sum(a * b, axis=-1))

    def test_auto_mesh_axes_composes_with_jit_and_explicit_sharding(self, mesh_1d):
        # https://docs.jax.dev/en/latest/301/index.html -- mixing Explicit
        # and Auto sharding: the outer jit runs under Explicit sharding
        # (via with_sharding_constraint), but `auto_mesh_axes` lets the
        # dummy "dot" subroutine run with the mesh's `batch` axis
        # temporarily switched to Auto.
        pm = PM({"batch": "x"})
        with jax.set_mesh(mesh_1d):
            auto_contract = auto_mesh_axes(_dummy_contract, ("batch",), pm)
            batch_spec = canonicalize_sharding(pm, ("batch",), "test")

            @jax.jit
            def outer(a, b):
                a = with_sharding_constraint(a, batch_spec)
                b = with_sharding_constraint(b, batch_spec)
                return auto_contract(a, b)

            # Same reasoning as the previous test: reshard before entering
            # the jit so the in-trace assertion has something true to check.
            a = reshard(jnp.arange(32.0).reshape(8, 4), batch_spec)
            b = reshard(jnp.ones((8, 4)), batch_spec)
            out = outer(a, b)

        assert jnp.allclose(out, jnp.sum(a * b, axis=-1))

    def test_resharded_arrays_feed_correctly_into_a_jitted_contract(self, mesh_1d):
        # reshard()/device_put() as producers, feeding a jitted consumer --
        # the pattern you'd actually see gluing a data pipeline to a
        # compiled training/inference step.
        pm = PM({"batch": "x"})
        with jax.set_mesh(mesh_1d):
            batch_spec = canonicalize_sharding(pm, ("batch",), "test")
            a = reshard(jnp.arange(32.0).reshape(8, 4), batch_spec)
            b = device_put(jnp.ones((8, 4)), batch_spec)

            jitted_contract = jax.jit(_dummy_contract)
            out = jitted_contract(a, b)

        assert jnp.allclose(out, jnp.sum(a * b, axis=-1))


class TestNamedArrayShardingRoundTrip:
    """A couple of end-to-end sanity checks mirroring how JAX's own test
    suite checks sharding round trips: put data on a mesh, read the
    resulting `.sharding` back off, and check it matches expectations."""

    def test_reshard_then_device_put_is_idempotent_on_axes(self, mesh_1d):
        x = named(jnp.arange(8.0), {"batch": 8})
        pm = PM({"batch": "x"})
        with jax.set_mesh(mesh_1d):
            once = reshard(x, pm)
            twice = device_put(once, pm)
        assert once.axes == twice.axes == x.axes

    def test_zeros_and_named_produce_same_axes_for_same_shape(self):
        a = zeros({"batch": 8, "feature": 4})
        b = named(jnp.zeros((8, 4)), {"batch": 8, "feature": 4})
        assert a.axes == b.axes
