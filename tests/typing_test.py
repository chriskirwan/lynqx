import lynqx as lqx
import pytest
from beartype import beartype
from jaxtyping import jaxtyped, TypeCheckError
from jaxtyping._storage import pop_shape_memo, push_shape_memo
from lynqx._src.custom_types import (
    _make_dtype_specifier,
    _make_named_array_type,
    _validate_ordered,
    _validate_unordered,
    NamedArrayAxisError,
    NamedAxisSpec,
)
from lynqx._src.named.creation import ones
from lynqx._src.typing import (
    Complex,
    Float,
    Int,
    Shaped,
)


class TestNamedAxisSpecGrammar:
    def test_scalar(self):
        spec = NamedAxisSpec("")
        assert spec.tokens == ()
        assert spec.ordered
        assert not spec.leading_extra
        assert not spec.trailing_extra
        assert not spec.is_wildcard
        assert spec.describe() == "scalar (no axes)"

    def test_wildcard(self):
        spec = NamedAxisSpec("...")
        assert spec.is_wildcard
        assert spec.describe() == "any axes (wildcard)"

    def test_unordered_open(self):
        spec = NamedAxisSpec("{a b:3}")
        assert not spec.ordered
        names = [t.name for t in spec.tokens]
        sizes = [t.size for t in spec.tokens]
        assert names == ["a", "b"]
        assert sizes == [None, 3]
        assert "unordered, open" in spec.describe()

    def test_unordered_with_no_tokens(self):
        # Whitespace-only unbraced spec: no tokens, unordered, open.
        spec = NamedAxisSpec("   ")
        assert spec.tokens == ()
        assert not spec.ordered
        assert spec.describe() == "any axes (unordered, open)"

    def test_ordered_exact(self):
        spec = NamedAxisSpec("a _ b:4")
        assert spec.ordered
        assert not spec.leading_extra
        assert not spec.trailing_extra
        assert [t.name for t in spec.tokens] == ["a", None, "b"]
        assert spec.tokens[2].size == 4
        assert "ordered, exact" in spec.describe()

    def test_ordered_suffix(self):
        spec = NamedAxisSpec("... a b")
        assert spec.ordered
        assert spec.leading_extra
        assert not spec.trailing_extra
        assert [t.name for t in spec.tokens] == ["a", "b"]
        assert "end with" in spec.describe()

    def test_ordered_prefix(self):
        spec = NamedAxisSpec("a b ...")
        assert spec.ordered
        assert not spec.leading_extra
        assert spec.trailing_extra
        assert [t.name for t in spec.tokens] == ["a", "b"]
        assert "start with" in spec.describe()

    def test_braced_wildcard_is_dtype_only_wildcard(self):
        spec = NamedAxisSpec("{...}")
        assert spec.is_wildcard
        assert not spec.ordered
        assert spec.leading_extra
        assert spec.trailing_extra
        assert spec.tokens == ()
        assert spec.describe() == "any axes (wildcard)"

    def test_ellipsis_rejected_inside_braces(self):
        with pytest.raises(ValueError, match="not meaningful inside a braced"):
            NamedAxisSpec("{a ... b}")

    def test_parentheses_are_no_longer_special_syntax(self):
        with pytest.raises(ValueError, match="Invalid axis token"):
            NamedAxisSpec("(a b)")

    def test_both_leading_and_trailing_extra_rejected(self):
        with pytest.raises(ValueError, match="contiguous-subsequence-anywhere"):
            NamedAxisSpec("... a ...")

    def test_ellipsis_only_valid_at_start_or_end(self):
        with pytest.raises(ValueError, match="only appear at the very start or end"):
            NamedAxisSpec("a ... b")

    def test_anonymous_axis_rejected_in_unordered(self):
        with pytest.raises(ValueError, match="anonymous axis"):
            NamedAxisSpec("{a _ b}")

    def test_invalid_identifier_rejected(self):
        with pytest.raises(ValueError, match="Invalid axis token"):
            NamedAxisSpec("1bad")

    def test_invalid_size_rejected(self):
        for bad in ["a:0", "a:-1", "a:01", "a: 3", "a:3.0"]:
            with pytest.raises(ValueError, match="Invalid size"):
                NamedAxisSpec(bad)

    def test_bare_digit_token_is_shorthand_for_anonymous_axis_with_size(self):
        spec = NamedAxisSpec("x y 3")
        assert [t.name for t in spec.tokens] == ["x", "y", None]
        assert [t.size for t in spec.tokens] == [None, None, 3]
        assert spec.tokens[2] == NamedAxisSpec("x y _:3").tokens[2]

    def test_bare_digit_token_matches_explicit_underscore_colon_form(self):
        assert NamedAxisSpec("x 3") == NamedAxisSpec("x _:3")

    def test_bare_digit_token_rejected_in_unordered_spec(self):
        with pytest.raises(ValueError, match="anonymous axis"):
            NamedAxisSpec("{a 3}")

    def test_invalid_bare_digit_size_rejected(self):
        for bad in ["0", "01"]:
            with pytest.raises(ValueError, match="Invalid size"):
                NamedAxisSpec(bad)

    def test_duplicate_named_axis_rejected(self):
        with pytest.raises(ValueError, match="Duplicate named axis"):
            NamedAxisSpec("a a")

    def test_duplicate_named_axis_rejected_in_braces(self):
        with pytest.raises(ValueError, match="Duplicate named axis"):
            NamedAxisSpec("{a a}")

    def test_equality_and_hash(self):
        assert NamedAxisSpec("a b:3") == NamedAxisSpec("a b:3")
        assert hash(NamedAxisSpec("a b:3")) == hash(NamedAxisSpec("a b:3"))
        assert NamedAxisSpec("a b:3") != NamedAxisSpec("a b:4")
        assert NamedAxisSpec("a b") != NamedAxisSpec("{a b}")  # ordered vs unordered

    def test_repr(self):
        assert repr(NamedAxisSpec("a b")) == "NamedAxisSpec('a b')"


class TestAxisValidationUnordered:
    def setup_method(self):
        push_shape_memo({})

    def teardown_method(self):
        pop_shape_memo()

    def test_exact_named_axes_pass(self):
        spec = NamedAxisSpec("{a b}")
        axes = (lqx.Axis(2, "a"), lqx.Axis(3, "b"))
        _validate_unordered(axes, spec)

    def test_extra_axes_on_instance_are_allowed(self):
        spec = NamedAxisSpec("{a}")
        axes = (lqx.Axis(2, "a"), lqx.Axis(3, "b"), lqx.Axis(4, None))
        _validate_unordered(axes, spec)

    def test_missing_axis_raises(self):
        spec = NamedAxisSpec("{a b}")
        axes = (lqx.Axis(2, "a"),)
        with pytest.raises(NamedArrayAxisError, match="missing axis 'b'"):
            _validate_unordered(axes, spec)

    def test_wrong_explicit_size_raises(self):
        spec = NamedAxisSpec("{a:3}")
        axes = (lqx.Axis(5, "a"),)
        with pytest.raises(NamedArrayAxisError, match="has size 5, expected 3"):
            _validate_unordered(axes, spec)

    def test_anonymous_axes_on_instance_are_ignored(self):
        spec = NamedAxisSpec("{a}")
        axes = (lqx.Axis(2, "a"), lqx.Axis(99, None))
        _validate_unordered(axes, spec)


class TestAxisValidationOrdered:
    def setup_method(self):
        push_shape_memo({})

    def teardown_method(self):
        pop_shape_memo()

    def test_exact_match_passes(self):
        spec = NamedAxisSpec("a b")
        axes = (lqx.Axis(2, "a"), lqx.Axis(3, "b"))
        _validate_ordered(axes, spec)

    def test_wrong_order_fails(self):
        spec = NamedAxisSpec("a b")
        axes = (lqx.Axis(3, "b"), lqx.Axis(2, "a"))
        with pytest.raises(NamedArrayAxisError):
            _validate_ordered(axes, spec)

    def test_exact_wrong_length_fails(self):
        spec = NamedAxisSpec("a b")
        axes = (lqx.Axis(2, "a"), lqx.Axis(3, "b"), lqx.Axis(4, "c"))
        with pytest.raises(NamedArrayAxisError):
            _validate_ordered(axes, spec)

    def test_anonymous_token_matches_any_anonymous_axis(self):
        spec = NamedAxisSpec("a _ b")
        axes = (lqx.Axis(2, "a"), lqx.Axis(999, None), lqx.Axis(3, "b"))
        _validate_ordered(axes, spec)

    def test_anonymous_token_rejects_named_axis(self):
        spec = NamedAxisSpec("a _ b")
        axes = (lqx.Axis(2, "a"), lqx.Axis(999, "oops"), lqx.Axis(3, "b"))
        with pytest.raises(NamedArrayAxisError):
            _validate_ordered(axes, spec)

    def test_anonymous_token_with_size_checks_size(self):
        spec = NamedAxisSpec("a _:5 b")
        ok_axes = (lqx.Axis(2, "a"), lqx.Axis(5, None), lqx.Axis(3, "b"))
        bad_axes = (lqx.Axis(2, "a"), lqx.Axis(6, None), lqx.Axis(3, "b"))
        _validate_ordered(ok_axes, spec)
        with pytest.raises(NamedArrayAxisError):
            _validate_ordered(bad_axes, spec)

    def test_bare_digit_token_checks_size_same_as_underscore_colon_form(self):
        spec = NamedAxisSpec("x y 3")
        ok_axes = (lqx.Axis(32, "x"), lqx.Axis(32, "y"), lqx.Axis(3, None))
        bad_axes = (lqx.Axis(32, "x"), lqx.Axis(32, "y"), lqx.Axis(4, None))
        _validate_ordered(ok_axes, spec)
        with pytest.raises(NamedArrayAxisError):
            _validate_ordered(bad_axes, spec)

    def test_suffix_allows_leading_extra_axes(self):
        spec = NamedAxisSpec("... a b")
        axes = (lqx.Axis(9, "extra1"), lqx.Axis(9, "extra2"), lqx.Axis(2, "a"), lqx.Axis(3, "b"))
        _validate_ordered(axes, spec)

    def test_suffix_fails_if_too_short(self):
        spec = NamedAxisSpec("... a b")
        axes = (lqx.Axis(2, "a"),)
        with pytest.raises(NamedArrayAxisError):
            _validate_ordered(axes, spec)

    def test_prefix_allows_trailing_extra_axes(self):
        spec = NamedAxisSpec("a b ...")
        axes = (lqx.Axis(2, "a"), lqx.Axis(3, "b"), lqx.Axis(9, "extra"))
        _validate_ordered(axes, spec)

    def test_scalar_spec_requires_zero_axes(self):
        spec = NamedAxisSpec("")
        _validate_ordered((), spec)
        with pytest.raises(NamedArrayAxisError):
            _validate_ordered((lqx.Axis(1, "a"),), spec)

    def test_wildcard_matches_any_axes_via_validate_ordered(self):
        spec = NamedAxisSpec("...")
        _validate_ordered((), spec)
        _validate_ordered((lqx.Axis(1, "a"), lqx.Axis(2, None)), spec)


class TestDtypeAndInstanceValidation:
    def test_matching_dtype_and_axes_passes(self):
        chain = lqx.Axis(4, "chain")
        arr = ones(chain, dtype="csingle")
        assert isinstance(arr, Complex[lqx.NamedArray, "chain"])

    def test_wrong_dtype_fails(self):
        chain = lqx.Axis(4, "chain")
        arr = ones(chain, dtype="csingle")
        assert not isinstance(arr, Int[lqx.NamedArray, "chain"])

    def test_wrong_axis_name_fails(self):
        chain = lqx.Axis(4, "chain")
        arr = ones(chain, dtype="csingle")
        assert not isinstance(arr, Complex[lqx.NamedArray, "batch"])

    def test_wildcard_ignores_axes_but_checks_dtype(self):
        chain = lqx.Axis(4, "chain")
        arr = ones(chain, dtype="csingle")
        assert isinstance(arr, Complex[lqx.NamedArray, "..."])
        assert not isinstance(arr, Int[lqx.NamedArray, "..."])

    def test_shaped_accepts_any_dtype(self):
        chain = lqx.Axis(4, "chain")
        arr = ones(chain, dtype="csingle")
        assert isinstance(arr, Shaped[lqx.NamedArray, "chain"])

    def test_scalar_spec_on_scalar_array(self):
        scalar = ones((), dtype="float32")
        assert isinstance(scalar, Float[lqx.NamedArray, ""])

    def test_plain_array_type_falls_through_to_stock_jaxtyping(self):
        import jax.numpy as jnp

        x = jnp.zeros((3, 4), dtype=jnp.float32)
        assert isinstance(x, Float[jnp.ndarray, "a b"])
        assert not isinstance(x, Int[jnp.ndarray, "a b"])

    def test_brace_syntax_rejected_for_plain_arrays(self):
        import jax.numpy as jnp

        with pytest.raises(TypeError, match="only valid for NamedArray"):
            Float[jnp.ndarray, "{a b}"]


class TestCaching:
    def test_same_dtype_and_dim_str_returns_identical_type(self):
        t1 = _make_named_array_type(Float, NamedAxisSpec("chain"))
        t2 = _make_named_array_type(Float, NamedAxisSpec("chain"))
        assert t1 is t2

    def test_equal_but_distinct_spec_instances_share_cache_entry(self):
        spec_a = NamedAxisSpec("a b:3")
        spec_b = NamedAxisSpec("a b:3")
        assert spec_a is not spec_b
        t1 = _make_named_array_type(Float, spec_a)
        t2 = _make_named_array_type(Float, spec_b)
        assert t1 is t2

    def test_different_dim_str_returns_different_type(self):
        t1 = _make_named_array_type(Float, NamedAxisSpec("chain"))
        t2 = _make_named_array_type(Float, NamedAxisSpec("batch"))
        assert t1 is not t2

    def test_dtype_specifier_is_cached_per_jt_dtype(self):
        import jaxtyping as jt

        s1 = _make_dtype_specifier(jt.Float)
        s2 = _make_dtype_specifier(jt.Float)
        assert s1 is s2

    def test_repeated_annotation_syntax_reuses_cache(self):
        c1 = Complex[lqx.NamedArray, "chain"]
        c2 = Complex[lqx.NamedArray, "chain"]
        assert c1 is c2


class TestBindAxisUnordered:
    def setup_method(self):
        push_shape_memo({})

    def teardown_method(self):
        pop_shape_memo()

    def test_first_occurrence_binds(self):
        spec = NamedAxisSpec("{chain}")
        _validate_unordered((lqx.Axis(4, "chain"),), spec)

    def test_consistent_second_occurrence_passes(self):
        spec = NamedAxisSpec("{chain}")
        _validate_unordered((lqx.Axis(4, "chain"),), spec)
        _validate_unordered((lqx.Axis(4, "chain"),), spec)

    def test_inconsistent_second_occurrence_raises(self):
        spec = NamedAxisSpec("{chain}")
        _validate_unordered((lqx.Axis(4, "chain"),), spec)
        with pytest.raises(NamedArrayAxisError, match="chain"):
            _validate_unordered((lqx.Axis(8, "chain"),), spec)

    def test_unrelated_axis_names_do_not_interact(self):
        _validate_unordered((lqx.Axis(4, "chain"),), NamedAxisSpec("{chain}"))
        _validate_unordered((lqx.Axis(99, "batch"),), NamedAxisSpec("{batch}"))


class TestBindAxisOrdered:
    def setup_method(self):
        push_shape_memo({})

    def teardown_method(self):
        pop_shape_memo()

    def test_anonymous_axes_are_not_bound(self):
        spec = NamedAxisSpec("batch _ chain")
        axes_a = (lqx.Axis(2, "batch"), lqx.Axis(5, None), lqx.Axis(4, "chain"))
        axes_b = (lqx.Axis(2, "batch"), lqx.Axis(999, None), lqx.Axis(4, "chain"))
        _validate_ordered(axes_a, spec)
        _validate_ordered(axes_b, spec)

    def test_named_axis_mismatch_across_calls_raises(self):
        spec = NamedAxisSpec("batch _ chain")
        axes_a = (lqx.Axis(2, "batch"), lqx.Axis(5, None), lqx.Axis(4, "chain"))
        axes_b = (lqx.Axis(2, "batch"), lqx.Axis(5, None), lqx.Axis(7, "chain"))
        _validate_ordered(axes_a, spec)
        with pytest.raises(NamedArrayAxisError, match="chain"):
            _validate_ordered(axes_b, spec)


class TestMemoResetsBetweenCalls:
    def test_fresh_push_pop_clears_bindings(self):
        spec = NamedAxisSpec("chain")

        push_shape_memo({})
        _validate_ordered((lqx.Axis(4, "chain"),), spec)
        pop_shape_memo()

        push_shape_memo({})
        try:
            _validate_ordered((lqx.Axis(8, "chain"),), spec)
        finally:
            pop_shape_memo()


class TestEndToEndJaxtyped:
    def test_matching_chain_sizes_succeed(self):
        c_chain = Complex[lqx.NamedArray, "chain"]

        @jaxtyped(typechecker=beartype)
        def run_chains(fields: c_chain, keys: c_chain) -> c_chain:
            return fields * keys

        chain = lqx.Axis(4, "chain")
        u = ones(chain, dtype="csingle")
        v = ones(chain, dtype="csingle")

        out = run_chains(u, v)
        assert out.axes == (chain,)

    def test_mismatched_chain_sizes_raise_type_violation_not_internal_error(self):
        c_chain = Complex[lqx.NamedArray, "chain"]

        @jaxtyped(typechecker=beartype)
        def run_chains(fields: c_chain, keys: c_chain) -> c_chain:
            return fields * keys

        chain_a = lqx.Axis(4, "chain")
        chain_b = lqx.Axis(8, "chain")
        u = ones(chain_a, dtype="csingle")
        v = ones(chain_b, dtype="csingle")

        with pytest.raises(TypeCheckError):
            run_chains(u, v)

    def test_binding_crosses_named_and_plain_array_arguments(self):
        import jax.numpy as jnp
        from jaxtyping import Int as PlainInt

        c_chain = Complex[lqx.NamedArray, "chain"]

        @jaxtyped(typechecker=beartype)
        def combine(fields: c_chain, indices: PlainInt[jnp.ndarray, "chain"]) -> c_chain:
            return fields

        chain = lqx.Axis(4, "chain")
        fields = ones(chain, dtype="csingle")

        combine(fields, jnp.zeros((4,), dtype=jnp.int32))

        with pytest.raises(TypeCheckError):
            combine(fields, jnp.zeros((7,), dtype=jnp.int32))
