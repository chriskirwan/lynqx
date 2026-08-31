import pytest
from hypothesis import given, strategies as st
from lynqx._src.axis import Axis
from lynqx._src.axis_util import (
    axis_index,
    axis_indices,
    axis_names,
    axis_selection_to_tuple,
    axis_shape_to_tuple,
    axis_sizes,
    axis_spec_to_tuple,
    AxisMatch,
    check_unique_axis_names,
    concatenate_axes,
    intersect_axes,
    is_anonymous_axis,
    is_axis_compatible,
    is_named_axis,
    make_axes,
    match_axes,
    MatchAxisResult,
    remove_axes,
    replace_axes,
    resolve_axes,
    union_axes,
    validate_unique_axes,
)
from strategies import (
    axes as axes_strategy,
    disjoint_named_axis_tuple_pairs,
    disjoint_named_axis_tuples,
    duplicate_named_axis_pair,
)


# Shared axis literals reused across the set-like algebra tests below
# (union/intersect/concatenate/remove/replace) so those tests aren't each
# re-declaring `Axis(4, "x")` from scratch.
X4 = Axis(4, "x")
X8 = Axis(8, "x")  # same name as X4, conflicting size -- for conflict tests
Y8 = Axis(8, "y")
Z16 = Axis(16, "z")


class TestIsAxisCompatible:
    @pytest.mark.parametrize(
        "a,b,expected",
        [
            (Axis(4, "x"), Axis(4, "x"), True),
            (Axis(4, "x"), Axis(8, "x"), False),
            (Axis(4, "x"), Axis(4, "y"), False),
            (Axis(4), Axis(4), True),
            (Axis(4), Axis(8), False),
            (Axis(4, "x"), Axis(4), False),
            (Axis(4), Axis(4, "x"), False),
        ],
        ids=[
            "same_name_same_size",
            "same_name_diff_size",
            "diff_name_same_size",
            "both_anon_same_size",
            "both_anon_diff_size",
            "named_vs_anon",
            "anon_vs_named",
        ],
    )
    def test_compatibility_matrix(self, a, b, expected):
        assert is_axis_compatible(a, b) is expected

    # Symmetry itself is a general property, not a specific case -- see
    # TestIsAxisCompatibleProperties below, which fuzzes it across named,
    # anonymous, and mixed pairs rather than one hand-picked equal pair.


class TestIsAxisCompatibleProperties:
    @given(axes_strategy, axes_strategy)
    def test_symmetric(self, a, b):
        assert is_axis_compatible(a, b) == is_axis_compatible(b, a)

    @given(axes_strategy)
    def test_reflexive(self, a):
        assert is_axis_compatible(a, a) is True


class TestCheckUniqueAxisNames:
    def test_allows_disjoint_named_axes(self):
        check_unique_axis_names((X4, Y8))  # must not raise

    def test_allows_repeated_anonymous_axes(self):
        check_unique_axis_names((Axis(4), Axis(4), Axis(8)))  # must not raise

    def test_allows_bare_string_specs_with_distinct_names(self):
        check_unique_axis_names(("x", "y"))  # must not raise

    def test_bare_string_and_axis_object_with_same_name_collide(self):
        # A bare string *is* its own name, so `"x"` and `Axis(4, "x")` are
        # the same collision as two `Axis` objects sharing a name.
        with pytest.raises(ValueError, match="'x'"):
            check_unique_axis_names(("x", Axis(4, "x")))

    def test_duplicate_named_axis_raises(self):
        with pytest.raises(ValueError, match="'x'"):
            check_unique_axis_names((X4, X8))  # same name, different size

    def test_duplicate_error_lists_every_colliding_index(self):
        axes = (X4, Y8, Axis(2, "x"), Axis(1, "x"))
        with pytest.raises(ValueError, match=r"'x' at \[0, 2, 3\]"):
            check_unique_axis_names(axes)

    def test_multiple_different_duplicate_names_all_reported(self):
        axes = (X4, X8, Y8, Axis(1, "y"))
        with pytest.raises(ValueError) as exc_info:
            check_unique_axis_names(axes)
        assert "'x'" in str(exc_info.value)
        assert "'y'" in str(exc_info.value)

    def test_single_bare_axis_is_normalized_to_a_one_tuple(self):
        check_unique_axis_names(X4)  # must not raise -- trivially unique

    def test_single_bare_string_is_normalized_to_a_one_tuple(self):
        # Also confirms the string isn't iterated character-by-character
        # (which would otherwise "duplicate" any repeated letter).
        check_unique_axis_names("xx")  # must not raise

    def test_non_iterable_scalar_raises_type_error(self):
        with pytest.raises(TypeError):
            check_unique_axis_names(5)

    def test_non_axislike_element_raises_type_error_naming_its_index(self):
        with pytest.raises(TypeError, match="index 1"):
            check_unique_axis_names([X4, 5])

    def test_bytes_input_raises_type_error(self):
        # NOTE: `axes` is reassigned to `axis_spec_to_tuple(axes)` *before*
        # the `isinstance(axes, (bytes, bytearray))` guard runs, and
        # `axis_spec_to_tuple` always returns a plain `tuple` -- so that
        # guard can never actually see a `bytes`/`bytearray` value and is
        # currently dead code. What actually raises here is the per-element
        # `AxisLike` check one level down (bytes iterate into ints), with a
        # less specific message than "Expected AxisSelector, got bytes"
        # would give. If the more specific message matters, the guard
        # needs to run *before* the `axis_spec_to_tuple` call instead.
        with pytest.raises(TypeError):
            check_unique_axis_names(b"ab")


class TestCheckUniqueAxisNamesProperties:
    @given(disjoint_named_axis_tuples())
    def test_never_raises_on_disjoint_named_axes(self, axes):
        check_unique_axis_names(axes)  # must not raise

    @given(duplicate_named_axis_pair(), disjoint_named_axis_tuples())
    def test_always_raises_on_a_duplicate_pair_anywhere_in_the_input(self, pair, rest):
        axes = pair + rest
        with pytest.raises(ValueError):
            check_unique_axis_names(axes)

    @given(st.integers(min_value=0, max_value=5))
    def test_any_number_of_anonymous_axes_never_collide(self, n):
        check_unique_axis_names(tuple(Axis(4) for _ in range(n)))  # must not raise


class TestValidateUniqueAxes:
    """Unit tests for the decorator via small dummy functions -- these
    exercise the wrapping/binding machinery itself, independent of which
    real functions it ends up applied to."""

    def test_decorator_factory_form_passes_valid_input_through(self):
        @validate_unique_axes(arg_names=("axes",))
        def fn(axes):
            return axes

        assert fn((X4, Y8)) == (X4, Y8)

    def test_decorator_factory_form_raises_on_duplicate(self):
        @validate_unique_axes(arg_names=("axes",))
        def fn(axes):
            return axes

        with pytest.raises(ValueError):
            fn((X4, X8))

    def test_direct_call_form_is_equivalent_to_decorator_form(self):
        def fn(axes):
            return axes

        wrapped = validate_unique_axes(fn, arg_names=("axes",))

        assert wrapped((X4, Y8)) == (X4, Y8)
        with pytest.raises(ValueError):
            wrapped((X4, X8))

    def test_only_the_named_parameters_are_validated(self):
        @validate_unique_axes(arg_names=("a",))
        def fn(a, b):
            return a, b

        # `b` is malformed but not in `arg_names`, so it's left untouched.
        assert fn((X4,), (X4, X4)) == ((X4,), (X4, X4))

    def test_each_named_parameter_is_checked_independently(self):
        @validate_unique_axes(arg_names=("a", "b"))
        def fn(a, b):
            return a, b

        with pytest.raises(ValueError):
            fn((X4, X8), (Y8,))
        with pytest.raises(ValueError):
            fn((X4,), (Y8, Y8))

    def test_works_with_positional_and_keyword_calls(self):
        @validate_unique_axes(arg_names=("axes",))
        def fn(x, axes):
            return x, axes

        with pytest.raises(ValueError):
            fn(1, (X4, X8))
        with pytest.raises(ValueError):
            fn(1, axes=(X4, X8))
        with pytest.raises(ValueError):
            fn(x=1, axes=(X4, X8))

    def test_missing_value_is_skipped_not_validated(self):
        # `if value is not None` means an omitted (default-`None`) or
        # explicitly-`None` argument is skipped entirely, letting a
        # parameter be genuinely optional rather than forced to always
        # supply a well-formed axes sequence.
        @validate_unique_axes(arg_names=("axes",))
        def fn(axes=None):
            return axes

        assert fn() is None  # default applied, then skipped
        assert fn(axes=None) is None  # explicit None, also skipped

    def test_typo_in_arg_names_silently_skips_validation(self):
        # `bound.arguments.get(name)` -- not `[name]` -- means a typo'd or
        # stale `arg_names` entry doesn't fail loudly: it just isn't found
        # in `bound.arguments`, `.get()` returns None, and validation for
        # that entry is silently skipped. This test documents that as
        # current behavior, not as something asserted to be correct --
        # worth deciding deliberately whether a typo should instead raise
        # (e.g. `bound.arguments[name]`, or validating `arg_names` against
        # `inspect.signature(fn).parameters` once, at decoration time).
        @validate_unique_axes(arg_names=("axess",))  # typo: should be "axes"
        def fn(axes):
            return axes

        result = fn((X4, X8))  # genuinely duplicate-named -- does NOT raise
        assert result == (X4, X8)

    def test_wraps_preserves_name_and_docstring(self):
        @validate_unique_axes(arg_names=("axes",))
        def some_fn(axes):
            """Some docstring."""
            return axes

        assert some_fn.__name__ == "some_fn"
        assert some_fn.__doc__ == "Some docstring."


class TestAxisSpecToTuple:
    _ax = Axis(4, "batch")

    @pytest.mark.parametrize(
        "spec,expected",
        [
            ("batch", ("batch",)),
            (_ax, (_ax,)),
            (["batch", _ax], ("batch", _ax)),
            ([], ()),
        ],
        ids=["single_string", "single_axis_object", "iterable_of_mixed_axislike", "empty_iterable"],
    )
    def test_normalizes_to_tuple(self, spec, expected):
        assert axis_spec_to_tuple(spec) == expected

    def test_generator_consumed_into_tuple(self):
        # Needs a fresh, stateful generator per call -- not representable
        # as a static parametrize value alongside the cases above.
        gen = (n for n in ("a", "b"))
        assert axis_spec_to_tuple(gen) == ("a", "b")


class TestAxisShapeToTuple:
    _ax = Axis(4, "x")

    @pytest.mark.parametrize(
        "shape,expected",
        [
            (_ax, (_ax,)),
            ((_ax, Axis(8)), (_ax, Axis(8))),
            ({"x": 4, "y": 8}, (Axis(4, "x"), Axis(8, "y"))),
        ],
        ids=["single_axis", "iterable_of_axes", "mapping_converted_size_to_name"],
    )
    def test_normalizes_to_tuple(self, shape, expected):
        assert axis_shape_to_tuple(shape) == expected

    def test_non_axis_member_raises_type_error(self):
        with pytest.raises(TypeError):
            axis_shape_to_tuple(["not_an_axis"])


class TestAxisSelectionToTuple:
    @pytest.mark.parametrize(
        "selection,expected",
        [
            ("batch", ("batch",)),
            (0, (0,)),
            (["batch", 1], ("batch", 1)),
        ],
        ids=["single_string", "single_int", "mixed_list"],
    )
    def test_normalizes(self, selection, expected):
        assert axis_selection_to_tuple(selection) == expected


class TestAxisProperties:
    def test_axis_names_mixed(self):
        axes = [Axis(4, "x"), Axis(8), "y"]
        assert axis_names(axes) == ("x", None, "y")

    @pytest.mark.parametrize(
        "spec,expected",
        [
            ([Axis(4, "x"), Axis(8)], (4, 8)),
            (Axis(4, "x"), (4,)),
            ({"x": 4, "y": 8}, (4, 8)),
        ],
        ids=["from_sequence", "from_single_axis", "from_mapping"],
    )
    def test_axis_sizes(self, spec, expected):
        assert axis_sizes(spec) == expected

    @pytest.mark.parametrize(
        "axis,expected_named,expected_anon",
        [
            (Axis(4, "x"), True, False),
            (Axis(4), False, True),
            ("x", True, False),
        ],
        ids=["named_axis_obj", "anon_axis_obj", "bare_string"],
    )
    def test_is_named_is_anonymous(self, axis, expected_named, expected_anon):
        assert is_named_axis(axis) is expected_named
        assert is_anonymous_axis(axis) is expected_anon


class TestAxisIndex:
    @pytest.fixture
    def selection(self):
        return (Axis(4, "batch"), Axis(8, "head"), Axis(16))

    @pytest.mark.parametrize(
        "selector,expected",
        [
            ("head", 1),
            (2, 2),
            (-1, 2),
            (3, None),
            (-4, None),
            ("nope", None),
        ],
        ids=[
            "by_name",
            "by_positive_int",
            "by_negative_int",
            "positive_int_out_of_range",
            "negative_int_out_of_range",
            "missing_name",
        ],
    )
    def test_lookup(self, selection, selector, expected):
        assert axis_index(selection, selector) == expected

    def test_find_by_axis_value(self, selection):
        # Needs the same fixture instance on both sides (the axis object
        # *is* an element of `selection`), so it can't share the static
        # parametrize table above.
        assert axis_index(selection, selection[0]) == 0

    def test_ambiguous_name_raises(self):
        selection = (Axis(4, "x"), Axis(8, "x"))
        with pytest.raises(ValueError, match="abmigious|ambiguous"):
            axis_index(selection, "x")


class TestAxisIndices:
    @pytest.fixture
    def selection(self):
        return (Axis(4, "batch"), Axis(8, "head"))

    def test_all_resolved(self, selection):
        assert axis_indices(selection, ["head", "batch"]) == (1, 0)

    def test_strict_raises_on_unresolved(self, selection):
        with pytest.raises(ValueError):
            axis_indices(selection, ["nope"], strict=True)

    def test_non_strict_returns_none_for_unresolved(self, selection):
        assert axis_indices(selection, ["nope"], strict=False) == (None,)

    def test_default_is_strict(self, selection):
        with pytest.raises(ValueError):
            axis_indices(selection, ["nope"])


class TestResolveAxes:
    def test_resolves_to_axis_objects(self):
        x, y = Axis(4, "x"), Axis(8, "y")
        assert resolve_axes((x, y), ["y", "x"]) == (y, x)

    # "raises on an unresolved selector" is shared with remove_axes/
    # replace_axes -- see TestUnresolvedSelectorRaises below.


class TestMatchAxes:
    def test_matches_by_name(self):
        source = (Axis(4, "batch"), Axis(8, "head"))
        target = (Axis(8, "head"), Axis(4, "batch"))
        result = match_axes(source, target)
        assert set(result.matches) == {AxisMatch(0, 1), AxisMatch(1, 0)}
        assert result.unmatched_source == ()
        assert result.unmatched_target == ()

    def test_matches_sorted_by_target_index(self):
        source = (Axis(4, "batch"), Axis(8, "head"))
        target = (Axis(8, "head"), Axis(4, "batch"))
        result = match_axes(source, target)
        targets = [m.target for m in result.matches]
        assert targets == sorted(targets)

    @pytest.mark.parametrize(
        "allow_positional_fallback,expected_matches",
        [(False, ()), (True, (AxisMatch(0, 0),))],
        ids=["fallback_disabled_stays_unmatched", "fallback_enabled_pairs_anonymous"],
    )
    def test_anonymous_pairing_depends_on_positional_fallback(self, allow_positional_fallback, expected_matches):
        source = (Axis(4),)
        target = (Axis(4),)
        result = match_axes(source, target, allow_positional_fallback=allow_positional_fallback)
        assert result.matches == expected_matches
        expected_unmatched = () if expected_matches else (0,)
        assert result.unmatched_source == expected_unmatched
        assert result.unmatched_target == expected_unmatched

    def test_positional_fallback_does_not_pair_unmatched_named_axes(self):
        source = (Axis(4, "extra"),)
        target = (Axis(8, "other"),)
        result = match_axes(source, target, allow_positional_fallback=True)
        assert result.matches == ()
        assert result.unmatched_source == (0,)
        assert result.unmatched_target == (0,)

    def test_named_matches_take_priority_over_positional_fallback(self):
        source = (Axis(4, "batch"), Axis(8))
        target = (Axis(8), Axis(4, "batch"))
        result = match_axes(source, target, allow_positional_fallback=True)
        assert AxisMatch(0, 1) in result.matches  # Axis match by name, not position
        assert AxisMatch(1, 0) in result.matches  # remaining Axis matched positionally

    def test_returns_match_axis_result_namedtuple(self):
        result = match_axes((), ())
        assert isinstance(result, MatchAxisResult)
        assert result == MatchAxisResult((), (), ())


class TestUnionAxes:
    def test_disjoint_names_appended(self):
        assert union_axes((X4,), (Y8,)) == (X4, Y8)

    def test_shared_name_deduplicated(self):
        assert union_axes((X4,), (X4,)) == (X4,)

    def test_anonymous_axes_from_b_always_appended(self):
        assert union_axes((X4,), (Axis(8), Axis(8))) == (X4, Axis(8), Axis(8))

    def test_order_is_a_then_new_from_b(self):
        a = (X4, Y8)
        b = (Y8, Z16)
        assert union_axes(a, b) == (X4, Y8, Z16)


class TestUnionAxesProperties:
    @given(disjoint_named_axis_tuples())
    def test_idempotent(self, a):
        assert union_axes(a, a) == a

    @given(disjoint_named_axis_tuple_pairs())
    def test_result_contains_every_input_name_exactly_once(self, pair):
        a, b = pair
        result_names = axis_names(union_axes(a, b))
        assert len(result_names) == len(set(result_names))
        assert set(axis_names(a)) | set(axis_names(b)) == set(result_names)


class TestIntersectAxes:
    def test_only_shared_names_kept_in_a_order(self):
        a = (X4, Y8)
        b = (Y8, Z16)
        assert intersect_axes(a, b) == (Y8,)

    def test_anonymous_axes_never_kept(self):
        a = (Axis(4), Y8)
        b = (Axis(4), Y8)
        assert intersect_axes(a, b) == (Y8,)

    def test_no_overlap_returns_empty(self):
        assert intersect_axes((X4,), (Y8,)) == ()


class TestSetLikeAxisOpsShared:
    """`union_axes` and `intersect_axes` share identical conflict
    detection: a name present in both inputs with different sizes always
    raises `ValueError`. One parametrized test over both functions instead
    of two near-identical ones per class."""

    @pytest.mark.parametrize("op", [union_axes, intersect_axes], ids=["union", "intersect"])
    def test_shared_name_conflicting_size_raises(self, op):
        with pytest.raises(ValueError, match="conflicting sizes"):
            op((X4,), (X8,))


class TestConcatenateAxes:
    @pytest.mark.parametrize(
        "a,b,expected",
        [
            ((X4,), (Y8,), (X4, Y8)),
            ((Axis(4),), (Axis(4),), (Axis(4), Axis(4))),
        ],
        ids=["simple_concat", "anonymous_axes_never_clash"],
    )
    def test_concat(self, a, b, expected):
        assert concatenate_axes(a, b) == expected

    def test_shared_name_raises(self):
        with pytest.raises(ValueError, match="sharing names"):
            concatenate_axes((X4,), (X4,))

    def test_accepts_generators(self):
        result = concatenate_axes(iter([X4]), iter([Y8]))
        assert result == (X4, Y8)


class TestConcatenateAxesProperties:
    @given(disjoint_named_axis_tuple_pairs())
    def test_length_is_additive(self, pair):
        a, b = pair
        assert len(concatenate_axes(a, b)) == len(a) + len(b)

    @given(disjoint_named_axis_tuple_pairs())
    def test_preserves_order_a_then_b(self, pair):
        a, b = pair
        assert concatenate_axes(a, b) == a + b

    @given(disjoint_named_axis_tuples(min_size=1))
    def test_sharing_every_name_with_itself_always_raises(self, a):
        with pytest.raises(ValueError, match="sharing names"):
            concatenate_axes(a, a)


class TestRemoveAxes:
    @pytest.mark.parametrize(
        "axes,selector,expected",
        [
            ((X4, Y8), "x", (Y8,)),
            ((X4, Y8, Z16), ["x", "z"], (Y8,)),
            ((X4, Y8, Z16), ["z", "x"], (Y8,)),
        ],
        ids=["single_name", "multiple", "order_preserved_regardless_of_selector_order"],
    )
    def test_remove(self, axes, selector, expected):
        assert remove_axes(axes, selector) == expected

    # "raises on an unresolved selector" -- see TestUnresolvedSelectorRaises.


class TestReplaceAxes:
    def test_replace_single_axis_same_arity(self):
        axes = (Axis(4, "x"), Axis(8, "y"))
        result = replace_axes(axes, "x", Axis(16, "z"))
        assert result == (Axis(16, "z"), Axis(8, "y"))

    def test_replace_preserves_position(self):
        axes = (Axis(4, "x"), Axis(8, "y"), Axis(16, "z"))
        result = replace_axes(axes, "y", Axis(2, "w"))
        assert result == (Axis(4, "x"), Axis(2, "w"), Axis(16, "z"))

    def test_duplicate_names_in_replacement_raises(self):
        axes = (Axis(4, "x"),)
        with pytest.raises(ValueError, match="duplicate"):
            replace_axes(axes, "x", [Axis(2, "a"), Axis(2, "a")])

    def test_replacement_name_colliding_with_retained_axis_raises(self):
        axes = (Axis(4, "x"), Axis(8, "y"))
        with pytest.raises(ValueError, match="collides"):
            replace_axes(axes, "x", Axis(2, "y"))

    def test_replacement_reusing_its_own_removed_name_is_fine(self):
        axes = (Axis(4, "x"), Axis(8, "y"))
        result = replace_axes(axes, "x", Axis(2, "x"))
        assert result == (Axis(2, "x"), Axis(8, "y"))

    # "raises on an unresolved selector" -- see TestUnresolvedSelectorRaises.

    def test_mismatched_arity_raises(self):
        axes = (Axis(4, "x"),)
        with pytest.raises(ValueError):
            replace_axes(axes, "x", [Axis(2, "a"), Axis(2, "b")])


class TestUnresolvedSelectorRaises:
    """`remove_axes`, `replace_axes`, and `resolve_axes` all raise
    `ValueError` when a selector doesn't resolve against the given axes --
    the same contract, previously verified separately (and identically) in
    each function's own test class."""

    @pytest.mark.parametrize(
        "op",
        [
            lambda axes: remove_axes(axes, "nope"),
            lambda axes: replace_axes(axes, "nope", Axis(2, "z")),
            lambda axes: resolve_axes(axes, ["nope"]),
        ],
        ids=["remove_axes", "replace_axes", "resolve_axes"],
    )
    def test_raises_on_unresolved_selector(self, op):
        with pytest.raises(ValueError):
            op((X4,))


class TestMakeAxes:
    def test_builds_named_axes_from_kwargs(self):
        assert make_axes(x=4, y=8) == (Axis(4, "x"), Axis(8, "y"))

    def test_empty_kwargs(self):
        assert make_axes() == ()

    def test_preserves_kwarg_order(self):
        result = make_axes(z=1, a=2)
        assert axis_names(result) == ("z", "a")
