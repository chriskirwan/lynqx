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


@pytest.fixture
def x4():
    return Axis(4, "x")


@pytest.fixture
def x8():
    return Axis(8, "x")


@pytest.fixture
def y8():
    return Axis(8, "y")


@pytest.fixture
def z16():
    return Axis(16, "z")


@pytest.fixture
def standard_selection():
    return (Axis(4, "batch"), Axis(8, "head"), Axis(16))


# ---------------------------------------------------------------------------
# Compatibility
# ---------------------------------------------------------------------------


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

    @given(axes_strategy, axes_strategy)
    def test_symmetric(self, a, b):
        assert is_axis_compatible(a, b) == is_axis_compatible(b, a)

    @given(axes_strategy)
    def test_reflexive(self, a):
        assert is_axis_compatible(a, a) is True


# ---------------------------------------------------------------------------
# Unique Axis Checking & Validation
# ---------------------------------------------------------------------------


class TestCheckUniqueAxisNames:
    def test_allows_disjoint_named_axes(self, x4, y8):
        check_unique_axis_names((x4, y8))

    def test_allows_repeated_anonymous_axes(self):
        check_unique_axis_names((Axis(4), Axis(4), Axis(8)))

    def test_allows_bare_string_specs_with_distinct_names(self):
        check_unique_axis_names(("x", "y"))

    def test_bare_string_and_axis_object_with_same_name_collide(self):
        with pytest.raises(ValueError, match="'x'"):
            check_unique_axis_names(("x", Axis(4, "x")))

    def test_duplicate_named_axis_raises(self, x4, x8):
        with pytest.raises(ValueError, match="'x'"):
            check_unique_axis_names((x4, x8))

    def test_duplicate_error_lists_every_colliding_index(self, x4, y8):
        axes = (x4, y8, Axis(2, "x"), Axis(1, "x"))
        with pytest.raises(ValueError, match=r"'x' at \[0, 2, 3\]"):
            check_unique_axis_names(axes)

    def test_multiple_different_duplicate_names_all_reported(self, x4, x8, y8):
        axes = (x4, x8, y8, Axis(1, "y"))
        with pytest.raises(ValueError) as exc_info:
            check_unique_axis_names(axes)
        assert "'x'" in str(exc_info.value)
        assert "'y'" in str(exc_info.value)

    def test_single_bare_axis_is_normalized_to_a_one_tuple(self, x4):
        check_unique_axis_names(x4)

    def test_single_bare_string_is_normalized_to_a_one_tuple(self):
        check_unique_axis_names("xx")

    def test_non_iterable_scalar_raises_type_error(self):
        with pytest.raises(TypeError):
            check_unique_axis_names(5)

    def test_non_axislike_element_raises_type_error_naming_its_index(self, x4):
        with pytest.raises(TypeError, match="index 1"):
            check_unique_axis_names([x4, 5])

    def test_bytes_input_raises_type_error(self):
        with pytest.raises(TypeError):
            check_unique_axis_names(b"ab")


class TestCheckUniqueAxisNamesProperties:
    @given(disjoint_named_axis_tuples())
    def test_never_raises_on_disjoint_named_axes(self, axes):
        check_unique_axis_names(axes)

    @given(duplicate_named_axis_pair(), disjoint_named_axis_tuples())
    def test_always_raises_on_a_duplicate_pair_anywhere_in_the_input(self, pair, rest):
        axes = pair + rest
        with pytest.raises(ValueError):
            check_unique_axis_names(axes)

    @given(st.integers(min_value=0, max_value=5))
    def test_any_number_of_anonymous_axes_never_collide(self, n):
        check_unique_axis_names(tuple(Axis(4) for _ in range(n)))


class TestValidateUniqueAxes:
    def test_decorator_factory_form_passes_valid_input_through(self, x4, y8):
        @validate_unique_axes(arg_names=("axes",))
        def fn(axes):
            return axes

        assert fn((x4, y8)) == (x4, y8)

    def test_decorator_factory_form_raises_on_duplicate(self, x4, x8):
        @validate_unique_axes(arg_names=("axes",))
        def fn(axes):
            return axes

        with pytest.raises(ValueError):
            fn((x4, x8))

    def test_direct_call_form_is_equivalent_to_decorator_form(self, x4, x8, y8):
        def fn(axes):
            return axes

        wrapped = validate_unique_axes(fn, arg_names=("axes",))

        assert wrapped((x4, y8)) == (x4, y8)
        with pytest.raises(ValueError):
            wrapped((x4, x8))

    def test_only_the_named_parameters_are_validated(self, x4):
        @validate_unique_axes(arg_names=("a",))
        def fn(a, b):
            return a, b

        assert fn((x4,), (x4, x4)) == ((x4,), (x4, x4))

    def test_each_named_parameter_is_checked_independently(self, x4, x8, y8):
        @validate_unique_axes(arg_names=("a", "b"))
        def fn(a, b):
            return a, b

        with pytest.raises(ValueError):
            fn((x4, x8), (y8,))
        with pytest.raises(ValueError):
            fn((x4,), (y8, y8))

    def test_works_with_positional_and_keyword_calls(self, x4, x8):
        @validate_unique_axes(arg_names=("axes",))
        def fn(x, axes):
            return x, axes

        with pytest.raises(ValueError):
            fn(1, (x4, x8))
        with pytest.raises(ValueError):
            fn(1, axes=(x4, x8))
        with pytest.raises(ValueError):
            fn(x=1, axes=(x4, x8))

    def test_missing_value_is_skipped_not_validated(self):
        @validate_unique_axes(arg_names=("axes",))
        def fn(axes=None):
            return axes

        assert fn() is None
        assert fn(axes=None) is None

    def test_typo_in_arg_names_silently_skips_validation(self, x4, x8):
        @validate_unique_axes(arg_names=("axess",))
        def fn(axes):
            return axes

        result = fn((x4, x8))
        assert result == (x4, x8)

    def test_wraps_preserves_name_and_docstring(self):
        @validate_unique_axes(arg_names=("axes",))
        def some_fn(axes):
            """Some docstring."""
            return axes

        assert some_fn.__name__ == "some_fn"
        assert some_fn.__doc__ == "Some docstring."


# ---------------------------------------------------------------------------
# Normalization & Spec Conversion (Consolidated)
# ---------------------------------------------------------------------------


class TestAxisNormalization:
    @pytest.mark.parametrize(
        "func, input_spec, expected",
        [
            (axis_spec_to_tuple, "batch", ("batch",)),
            (axis_spec_to_tuple, Axis(4, "batch"), (Axis(4, "batch"),)),
            (axis_spec_to_tuple, ["batch", Axis(4, "batch")], ("batch", Axis(4, "batch"))),
            (axis_spec_to_tuple, [], ()),
            (axis_shape_to_tuple, Axis(4, "x"), (Axis(4, "x"),)),
            (axis_shape_to_tuple, (Axis(4, "x"), Axis(8)), (Axis(4, "x"), Axis(8))),
            (axis_shape_to_tuple, {"x": 4, "y": 8}, (Axis(4, "x"), Axis(8, "y"))),
            (axis_selection_to_tuple, "batch", ("batch",)),
            (axis_selection_to_tuple, 0, (0,)),
            (axis_selection_to_tuple, ["batch", 1], ("batch", 1)),
            (axis_selection_to_tuple, None, ()),
        ],
        ids=[
            "spec_single_string",
            "spec_single_axis_object",
            "spec_iterable_mixed",
            "spec_empty",
            "shape_single_axis",
            "shape_iterable_axes",
            "shape_mapping",
            "selection_string",
            "selection_int",
            "selection_list",
            "selection_none",
        ],
    )
    def test_normalizes_to_tuple(self, func, input_spec, expected):
        assert func(input_spec) == expected

    def test_spec_generator_consumed_into_tuple(self):
        gen = (n for n in ("a", "b"))
        assert axis_spec_to_tuple(gen) == ("a", "b")

    def test_shape_non_axis_member_raises_type_error(self):
        with pytest.raises(TypeError):
            axis_shape_to_tuple(["not_an_axis"])


# ---------------------------------------------------------------------------
# Axis Properties & Accessors
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Axis Index & Resolution
# ---------------------------------------------------------------------------


class TestAxisIndex:
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
    def test_lookup(self, standard_selection, selector, expected):
        assert axis_index(standard_selection, selector) == expected

    def test_find_by_axis_value(self, standard_selection):
        assert axis_index(standard_selection, standard_selection[0]) == 0

    def test_ambiguous_name_raises(self):
        selection = (Axis(4, "x"), Axis(8, "x"))
        with pytest.raises(ValueError, match="abmigious|ambiguous"):
            axis_index(selection, "x")

    def test_none_selector_returns_none(self, standard_selection):
        assert axis_index(standard_selection, None) is None


class TestAxisIndices:
    def test_all_resolved(self, standard_selection):
        assert axis_indices(standard_selection, ["head", "batch"]) == (1, 0)

    def test_strict_raises_on_unresolved(self, standard_selection):
        with pytest.raises(ValueError):
            axis_indices(standard_selection, ["nope"], strict=True)

    def test_non_strict_returns_none_for_unresolved(self, standard_selection):
        assert axis_indices(standard_selection, ["nope"], strict=False) == (None,)

    def test_default_is_strict(self, standard_selection):
        with pytest.raises(ValueError):
            axis_indices(standard_selection, ["nope"])

    def test_none_selection_returns_empty_tuple(self, standard_selection):
        assert axis_indices(standard_selection, None) == ()

    def test_none_selection_is_not_strict_checked(self, standard_selection):
        # Even with strict=True (the default), None means "no selectors were
        # given" -- there's nothing to fail to resolve.
        assert axis_indices(standard_selection, None, strict=True) == ()


class TestResolveAxes:
    def test_resolves_to_axis_objects(self):
        x, y = Axis(4, "x"), Axis(8, "y")
        assert resolve_axes((x, y), ["y", "x"]) == (y, x)

    def test_none_selector_resolves_to_no_axes(self, x4, y8):
        assert resolve_axes((x4, y8), None) == ()


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
        assert AxisMatch(0, 1) in result.matches
        assert AxisMatch(1, 0) in result.matches

    def test_returns_match_axis_result_namedtuple(self):
        result = match_axes((), ())
        assert isinstance(result, MatchAxisResult)
        assert result == MatchAxisResult((), (), ())


# ---------------------------------------------------------------------------
# Set-Like Axis Operations
# ---------------------------------------------------------------------------


class TestUnionAxes:
    def test_disjoint_names_appended(self, x4, y8):
        assert union_axes((x4,), (y8,)) == (x4, y8)

    def test_shared_name_deduplicated(self, x4):
        assert union_axes((x4,), (x4,)) == (x4,)

    def test_anonymous_axes_from_b_always_appended(self, x4):
        assert union_axes((x4,), (Axis(8), Axis(8))) == (x4, Axis(8), Axis(8))

    def test_order_is_a_then_new_from_b(self, x4, y8, z16):
        a = (x4, y8)
        b = (y8, z16)
        assert union_axes(a, b) == (x4, y8, z16)


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
    def test_only_shared_names_kept_in_a_order(self, x4, y8, z16):
        a = (x4, y8)
        b = (y8, z16)
        assert intersect_axes(a, b) == (y8,)

    def test_anonymous_axes_never_kept(self, y8):
        a = (Axis(4), y8)
        b = (Axis(4), y8)
        assert intersect_axes(a, b) == (y8,)

    def test_no_overlap_returns_empty(self, x4, y8):
        assert intersect_axes((x4,), (y8,)) == ()


class TestSetLikeAxisOpsShared:
    @pytest.mark.parametrize("op", [union_axes, intersect_axes], ids=["union", "intersect"])
    def test_shared_name_conflicting_size_raises(self, op, x4, x8):
        with pytest.raises(ValueError, match="conflicting sizes"):
            op((x4,), (x8,))


class TestConcatenateAxes:
    @pytest.mark.parametrize(
        "a,b,expected",
        [
            ((Axis(4, "x"),), (Axis(8, "y"),), (Axis(4, "x"), Axis(8, "y"))),
            ((Axis(4),), (Axis(4),), (Axis(4), Axis(4))),
        ],
        ids=["simple_concat", "anonymous_axes_never_clash"],
    )
    def test_concat(self, a, b, expected):
        assert concatenate_axes(a, b) == expected

    def test_shared_name_raises(self, x4):
        with pytest.raises(ValueError, match="sharing names"):
            concatenate_axes((x4,), (x4,))


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


# ---------------------------------------------------------------------------
# Axis Manipulation Operations
# ---------------------------------------------------------------------------


class TestRemoveAxes:
    @pytest.mark.parametrize(
        "axes,selector,expected",
        [
            ((Axis(4, "x"), Axis(8, "y")), "x", (Axis(8, "y"),)),
            ((Axis(4, "x"), Axis(8, "y"), Axis(16, "z")), ["x", "z"], (Axis(8, "y"),)),
            ((Axis(4, "x"), Axis(8, "y"), Axis(16, "z")), ["z", "x"], (Axis(8, "y"),)),
        ],
        ids=["single_name", "multiple", "order_preserved_regardless_of_selector_order"],
    )
    def test_remove(self, axes, selector, expected):
        assert remove_axes(axes, selector) == expected

    def test_none_selector_is_a_no_op(self, x4, y8):
        axes = (x4, y8)
        assert remove_axes(axes, None) == axes


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

    def test_mismatched_arity_raises(self):
        axes = (Axis(4, "x"),)
        with pytest.raises(ValueError):
            replace_axes(axes, "x", [Axis(2, "a"), Axis(2, "b")])

    def test_none_selector_is_a_no_op_ignoring_new(self, x4, y8):
        axes = (x4, y8)
        # `new` would be a mismatched-arity replacement if `old` weren't None --
        # confirms `new` is genuinely ignored, not just "happens to line up".
        assert replace_axes(axes, None, [Axis(2, "a"), Axis(2, "b"), Axis(2, "c")]) == axes


class TestUnresolvedSelectorRaises:
    @pytest.mark.parametrize(
        "func, extra_args",
        [
            (remove_axes, ()),
            (replace_axes, (Axis(2, "z"),)),
            (resolve_axes, ()),
        ],
        ids=["remove_axes", "replace_axes", "resolve_axes"],
    )
    def test_raises_on_unresolved_selector(self, func, extra_args, x4):
        # Passes trailing positional args cleanly without needing inline lambdas
        args = ("nope",) + extra_args if func != resolve_axes else (["nope"],)
        with pytest.raises(ValueError):
            func((x4,), *args)


class TestMakeAxes:
    def test_builds_named_axes_from_kwargs(self):
        assert make_axes(x=4, y=8) == (Axis(4, "x"), Axis(8, "y"))

    def test_empty_kwargs(self):
        assert make_axes() == ()

    def test_preserves_kwarg_order(self):
        result = make_axes(z=1, a=2)
        assert axis_names(result) == ("z", "a")
