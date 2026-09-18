from hypothesis import given, strategies as st
from jax import P
from lynqx._src.axis import Axis
from lynqx._src.partition import PM
from strategies import (
    axes as axis_strategy,
    axis_names as name_strategy,
)


class TestPMConstruction:
    def test_default_is_empty(self):
        pm = PM()
        assert pm.get_logical_axes() == ()
        assert pm.partition == ()

    def test_mapping_and_partition_stored(self):
        pm = PM({"batch": "data"}, ("model",))
        assert pm.get_logical_axes() == ("batch",)
        assert pm.partition == ("model",)

    def test_mapping_is_defensively_copied(self):
        mapping = {"batch": "data"}
        pm = PM(mapping)
        mapping["batch"] = "mutated"
        assert pm.get_physical_axis("batch") == "data"

    def test_partition_coerced_to_tuple(self):
        pm = PM(partition=["a", "b"])
        assert pm.partition == ("a", "b")
        assert isinstance(pm.partition, tuple)


class TestGetPhysicalAxis:
    def test_string_key_direct_hit(self):
        pm = PM({"batch": "data"})
        assert pm.get_physical_axis("batch") == "data"

    def test_axis_object_used_as_dict_key_direct_hit(self):
        ax = Axis(4, "batch")
        pm = PM({ax: "data"})
        assert pm.get_physical_axis(ax) == "data"

    def test_axis_key_resolved_by_string_lookup(self):
        pm = PM({Axis(4, "batch"): "data"})
        assert pm.get_physical_axis("batch") == "data"

    def test_axis_key_resolved_by_different_axis_instance_with_same_name(self):
        pm = PM({Axis(4, "batch"): "data"})
        assert pm.get_physical_axis(Axis(8, "batch")) == "data"

    def test_named_axis_falls_back_to_name_lookup(self):
        pm = PM({"batch": "data"})
        assert pm.get_physical_axis(Axis(4, "batch")) == "data"

    def test_anonymous_axis_is_never_mapped(self):
        pm = PM({"batch": "data"})
        assert pm.get_physical_axis(Axis(4)) is None

    def test_unmapped_name_returns_none(self):
        pm = PM({"batch": "data"})
        assert pm.get_physical_axis("head") is None

    def test_explicit_none_mapping_is_returned_as_none(self):
        pm = PM({"batch": None})
        assert pm.get_physical_axis("batch") is None


class TestIsMapped:
    def test_true_for_direct_string_key(self):
        pm = PM({"batch": "data"})
        assert pm.is_mapped("batch") is True

    def test_true_for_string_lookup_when_keyed_by_axis_object(self):
        pm = PM({Axis(4, "batch"): "data"})
        assert pm.is_mapped("batch") is True

    def test_true_for_axis_object_resolved_by_name(self):
        pm = PM({"batch": "data"})
        assert pm.is_mapped(Axis(4, "batch")) is True

    def test_true_for_different_axis_instance_when_keyed_by_axis_object(self):
        pm = PM({Axis(4, "batch"): "data"})
        assert pm.is_mapped(Axis(8, "batch")) is True

    def test_true_even_when_mapped_to_none(self):
        # Being present (even mapped to None, i.e. explicitly replicated) is
        # distinct from not being mapped at all.
        pm = PM({"batch": None})
        assert pm.is_mapped("batch") is True

    def test_false_for_anonymous_axis(self):
        pm = PM({"batch": "data"})
        assert pm.is_mapped(Axis(4)) is False

    def test_false_for_unmapped_name(self):
        pm = PM()
        assert pm.is_mapped("batch") is False


class TestWithMappings:
    def test_adds_new_entries(self):
        pm = PM({"batch": "data"})
        pm2 = pm.with_mappings(head="model")
        assert pm2.get_logical_axes() == ("batch", "head")

    def test_original_is_untouched(self):
        pm = PM({"batch": "data"})
        pm.with_mappings(head="model")
        assert pm.get_logical_axes() == ("batch",)

    def test_overrides_existing_entry(self):
        pm = PM({"batch": "data"})
        pm2 = pm.with_mappings(batch="other")
        assert pm2.get_physical_axis("batch") == "other"

    def test_preserves_partition(self):
        pm = PM({"batch": "data"}, ("model",))
        pm2 = pm.with_mappings(head="model")
        assert pm2.partition == ("model",)

    def test_returns_new_instance(self):
        pm = PM()
        assert pm.with_mappings(x="y") is not pm


class TestWithPartitions:
    def test_replaces_partition_tuple(self):
        pm = PM(partition=("a",))
        pm2 = pm.with_partitions("b", "c")
        assert pm2.partition == ("b", "c")

    def test_is_a_replacement_not_a_merge(self):
        pm = PM(partition=("a", "b"))
        pm2 = pm.with_partitions("c")
        assert pm2.partition == ("c",)

    def test_preserves_mapping(self):
        pm = PM({"batch": "data"}, ("a",))
        pm2 = pm.with_partitions("b")
        assert pm2.get_logical_axes() == ("batch",)

    def test_no_args_clears_partition(self):
        pm = PM(partition=("a", "b"))
        pm2 = pm.with_partitions()
        assert pm2.partition == ()

    def test_original_is_untouched(self):
        pm = PM(partition=("a",))
        pm.with_partitions("b")
        assert pm.partition == ("a",)


class TestEqualityAndHash:
    def test_equal_mapping_and_partition(self):
        a = PM({"batch": "data"}, ("model",))
        b = PM({"batch": "data"}, ("model",))
        assert a == b
        assert hash(a) == hash(b)

    def test_mapping_key_order_irrelevant_to_equality_and_hash(self):
        a = PM({"batch": "data", "head": "model"})
        b = PM({"head": "model", "batch": "data"})
        assert a == b
        assert hash(a) == hash(b)

    def test_different_mapping_value_not_equal(self):
        a = PM({"batch": "data"})
        b = PM({"batch": "other"})
        assert a != b

    def test_different_partition_not_equal(self):
        a = PM(partition=("a",))
        b = PM(partition=("b",))
        assert a != b

    def test_partition_order_matters(self):
        a = PM(partition=("a", "b"))
        b = PM(partition=("b", "a"))
        assert a != b

    def test_not_equal_to_unrelated_type(self):
        pm = PM({"batch": "data"})
        assert pm != {"batch": "data"}
        assert pm != "PM({'batch': 'data'})"

    def test_usable_as_set_member(self):
        a = PM({"batch": "data"})
        b = PM({"batch": "data"})
        c = PM({"batch": "other"})
        assert len({a, b, c}) == 2


class TestPartitionSpec:
    def test_none_axes_returns_empty_spec(self):
        pm = PM({"batch": "data"})
        assert pm.partition_spec(None) == P()

    def test_single_axis_spec_is_normalized(self):
        # A bare string/Axis (not wrapped in a sequence) should be treated
        # like a length-1 axis spec, per `axis_spec_to_tuple`.
        pm = PM({"batch": "data"})
        assert pm.partition_spec("batch") == P("data")

    def test_mapped_named_axis_resolved_by_name_string(self):
        pm = PM({"batch": "data"})
        assert pm.partition_spec(("batch",)) == P("data")

    def test_mapped_named_axis_resolved_from_axis_object(self):
        pm = PM({"batch": "data"})
        assert pm.partition_spec((Axis(4, "batch"),)) == P("data")

    def test_axis_key_resolved_in_partition_spec(self):
        pm = PM({Axis(4, "batch"): "data"})
        assert pm.partition_spec(("batch",)) == P("data")
        assert pm.partition_spec((Axis(8, "batch"),)) == P("data")

    def test_unmapped_named_axis_is_replicated(self):
        pm = PM()
        assert pm.partition_spec(("batch",)) == P(None)

    def test_anonymous_axis_consumes_partition_positionally(self):
        pm = PM(partition=("model",))
        assert pm.partition_spec((Axis(4),)) == P("model")

    def test_anonymous_axes_consumed_in_order(self):
        pm = PM(partition=("a", "b"))
        assert pm.partition_spec((Axis(4), Axis(8))) == P("a", "b")

    def test_anonymous_axis_replicated_once_partition_is_exhausted(self):
        pm = PM(partition=("a",))
        assert pm.partition_spec((Axis(4), Axis(8))) == P("a", None)

    def test_mapped_and_anonymous_axes_can_be_mixed(self):
        pm = PM({"batch": "data"}, partition=("model",))
        assert pm.partition_spec(("batch", Axis(8))) == P("data", "model")

    def test_axis_mapped_to_none_still_consumes_a_positional_slot(self):
        # `get_physical_axis` returns None both when a logical axis is
        # explicitly mapped to None (replicated) and when it isn't mapped at
        # all, so `partition_spec` cannot tell the two apart: an axis
        # mapped to None still falls through to (and consumes) the next
        # slot in the anonymous-axis partition fallback.
        pm = PM({"batch": None}, partition=("model",))
        result = pm.partition_spec(("batch", Axis(8)))
        assert result == P("model", None)

    def test_empty_axes_tuple_returns_empty_spec(self):
        pm = PM({"batch": "data"}, partition=("model",))
        assert pm.partition_spec(()) == P()


class TestPartitionSpecProperties:
    """The example tests above pin down specific resolution rules (mapped
    vs. anonymous, exhausted partition, mixed). The property that should
    hold regardless of *which* rule fires: `partition_spec` never adds or
    drops entries -- one axis in, one PartitionSpec entry out.
    """

    @given(st.lists(axis_strategy, max_size=5), st.lists(name_strategy, max_size=5))
    def test_output_length_always_matches_input_length(self, axis_list, partition):
        pm = PM(partition=tuple(partition))
        result = pm.partition_spec(tuple(axis_list))
        assert len(result) == len(axis_list)

    @given(st.lists(axis_strategy, max_size=5))
    def test_no_partition_means_every_named_axis_is_replicated(self, axis_list):
        # With an empty `partition` fallback, only names actually present
        # in `_mapping` can resolve to something other than None -- and
        # here `_mapping` is empty too.
        pm = PM()
        result = pm.partition_spec(tuple(axis_list))
        assert all(entry is None for entry in result)
