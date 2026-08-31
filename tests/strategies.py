from hypothesis import strategies as st
from lynqx._src.axis import Axis


axis_names = st.sampled_from(["x", "y", "z", "batch", "link"])
axis_sizes = st.integers(min_value=1, max_value=16)


@st.composite
def named_axes(draw):
    return Axis(draw(axis_sizes), draw(axis_names))


@st.composite
def anonymous_axes(draw):
    return Axis(draw(axis_sizes))


@st.composite
def disjoint_named_axis_tuples(draw, min_size=0, max_size=4):
    """A tuple of named axes with no repeated name -- the precondition
    `concatenate_axes` and friends actually want."""
    names = draw(st.lists(axis_names, min_size=min_size, max_size=max_size, unique=True))
    return tuple(Axis(draw(axis_sizes), name) for name in names)


@st.composite
def disjoint_named_axis_tuple_pairs(draw, max_size=3):
    """Two tuples of named axes, disjoint both internally and from each
    other -- for testing operations like `concatenate_axes`/`union_axes`
    where cross-input disjointness is the interesting precondition.

    """
    names = draw(st.lists(axis_names, min_size=0, max_size=max_size * 2, unique=True))
    split = draw(st.integers(min_value=0, max_value=len(names)))
    a = tuple(Axis(draw(axis_sizes), name) for name in names[:split])
    b = tuple(Axis(draw(axis_sizes), name) for name in names[split:])
    return a, b


@st.composite
def duplicate_named_axis_pair(draw):
    """Two named axes sharing a name -- sizes may or may not match, since
    a duplicate *name* is the violation `check_unique_axis_names` cares
    about, independent of whether the sizes happen to agree (that's a
    separate concern, already covered by union_axes/intersect_axes's own
    "conflicting sizes" checks elsewhere)."""
    name = draw(axis_names)
    return Axis(draw(axis_sizes), name), Axis(draw(axis_sizes), name)


axes = st.one_of(named_axes(), anonymous_axes())
