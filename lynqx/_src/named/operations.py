import jax.numpy as jnp

from lynqx._src.axis_util import axis_shape_to_tuple, match_axes, validate_unique_axes
from lynqx._src.named import constructors, util
from lynqx._src.sharding import canonicalize_sharding
from lynqx._src.typing import AxisShape, NamedArrayLike, ShardingLike


# Broadcasting support


@validate_unique_axes(arg_names=("shape",))
def broadcast_to(
    a: NamedArrayLike,
    shape: AxisShape,
    *,
    out_sharding: ShardingLike | None = None,
):
    """Broadcast a `NamedArray` to a specified shape.

    Lynqx uses an extended broadcasting semantics that allows for broadcasting to a shape with different axes, as
    long as the existing axes are preserved and any new axes are added with size 1. This is similar to NumPy's
    broadcasting rules, but with the added constraint that named axes must match.

    Args:
        a: The `NamedArray` to broadcast.
        shape: The target shape to broadcast to.
        out_sharding: Optional sharding specification for the output array. If not specified, it will be determined
            automatically by the compiler.

    Returns:
        A copy of `array` that is broadcasted to the specified shape.
    """
    named = util.ensure_named("broadcast_to", a)
    target = axis_shape_to_tuple(shape)

    matched = match_axes(named.axes, target, allow_positional_fallback=True)
    if matched.unmatched_source:
        missing = [named.axes[i] for i in matched.unmatched_source]
        raise ValueError(f"broadcast_to cannot drop axes {missing}; every existing axis must appear in `shape`")

    for m in matched.matches:
        src_ax, tgt_ax = named.axes[m.source], target[m.target]
        if src_ax.size != tgt_ax.size and src_ax.size != 1:
            raise ValueError(f"Cannot broadcast axis {src_ax.name!r} of size {src_ax.size} to {tgt_ax.size}")

    tgt_to_src = {m.target: m.source for m in matched.matches}

    expanded_shape = []
    for tgt_i in range(len(target)):
        if tgt_i in tgt_to_src:
            expanded_shape.append(named.axes[tgt_to_src[tgt_i]].size)
        else:
            expanded_shape.append(1)

    reshaped_array = jnp.reshape(named.array, tuple(expanded_shape))

    # Broadcast to the full target shape
    target_sizes = tuple(ax.size for ax in target)
    jax_sharding = canonicalize_sharding(out_sharding, target, "broadcast_to")
    jax_array = jnp.broadcast_to(reshaped_array, target_sizes, out_sharding=jax_sharding)

    return constructors.array(jax_array, target)


def broadcast_arrays(*arrays: NamedArrayLike, out_sharding: ShardingLike | None = None):
    """Broadcast multiple NamedArrays to a common shape.

    Args:
        *arrays: The NamedArrays to broadcast.
        out_sharding: Optional sharding specification for the output arrays.

    Returns:
        A tuple of broadcasted NamedArrays.
    """
    named_arrays = util.ensure_named_tuple("broadcast_arrays", arrays)
    if not named_arrays:
        return ()

    # Determine the common target shape
    target_shape = named_arrays[0].axes
    for arr in named_arrays[1:]:
        _, _, target_shape = util.align_shapes_for_broadcast(target_shape, arr.axes)

    # Broadcast each array to the common shape
    broadcasted_arrays = tuple(broadcast_to(arr, target_shape, out_sharding=out_sharding) for arr in named_arrays)
    return broadcasted_arrays
