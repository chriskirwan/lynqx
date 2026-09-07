import jax.lax
import jax.numpy as jnp

from lynqx._src.axis_util import axis_shape_to_tuple, axis_sizes, match_axes, validate_unique_axes
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

    matched = match_axes(
        named.axes,
        target,
        allow_positional_fallback=True,
    )

    if matched.unmatched_source:
        missing = [named.axes[i] for i in matched.unmatched_source]
        raise ValueError(f"broadcast_to cannot drop axes {missing}; every existing axis must appear in `shape`")

    # Validate that every matched axis is broadcast-compatible.
    for m in matched.matches:
        src_ax = named.axes[m.source]
        tgt_ax = target[m.target]

        if src_ax.size != tgt_ax.size and src_ax.size != 1:
            raise ValueError(f"Cannot broadcast axis {src_ax.name!r} of size {src_ax.size} to {tgt_ax.size}")

    target_sizes = axis_sizes(target)
    jax_sharding = canonicalize_sharding(
        out_sharding,
        target,
        "broadcast_to",
    )

    if named.array.ndim == 0:
        jax_array = jnp.broadcast_to(
            named.array,
            target_sizes,
            out_sharding=jax_sharding,
        )
        return constructors.array(jax_array, target)

    broadcast_dimensions = tuple(m.target for m in matched.matches)

    # If source axes are not already in target order, transpose first.
    source_order = tuple(m.source for m in sorted(matched.matches, key=lambda m: m.target))
    identity = tuple(range(named.array.ndim))

    if source_order != identity:
        array = jnp.transpose(named.array, source_order)

        # After transposing, matches are in target order.
        broadcast_dimensions = tuple(m.target for m in sorted(matched.matches, key=lambda m: m.target))
    else:
        array = named.array

    jax_array = jax.lax.broadcast_in_dim(
        array,
        target_sizes,
        broadcast_dimensions=broadcast_dimensions,
    )

    return constructors.array(jax_array, target, out_sharding=out_sharding)


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

    target_shape = named_arrays[0].axes
    for arr in named_arrays[1:]:
        _, _, target_shape = util.align_shapes_for_broadcast(target_shape, arr.axes)

    broadcasted_arrays = tuple(broadcast_to(arr, target_shape, out_sharding=out_sharding) for arr in named_arrays)
    return broadcasted_arrays


def broadcast_shapes(*shapes: AxisShape) -> AxisShape:
    """Broadcast multiple shapes to a common shape.

    Args:
        *shapes: The shapes to broadcast.

    Returns:
        The common broadcasted shape.
    """
    if not shapes:
        return ()

    target_shape = axis_shape_to_tuple(shapes[0])
    for shape in shapes[1:]:
        _, _, target_shape = util.align_shapes_for_broadcast(target_shape, axis_shape_to_tuple(shape))

    return target_shape
