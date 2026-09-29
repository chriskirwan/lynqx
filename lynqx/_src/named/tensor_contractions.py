from collections.abc import Sequence
from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp
from jax.lax import dot as lax_dot, PrecisionLike
from jax.typing import DTypeLike

from lynqx._src.axis_util import (
    axis_indices,
    axis_names,
    axis_selection_to_tuple,
    match_axes,
    remove_axes,
)
from lynqx._src.named import constructors, util
from lynqx._src.sharding import canonicalize_sharding
from lynqx._src.typing import Axis, AxisSelector, NamedArray, NamedArrayLike, ShardingLike


MatrixSpec = tuple[AxisSelector, AxisSelector]

DotContraction = tuple[
    tuple[Sequence[AxisSelector], Sequence[AxisSelector]],  # (lhs_contract, rhs_contract)
    tuple[Sequence[AxisSelector], Sequence[AxisSelector]],  # (lhs_batch, rhs_batch)
]


class TensorContraction(NamedTuple):
    """Axis specification for binary tensor contractions"""

    lhs_contract: Sequence[AxisSelector]
    rhs_contract: Sequence[AxisSelector]
    lhs_batch: Sequence[AxisSelector] = ()
    rhs_batch: Sequence[AxisSelector] = ()

    def to_dot_axes(self) -> DotContraction:
        return (
            (self.lhs_contract, self.rhs_contract),
            (self.lhs_batch, self.rhs_batch),
        )


def _parse_dot_axes(
    lhs_axes: tuple[Axis, ...],
    rhs_axes: tuple[Axis, ...],
    axis_spec: TensorContraction,
):
    lhs_contract, rhs_contract, lhs_batch, rhs_batch = axis_spec
    lhs_contract = axis_selection_to_tuple(lhs_contract)
    rhs_contract = axis_selection_to_tuple(rhs_contract)
    lhs_batch = axis_selection_to_tuple(lhs_batch)
    rhs_batch = axis_selection_to_tuple(rhs_batch)

    if len(lhs_contract) != len(rhs_contract):
        raise ValueError(
            f"lhs and rhs contracting axes must be the same length, got {len(lhs_contract)} vs {len(rhs_contract)}"
        )
    if len(lhs_batch) != len(rhs_batch):
        raise ValueError(f"lhs and rhs batch axes must be the same length, got {len(lhs_batch)} vs {len(rhs_batch)}")

    def _to_dim_tuple(
        axes_spec: tuple[Axis, ...],
        sel: tuple[AxisSelector, ...],
    ) -> tuple[int, ...]:
        if not sel:
            return ()
        raw = axis_indices(axes_spec, sel)
        if any(i is None for i in raw):
            missing = [s for s, i in zip(sel, raw) if i is None]
            raise AssertionError(f"failed to resolve indices for {missing} — this is a bug.")
        return tuple(raw)

    lhs_contract_dims = _to_dim_tuple(lhs_axes, lhs_contract)
    rhs_contract_dims = _to_dim_tuple(rhs_axes, rhs_contract)
    lhs_batch_dims = _to_dim_tuple(lhs_axes, lhs_batch)
    rhs_batch_dims = _to_dim_tuple(rhs_axes, rhs_batch)

    resolved_lhs_contract = tuple(lhs_axes[i] for i in lhs_contract_dims)
    resolved_rhs_contract = tuple(rhs_axes[i] for i in rhs_contract_dims)
    resolved_lhs_batch = tuple(lhs_axes[i] for i in lhs_batch_dims)
    resolved_rhs_batch = tuple(rhs_axes[i] for i in rhs_batch_dims)

    for lcx, rcx in zip(resolved_lhs_contract, resolved_rhs_contract):
        if lcx.size != rcx.size:
            raise ValueError(f"Contracting axis size mismatch: lhs `{lcx}` vs rhs `{rcx}`")
    for lbx, rbx in zip(resolved_lhs_batch, resolved_rhs_batch):
        if lbx.size != rbx.size:
            raise ValueError(f"Batch axis size mismatch: lhs `{lbx}` vs rhs `{rbx}`")

    lhs_batch_names = axis_names(resolved_lhs_batch)
    rhs_batch_names = axis_names(resolved_rhs_batch)
    for lname, rname in zip(lhs_batch_names, rhs_batch_names):
        if lname is not None and rname is not None and lname != rname:
            raise ValueError(
                f"Batch axis names must match on both sides, "
                f"got lhs: `{lhs_batch_names}` vs rhs: `{rhs_batch_names}`. "
                "Consider renaming one side before calling `dot`."
            )
    if overlap := set(lhs_contract_dims) & set(lhs_batch_dims):
        raise ValueError(f"lhs axis positions appear in both contracting and batch: {sorted(overlap)}")
    if overlap := set(rhs_contract_dims) & set(rhs_batch_dims):
        raise ValueError(f"rhs axis positions appear in both contracting and batch: {sorted(overlap)}")

    dimension_numbers = (
        (lhs_contract_dims, rhs_contract_dims),
        (lhs_batch_dims, rhs_batch_dims),
    )

    used_lhs_idx = set(lhs_contract_dims) | set(lhs_batch_dims)
    used_rhs_idx = set(rhs_contract_dims) | set(rhs_batch_dims)

    batch_out: tuple[Axis, ...] = resolved_lhs_batch
    free_lhs: tuple[Axis, ...] = tuple(ax for i, ax in enumerate(lhs_axes) if i not in used_lhs_idx)
    free_rhs: tuple[Axis, ...] = tuple(ax for i, ax in enumerate(rhs_axes) if i not in used_rhs_idx)
    output_axes: tuple[Axis, ...] = batch_out + free_lhs + free_rhs

    named_output = [ax.name for ax in output_axes if ax.name is not None]
    if len(named_output) != len(set(named_output)):
        seen: set[str] = set()
        duplicates: set[str] = set()
        for n in named_output:
            (duplicates if n in seen else seen).add(n)
        raise ValueError(
            f"Output axis name collision — the following names appear in both "
            f"the lhs and rhs free axes: {duplicates}. "
            "Rename one side before calling `dot`."
        )

    return dimension_numbers, output_axes


@eqx.filter_jit
def _dot(
    lhs: NamedArray,
    rhs: NamedArray,
    spec: TensorContraction,
    precision: PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    dimension_numbers, out_axes = _parse_dot_axes(lhs.axes, rhs.axes, spec)

    canonical_sharding = canonicalize_sharding(out_axes, out_sharding, "dot")
    jax_array = lax_dot(
        lhs.array,
        rhs.array,
        dimension_numbers=dimension_numbers,
        precision=precision,
        preferred_element_type=preferred_element_type,
        out_sharding=canonical_sharding,
    )

    return constructors.array(jax_array, out_axes)


def dot(
    lhs: NamedArrayLike,
    rhs: NamedArrayLike,
    spec: DotContraction,
    precision: PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """General binary contraction."""
    lhs, rhs = util.ensure_named("dot", lhs, rhs)

    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = spec
    contraction_spec = TensorContraction(lhs_contract, rhs_contract, lhs_batch, rhs_batch)

    return _dot(lhs, rhs, contraction_spec, precision, preferred_element_type, out_sharding=out_sharding)


def _infer_batch_axes(
    lhs: NamedArray,
    rhs: NamedArray,
    *,
    lhs_exclude: int,
    rhs_exclude: int,
    allow_positional_fallback: bool = True,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Infer implicit batch axis positions for a single-axis contraction.

    Delegates to `match_axes`: named axes shared between `lhs` and `rhs`
    (other than the axis already claimed by contraction on each side) are
    matched by name; with `allow_positional_fallback` (the default, matching
    the right-aligned convention used for other binary ops), any remaining
    anonymous axes are right-aligned and matched by position too.

    Returns index pairs into the *original* `lhs.axes` / `rhs.axes`.
    """
    lhs_axes = lhs.axes
    rhs_axes = rhs.axes

    lhs_candidate_idx = [i for i in range(len(lhs_axes)) if i != lhs_exclude]
    rhs_candidate_idx = [i for i in range(len(rhs_axes)) if i != rhs_exclude]

    lhs_candidates = tuple(lhs_axes[i] for i in lhs_candidate_idx)
    rhs_candidates = tuple(rhs_axes[i] for i in rhs_candidate_idx)

    # First perform named matching via match_axes without positional fallback
    named_result = match_axes(lhs_candidates, rhs_candidates, allow_positional_fallback=False)

    matched_lhs_cand = {m.source for m in named_result.matches}
    matched_rhs_cand = {m.target for m in named_result.matches}

    lhs_batch_list = [lhs_candidate_idx[m.source] for m in named_result.matches]
    rhs_batch_list = [rhs_candidate_idx[m.target] for m in named_result.matches]

    if allow_positional_fallback:
        # Map original distance from the end (len - 1 - original_index) for anonymous candidates
        lhs_anon_by_dist = {
            len(lhs_axes) - 1 - lhs_candidate_idx[c_idx]: lhs_candidate_idx[c_idx]
            for c_idx in range(len(lhs_candidates))
            if c_idx not in matched_lhs_cand and lhs_candidates[c_idx].name is None
        }
        rhs_anon_by_dist = {
            len(rhs_axes) - 1 - rhs_candidate_idx[c_idx]: rhs_candidate_idx[c_idx]
            for c_idx in range(len(rhs_candidates))
            if c_idx not in matched_rhs_cand and rhs_candidates[c_idx].name is None
        }

        # Match anonymous axes that share the exact same trailing distance in the original tuples
        common_distances = set(lhs_anon_by_dist.keys()) & set(rhs_anon_by_dist.keys())
        for dist in sorted(common_distances):
            lhs_batch_list.append(lhs_anon_by_dist[dist])
            rhs_batch_list.append(rhs_anon_by_dist[dist])

    batch_pairs = sorted(zip(lhs_batch_list, rhs_batch_list))

    return (
        tuple(lhs_idx for lhs_idx, _ in batch_pairs),
        tuple(rhs_idx for _, rhs_idx in batch_pairs),
    )


def vecdot(
    lhs: NamedArrayLike,
    rhs: NamedArrayLike,
    axis: AxisSelector,
    precision: PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    lhs, rhs = util.ensure_named("vecdot", lhs, rhs)

    if isinstance(axis, tuple):
        lhs_axis, rhs_axis = axis
    else:
        lhs_axis = rhs_axis = axis

    lhs_idx = axis_indices(lhs.axes, (lhs_axis,))[0]
    rhs_idx = axis_indices(rhs.axes, (rhs_axis,))[0]
    lhs_batch_idx, rhs_batch_idx = _infer_batch_axes(lhs, rhs, lhs_exclude=lhs_idx, rhs_exclude=rhs_idx)

    contraction_spec = TensorContraction((lhs_axis,), (rhs_axis,), lhs_batch_idx, rhs_batch_idx)
    return _dot(lhs, rhs, contraction_spec, precision, preferred_element_type, out_sharding=out_sharding)


def matmul(
    lhs: NamedArrayLike,
    rhs: NamedArrayLike,
    contract: tuple[tuple[AxisSelector], tuple[AxisSelector]],
    precision: PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    lhs, rhs = util.ensure_named("matmul", lhs, rhs)
    lhs_contract, rhs_contract = contract

    lhs_idx = axis_indices(lhs.axes, lhs_contract)[0]
    rhs_idx = axis_indices(rhs.axes, rhs_contract)[0]
    lhs_batch_idx, rhs_batch_idx = _infer_batch_axes(lhs, rhs, lhs_exclude=lhs_idx, rhs_exclude=rhs_idx)

    contraction_spec = TensorContraction(lhs_contract, rhs_contract, lhs_batch_idx, rhs_batch_idx)
    return _dot(lhs, rhs, contraction_spec, precision, preferred_element_type, out_sharding=out_sharding)


def outer(
    lhs: NamedArrayLike,
    rhs: NamedArrayLike,
    precision: PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
    *,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """."""
    lhs, rhs = util.ensure_named("outer", lhs, rhs)
    contraction_spec = TensorContraction((), (), (), ())
    return _dot(lhs, rhs, contraction_spec, precision, preferred_element_type, out_sharding=out_sharding)


def trace(
    a: NamedArrayLike,
    offset: int = 0,
    axis1: AxisSelector = 0,
    axis2: AxisSelector = 1,
    dtype: DTypeLike | None = None,
) -> NamedArray:
    a = util.ensure_named("trace", a)

    idx1, idx2 = axis_indices(a.axes, (axis1, axis2))
    if idx1 == idx2:
        raise ValueError("Cannot trace along the same named axes")

    jax_array = jnp.trace(a.array, offset, idx1, idx2, dtype=dtype)
    out_axes = remove_axes(a.axes, (idx1, idx2))
    return constructors.array(jax_array, out_axes)
