from collections.abc import Hashable, Sequence
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


_Node = tuple[int, int]  # (operand index, axis position within that operand)

MultiDotAxes = Sequence[tuple[Sequence[AxisSelector], Sequence[AxisSelector]]]


def _chain_sublists(
    operand_axes: tuple[tuple[Axis, ...], ...],
    axes: tuple[tuple[tuple[AxisSelector, ...], tuple[AxisSelector, ...]], ...],
    allow_positional_fallback: bool = True,
) -> tuple[list[list[int]], list[int], tuple[Axis, ...]]:
    """Translate a chain contraction spec into `einsum` sublists, i.e. the interleaved "explicit form via indices".

    Returns ``(sublists, out_sublist, out_axes)``: one list of integer labels per operand, the labels of the output
    and the output axes. Axes carrying the same label are the same index in the contraction.

    ``axes[k]`` contracts operand ``k`` (lhs) with operand ``k + 1`` (rhs). If ``len(axes) == len(operands)``
    the final spec contracts the last operand with the first, closing the chain into a ring (a trace). Both ends of a
    contracted pair share one label that does not appear in the output.

    Every axis not claimed by a contraction is *free*. Free axes that are **named** and share a name across operands
    are matched as batch axes (kept once in the output), mirroring `matmul`/`vecdot`. With
    `allow_positional_fallback`, remaining anonymous free axes are right-aligned (same trailing distance in each
    operand's original axes) and matched by position too. Contracted axes never take part in matching, so the usual
    ``(a, b) @ (a, b)`` matrix chain works without name collisions.

    Output axis order: batch axes (first-appearance order), then the remaining free axes in operand order.

    `axes` must already be normalised to tuples of selectors (see `multi_dot`).
    """
    n = len(operand_axes)
    if len(axes) not in (n - 1, n):
        raise ValueError(
            f"`axes` must have one (lhs, rhs) spec per adjacent pair: expected {n - 1} (open chain) "
            f"or {n} (closed ring) for {n} operands, got {len(axes)}"
        )

    labels: dict[Hashable, int] = {}

    def label(key: Hashable) -> int:
        return labels.setdefault(key, len(labels))

    def resolve(op: int, sel: tuple[AxisSelector, ...], spec_idx: int, side: str) -> tuple[int, ...]:
        if not sel:
            return ()
        raw = axis_indices(operand_axes[op], sel)
        if any(i is None for i in raw):
            missing = [s for s, i in zip(sel, raw) if i is None]
            raise ValueError(f"axes[{spec_idx}] {side}: could not resolve {missing} in operand {op}")
        return tuple(raw)

    label_of: dict[_Node, int] = {}

    # 1. Explicit contractions: the two ends of each contracted pair get one fresh label.
    for k, (lhs_sel, rhs_sel) in enumerate(axes):
        lhs_op, rhs_op = k, (k + 1) % n
        if len(lhs_sel) != len(rhs_sel):
            raise ValueError(
                f"axes[{k}]: lhs and rhs contracting axes must be the same length, "
                f"got {len(lhs_sel)} vs {len(rhs_sel)}"
            )
        lhs_idx = resolve(lhs_op, lhs_sel, k, "lhs")
        rhs_idx = resolve(rhs_op, rhs_sel, k, "rhs")
        for p, (i, j) in enumerate(zip(lhs_idx, rhs_idx)):
            lax_, rax_ = operand_axes[lhs_op][i], operand_axes[rhs_op][j]
            if lax_.size != rax_.size:
                raise ValueError(f"axes[{k}]: contracting axis size mismatch: `{lax_}` vs `{rax_}`")
            for node in ((lhs_op, i), (rhs_op, j)):
                if node in label_of:
                    raise ValueError(
                        f"axes[{k}]: axis `{operand_axes[node[0]][node[1]]}` of operand {node[0]} "
                        "is contracted more than once"
                    )
                label_of[node] = label(("contract", k, p))

    # 2. Free axes: named axes match by name, anonymous axes (optionally) by distance from the end of their operand.
    #    These key spaces are disjoint from each other and from the contracted axes, so a plain dict suffices.
    groups: dict[int, list[Axis]] = {}  # label -> the free axes carrying it
    for k, op_axes in enumerate(operand_axes):
        for i, ax in enumerate(op_axes):
            if (k, i) in label_of:
                continue
            if ax.name is not None:
                key: Hashable = ("name", ax.name)
            elif allow_positional_fallback:
                key = ("dist", len(op_axes) - 1 - i)
            else:
                key = ("free", k, i)
            lab = label_of[(k, i)] = label(key)
            group = groups.setdefault(lab, [])
            if group and group[0].size != ax.size:
                raise ValueError(f"Batch axis size mismatch: `{group[0]}` vs `{ax}`")
            group.append(ax)

    # 3. Output: batch axes (label on several operands) first, then the remaining free axes.
    batch = [lab for lab, group in groups.items() if len(group) > 1]
    single = [lab for lab, group in groups.items() if len(group) == 1]
    out_sublist = batch + single
    out_axes = tuple(groups[lab][0] for lab in out_sublist)

    sublists = [[label_of[(k, i)] for i in range(len(op_axes))] for k, op_axes in enumerate(operand_axes)]
    return sublists, out_sublist, out_axes


@eqx.filter_jit
def _multi_dot(
    arrays: tuple[NamedArray, ...],
    axes: tuple[tuple[tuple[AxisSelector, ...], tuple[AxisSelector, ...]], ...],
    optimize: str | tuple[tuple[int, ...], ...],
    precision: PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
    *,
    allow_positional_fallback: bool = True,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    sublists, out_sublist, out_axes = _chain_sublists(tuple(a.axes for a in arrays), axes, allow_positional_fallback)

    canonical_sharding = canonicalize_sharding(out_axes, out_sharding, "multi_dot")
    path = optimize if isinstance(optimize, str) else list(optimize)
    # `jnp.einsum` also takes the interleaved ``op0, sublist0, op1, sublist1, ..., out_sublist`` form, so no subscript
    # string is built. It hands path search to `opt_einsum.contract_path` (shapes only, at trace time) and runs each
    # pairwise step through `lax.dot_general`, which is what makes `precision`, `preferred_element_type` and
    # `out_sharding` apply.
    operands = [x for a, sublist in zip(arrays, sublists) for x in (a.array, sublist)]
    jax_array = jnp.einsum(
        *operands,
        out_sublist,
        optimize=path,
        precision=precision,
        preferred_element_type=preferred_element_type,
        out_sharding=canonical_sharding,
    )
    return constructors.array(jax_array, out_axes)


def multi_dot(
    *arrays: NamedArrayLike,
    axes: MultiDotAxes,
    optimize: str | Sequence[tuple[int, ...]] = "auto",
    precision: PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
    allow_positional_fallback: bool = True,
    out_sharding: ShardingLike | None = None,
) -> NamedArray:
    """Contract a chain of arrays, choosing the pairwise contraction order with `opt_einsum`.

    `axes[k]` is a ``(lhs_axes, rhs_axes)`` pair giving the axes of ``arrays[k]`` and ``arrays[k + 1]`` to contract.
    Pass one extra spec, contracting ``arrays[-1]`` with ``arrays[0]``, to close the chain into a ring (a trace).

    Free named axes with the same name on several operands are treated as batch axes and appear once in the output
    (batch axes first, then remaining free axes in operand order). With `allow_positional_fallback` (default), free
    anonymous axes are right-aligned across operands and matched by position, so ``f32[..., ?, ?]`` chains batch over
    the leading dims. Anonymous axes are matched by distance from the end of each operand's original axes, before
    contraction. No conjugation is
    applied; conjugate/transpose operands beforehand (e.g. for ``U^dagger``).

    ``optimize`` is anything `opt_einsum` accepts as a path strategy (``"auto"``, ``"optimal"``, ``"greedy"``,
    ``"dp"``, ...) or an explicit path (a sequence of index tuples).

    Example (open plaquette, then traced at the end)::

        link = (("b",), ("a",))
        P = multi_dot(U_mu, U_nu_shifted, Ud_mu_shifted, Ud_nu, axes=(link, link, link))
        # P has axes (x, y, z, t, a, b)

        # or close the ring directly and let opt_einsum order the whole trace:
        tr_P = multi_dot(U_mu, U_nu_shifted, Ud_mu_shifted, Ud_nu, axes=(link, link, link, (("b",), ("a",))))
    """
    if len(arrays) < 2:
        raise ValueError(f"`multi_dot` requires at least two arrays, got {len(arrays)}")

    named = util.ensure_named("multi_dot", *arrays)
    spec = tuple((axis_selection_to_tuple(lhs), axis_selection_to_tuple(rhs)) for lhs, rhs in axes)
    opt = optimize if isinstance(optimize, str) else tuple(tuple(p) for p in optimize)

    return _multi_dot(
        tuple(named),
        spec,
        opt,
        precision,
        preferred_element_type,
        allow_positional_fallback=allow_positional_fallback,
        out_sharding=out_sharding,
    )
