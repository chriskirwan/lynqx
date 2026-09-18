import re
import weakref
from collections.abc import Iterable
from typing import Any, ClassVar, NamedTuple

import numpy as np
from jaxtyping._storage import get_shape_memo

from lynqx._src.typing import Axis, NamedArray


class NamedArrayCheckError(TypeError):
    pass


class NamedArrayDTypeError(NamedArrayCheckError):
    pass


class NamedArrayAxisError(NamedArrayCheckError):
    pass


class _AxisToken(NamedTuple):
    name: str | None
    size: int | None


# enforce digits-only (no underscores, signs, or whitespace) and size > 0.
_SIZE_RE = re.compile(r"[1-9][0-9]*")


def _parse_token(tok: str, dim_str: str) -> _AxisToken:
    name, sep, size_str = tok.partition(":")

    if name == "_":
        name = None
    elif not name.isidentifier():
        raise ValueError(f"Invalid axis token {tok!r} in dim_str {dim_str!r}. Use a Python identifier or `_`.")

    size: int | None = None
    if sep == ":":
        if not _SIZE_RE.fullmatch(size_str):
            raise ValueError(
                f"Invalid size {size_str!r} for axis token {tok!r} in dim_str "
                f"{dim_str!r} — expected a positive integer."
            )
        size = int(size_str)

    return _AxisToken(name, size)


class NamedAxisSpec:
    """
    Parsed, hashable axis specification for NamedArray type annotations.

    Grammar
    -------
        ``""``                  scalar — zero axes
        ``"..."``               wildcard — any axes, dtype-only check
        ``"a b:3"`` or ``"(a b:3)"`` ordered, exact: axis-for-axis match in sequence.
        ``"(... a b)"``         ordered suffix constraint
        ``"(a b ...)"``         ordered prefix constraint
        ``"{a b:3}"``           unordered, set-based: named axes ⊇ {a, b:3}.
                                Anonymous axes on the instance are ignored; `_` is not
                                valid here (matching an anonymous axis needs a position).

    Note: a braced spec with no tokens (``"{...}"`` or ``"(...)"``) is treated as a
    dtype-checked wildcard — it matches any axes but still checks dtype.
    """

    __slots__ = (
        "tokens",
        "ordered",
        "leading_extra",
        "trailing_extra",
        "is_wildcard",
        "_raw",
        "_key",
        "_describe",
    )

    def __init__(self, dim_str: str) -> None:
        self._raw = dim_str.strip()

        self.tokens: tuple[_AxisToken, ...] = ()
        self.ordered = True
        self.leading_extra = False
        self.trailing_extra = False
        self.is_wildcard = False

        if self._raw == "...":
            self.is_wildcard = True
            self.ordered = False
            self.leading_extra = self.trailing_extra = True
            self._finalize()
            return

        if self._raw == "":
            if dim_str == "":
                # Scalar spec
                self.ordered = True
                self.leading_extra = self.trailing_extra = False
                self._finalize()
                return
            else:
                # Whitespace-only unbraced spec: no tokens, unordered, open.
                self.ordered = False
                self.leading_extra = self.trailing_extra = True
                self._finalize()
                return

        rest = self._raw
        if rest.startswith("{") and rest.endswith("}"):
            self.ordered = False
            inner = rest[1:-1].strip()
            if inner == "...":
                self.is_wildcard = True
                self.leading_extra = self.trailing_extra = True
                self._finalize()
                return
            self.leading_extra = self.trailing_extra = True
            raw_tokens = [t for t in inner.split() if t != "..."]
        elif rest.startswith("(") and rest.endswith(")"):
            self.ordered = True
            inner = rest[1:-1].strip()
            if inner == "...":
                self.is_wildcard = True
                self.leading_extra = self.trailing_extra = True
                self._finalize()
                return

            self.leading_extra = inner.startswith("...")
            if self.leading_extra:
                inner = inner[3:].strip()
            self.trailing_extra = inner.endswith("...")
            if self.trailing_extra:
                inner = inner[:-3].strip()

            if self.leading_extra and self.trailing_extra and inner:
                raise ValueError(
                    f"{dim_str!r}: '(... x ...)' "
                    "(contiguous-subsequence-anywhere) is not supported — use a "
                    "prefix ('(x ...)') or suffix ('(... x)') constraint."
                )

            raw_tokens = [t for t in inner.split() if t != "..."]
        else:
            self.ordered = True
            self.leading_extra = self.trailing_extra = False
            raw_tokens = rest.split()

        tokens = tuple(_parse_token(t, dim_str) for t in raw_tokens)

        if not self.ordered and any(t.name is None for t in tokens):
            raise ValueError(
                f"{dim_str!r}: `_` (anonymous axis) is only valid inside an "
                "ordered spec — matching an anonymous axis requires a position."
            )

        named = [t.name for t in tokens if t.name is not None]
        if len(set(named)) != len(named):
            raise ValueError(f"Duplicate named axis token in dim_str {dim_str!r}.")

        self.tokens = tokens
        self._finalize()

    def _finalize(self) -> None:
        self._key = (
            self.ordered,
            self.leading_extra,
            self.trailing_extra,
            self.is_wildcard,
            self.tokens,
        )
        self._describe = self._compute_describe()

    def _token_strs(self) -> list[str]:
        return [f"{t.name or '_'}{f':{t.size}' if t.size is not None else ''}" for t in self.tokens]

    def _compute_describe(self) -> str:
        strs = self._token_strs()
        if self.is_wildcard:
            return "any axes (wildcard)"
        if not self.tokens and self.ordered and not self.leading_extra and not self.trailing_extra:
            return "scalar (no axes)"
        if not self.ordered:
            return f"named axes ⊇ {{{', '.join(strs)}}}  (unordered, open)" if strs else "any axes (unordered, open)"

        parts = (["..."] if self.leading_extra else []) + strs + (["..."] if self.trailing_extra else [])
        braced = "(" + " ".join(parts) + ")"
        if not self.leading_extra and not self.trailing_extra:
            return f"axes == {braced}  (ordered, exact)"
        return f"axes {'end' if self.leading_extra else 'start'} with {braced}"

    def describe(self) -> str:
        return self._describe

    def __repr__(self) -> str:
        return f"NamedAxisSpec({self._raw!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, NamedAxisSpec) and self._key == other._key

    def __hash__(self) -> int:
        return hash(self._key)


def _token_matches(axis: Axis, token: _AxisToken) -> bool:
    if (axis.name is None) != (token.name is None):
        return False
    if token.name is not None and axis.name != token.name:
        return False
    return token.size is None or axis.size == token.size


def _bind_axis(name: str, size: int, spec_describe: str) -> None:
    """Bind a named axis's size into jaxtyping's per-call shape memo.

    This mirrors jaxtyping's own cross-argument dimension-binding semantics
    (see ``jaxtyping._storage``): within one call to a function wrapped in
    ``@jaxtyped(typechecker=...)``, the *first* argument to mention a given
    dim name fixes its size in a shared, per-call memo; every later argument
    (named-array or plain-array, ours or jaxtyping's own) that mentions the
    same name must agree, or the check fails right here — as a normal type
    violation raised by the typechecker — instead of surfacing later as an
    opaque broadcasting/shape error from inside the function body.

    Outside a ``@jaxtyped`` call (e.g. a bare ``isinstance`` check at import
    time), ``get_shape_memo`` hands back a fresh, unshared dict each time, so
    this degrades to a no-op consistency check with nothing to persist —
    matching jaxtyping's own documented behaviour.
    """
    single_memo, _variadic_memo, _pytree_memo, _arguments = get_shape_memo()
    existing = single_memo.get(name)
    if existing is None:
        single_memo[name] = size
    elif existing != size:
        raise NamedArrayAxisError(
            f"Size mismatch for axis {name!r}: got {size}, but a previous "
            f"argument in this call bound {name!r}={existing} — {spec_describe}"
        )


def _validate_unordered(axes: tuple[Axis, ...], spec: NamedAxisSpec) -> None:
    named = {a.name: a.size for a in axes if a.name is not None}
    for t in spec.tokens:
        if t.name not in named:
            raise NamedArrayAxisError(f"NamedArray missing axis {t.name!r} — {spec.describe()}")
        if t.size is not None and named[t.name] != t.size:
            raise NamedArrayAxisError(
                f"NamedArray axis {t.name!r} has size {named[t.name]}, expected {t.size} — {spec.describe()}"
            )
        _bind_axis(t.name, named[t.name], spec.describe())


def _validate_ordered(axes: tuple[Axis, ...], spec: NamedAxisSpec) -> None:
    n = len(spec.tokens)

    if not spec.leading_extra and not spec.trailing_extra:
        window = axes
        ok = len(axes) == n
    elif spec.leading_extra:  # suffix
        window = axes[-n:] if n else ()
        ok = len(axes) >= n
    else:  # prefix
        window = axes[:n]
        ok = len(axes) >= n

    if ok:
        ok = all(_token_matches(a, t) for a, t in zip(window, spec.tokens))

    if not ok:
        raise NamedArrayAxisError(
            f"NamedArray axes {list(axes)} do not satisfy {spec.describe()} (got window {list(window)})"
        )

    for a, t in zip(window, spec.tokens):
        if t.name is not None:
            _bind_axis(t.name, a.size, spec.describe())


def _validate(
    instance: Any,
    dtype_name: str,
    valid_dtypes: frozenset[np.dtype] | None,
    spec: NamedAxisSpec,
) -> None:
    if valid_dtypes is not None:
        try:
            actual_dtype = np.dtype(instance.dtype)
        except Exception as exc:
            raise NamedArrayDTypeError(f"Could not resolve dtype of {instance!r}: {exc}") from exc
        if actual_dtype not in valid_dtypes:
            valid_names = sorted(str(d) for d in valid_dtypes)
            raise NamedArrayDTypeError(
                f"NamedArray dtype mismatch: got {actual_dtype!r}, expected {dtype_name} ({', '.join(valid_names)})"
            )

    if spec.is_wildcard:
        return

    if not hasattr(instance, "axes"):
        raise NamedArrayAxisError(f"Expected NamedArray with .axes attribute, got {type(instance)}")
    axes: tuple[Axis, ...] = instance.axes

    if spec.ordered:
        _validate_ordered(axes, spec)
    else:
        _validate_unordered(axes, spec)


class _MetaNamedArrayType(type):
    _dtype_name: str
    _valid_dtypes: frozenset[np.dtype] | None
    _axis_spec: NamedAxisSpec

    def __instancecheck__(cls, instance: Any) -> bool:
        if not isinstance(instance, NamedArray):
            return False
        try:
            _validate(instance, cls._dtype_name, cls._valid_dtypes, cls._axis_spec)
        except NamedArrayCheckError:
            return False
        return True

    def __repr__(cls) -> str:
        return f"<{cls.__name__} — {cls._axis_spec.describe()}>"


_type_cache: weakref.WeakValueDictionary[tuple[type, NamedAxisSpec], type] = weakref.WeakValueDictionary()


def _make_named_array_type(dtype_class: Any, axis_spec: NamedAxisSpec) -> type:
    cache_key = (dtype_class, axis_spec)
    if (cached := _type_cache.get(cache_key)) is not None:
        return cached

    raw_dtypes = getattr(dtype_class, "dtypes", None)
    valid_dtypes = (
        frozenset(np.dtype(d) for d in raw_dtypes)
        if isinstance(raw_dtypes, Iterable) and not isinstance(raw_dtypes, (str, bytes))
        else None
    )

    new_type = _MetaNamedArrayType(
        f"{dtype_class.__name__}[NamedArray, {axis_spec._raw!r}]",
        (NamedArray,),
        {
            "__module__": __name__,
            "_dtype_name": dtype_class.__name__,
            "_valid_dtypes": valid_dtypes,
            "_axis_spec": axis_spec,
        },
    )
    _type_cache[cache_key] = new_type
    return new_type


# internal specifier classes per jt_dtype so the same DType doesn't get a
# fresh class on every annotation.
_specifier_cache: weakref.WeakValueDictionary[type, type] = weakref.WeakValueDictionary()


def _make_dtype_specifier(jt_dtype: Any) -> type:
    if (cached := _specifier_cache.get(jt_dtype)) is not None:
        return cached

    class _NamedAwareDtype:
        dtypes: ClassVar[Any] = getattr(jt_dtype, "dtypes", None)

        def __class_getitem__(cls, item: tuple[type, str]) -> Any:
            array_type, dim_str = item
            is_named = array_type is NamedArray or (
                isinstance(array_type, type) and issubclass(array_type, NamedArray)
            )
            if is_named:
                return _make_named_array_type(jt_dtype, NamedAxisSpec(dim_str))

            if dim_str.strip().startswith("{"):
                raise TypeError("Brace syntax {…} is only valid for NamedArray, not plain arrays.")
            return jt_dtype[array_type, dim_str]

    _NamedAwareDtype.__name__ = jt_dtype.__name__
    _specifier_cache[jt_dtype] = _NamedAwareDtype
    return _NamedAwareDtype
