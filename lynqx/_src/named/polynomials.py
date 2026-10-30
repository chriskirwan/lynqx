from typing import Literal, NamedTuple, overload

import jax.numpy as jnp

from lynqx._src.axis_util import axis_index
from lynqx._src.named import constructors, util
from lynqx._src.typing import Axis, AxisLike, NamedArray, NamedArrayLike


# Helpers


def _coeffs(fn_name: str, arg_name: str, p: NamedArrayLike) -> NamedArray:
    """Convert ``p`` to a `NamedArray` and check that it is a rank-1 coefficient array."""
    p = util.ensure_named(fn_name, p)
    if p.ndim != 1:
        raise ValueError(
            f"polynomial coefficient array `{arg_name}` for `{fn_name}` must be 1-dimensional, got {p.ndim}"
        )
    return p


def _coeff_result(jax_array, axis: AxisLike | None, fn_name: str, arg_name: str) -> NamedArray:
    """Wrap a rank-1 JAX coefficient array, resolving the (new) output coefficient axis."""
    out_axis = util.resolve_new_axis(axis, jax_array.shape[0], fn_name, arg_name)
    return constructors.array(jax_array, out_axis)


# Roots <-> coefficients


def roots(p: NamedArrayLike, axis: AxisLike | None = None, *, strip_zeros: bool = True) -> NamedArray:
    """Returns the roots of a polynomial given the coefficients ``p``.

    Jax implementation of :func:`numpy.roots`.

    Args:
        p: Array of polynomial coefficients having rank-1.
        axis: Optional name (or :class:`~lynqx.Axis`) of the output axis indexing the roots.
        strip_zeros: If ``True``, then leading zeros in the coefficients will be stripped, similar to
            :func:`numpy.roots`. If set to ``False``, leading zeros will not be stripped, and undefined roots will be
            represented by ``NaN`` values in the function output. ``strip_zeros`` must be set to ``False`` for the
            function to be compatible with :func:`jax.jit` and other transformations.

    Returns:
        An array containing the roots of the polynomial.

    Note:
        Unlike ``np.roots`` of this function, the ``jnp.roots`` returns the roots in a complex array regardless of the
        values of the roots.
    """
    p = _coeffs("roots", "p", p)

    jax_array = jnp.roots(p.array, strip_zeros=strip_zeros)
    return _coeff_result(jax_array, axis, "roots", "p")


def poly(seq_of_zeros: NamedArrayLike, axis: AxisLike | None = None) -> NamedArray:
    """Returns the coefficients of the monic polynomial with the given roots.

    Jax implementation of :func:`numpy.poly`.

    Args:
        seq_of_zeros: Either a rank-1 array of ``N`` roots, or a square rank-2 array, in which case the roots are the
            eigenvalues of the matrix (i.e. the result is its characteristic polynomial).
        axis: Optional name (or :class:`~lynqx.Axis`) of the output coefficient axis, which has size ``N + 1``.

    Returns:
        A rank-1 array of the ``N + 1`` polynomial coefficients, highest degree first.
    """
    seq = util.ensure_named("poly", seq_of_zeros)
    if seq.ndim == 0 or seq.ndim > 2 or (seq.ndim == 2 and seq.shape[0] != seq.shape[1]):
        raise ValueError(f"`seq_of_zeros` for `poly` must be rank-1 or a square rank-2 array, got shape {seq.shape}")

    return _coeff_result(jnp.poly(seq.array), axis, "poly", "seq_of_zeros")


# Coefficient arithmetic


def polyadd(a1: NamedArrayLike, a2: NamedArrayLike, axis: AxisLike | None = None) -> NamedArray:
    """Returns the sum of two polynomials.

    Jax implementation of :func:`numpy.polyadd`.

    Args:
        a1: Rank-1 array of coefficients of the first polynomial, highest degree first.
        a2: Rank-1 array of coefficients of the second polynomial, highest degree first.
        axis: Optional name (or :class:`~lynqx.Axis`) of the output coefficient axis.

    Returns:
        Rank-1 array of the coefficients of ``a1 + a2``, with size ``max(len(a1), len(a2))``.
    """
    a1 = _coeffs("polyadd", "a1", a1)
    a2 = _coeffs("polyadd", "a2", a2)
    return _coeff_result(jnp.polyadd(a1.array, a2.array), axis, "polyadd", "a1")


def polysub(a1: NamedArrayLike, a2: NamedArrayLike, axis: AxisLike | None = None) -> NamedArray:
    """Returns the difference of two polynomials.

    Jax implementation of :func:`numpy.polysub`.

    Args:
        a1: Rank-1 array of coefficients of the first polynomial, highest degree first.
        a2: Rank-1 array of coefficients of the second polynomial, highest degree first.
        axis: Optional name (or :class:`~lynqx.Axis`) of the output coefficient axis.

    Returns:
        Rank-1 array of the coefficients of ``a1 - a2``, with size ``max(len(a1), len(a2))``.
    """
    a1 = _coeffs("polysub", "a1", a1)
    a2 = _coeffs("polysub", "a2", a2)
    return _coeff_result(jnp.polysub(a1.array, a2.array), axis, "polysub", "a1")


def polymul(
    a1: NamedArrayLike,
    a2: NamedArrayLike,
    axis: AxisLike | None = None,
    *,
    trim_leading_zeros: bool = False,
) -> NamedArray:
    """Returns the product of two polynomials.

    Jax implementation of :func:`numpy.polymul`.

    Args:
        a1: Rank-1 array of coefficients of the first polynomial, highest degree first.
        a2: Rank-1 array of coefficients of the second polynomial, highest degree first.
        axis: Optional name (or :class:`~lynqx.Axis`) of the output coefficient axis.
        trim_leading_zeros: If ``True``, leading zeros are stripped from the inputs, as :func:`numpy.polymul` does.
            This makes the output shape data dependent, so it must be ``False`` (the default) for compatibility
            with :func:`jax.jit` and other transformations.

    Returns:
        Rank-1 array of the coefficients of ``a1 * a2``, with size ``len(a1) + len(a2) - 1``
        (when ``trim_leading_zeros=False``).
    """
    a1 = _coeffs("polymul", "a1", a1)
    a2 = _coeffs("polymul", "a2", a2)
    jax_array = jnp.polymul(a1.array, a2.array, trim_leading_zeros=trim_leading_zeros)
    return _coeff_result(jax_array, axis, "polymul", "a1")


class PolyDiv(NamedTuple):
    """Quotient and remainder returned by :func:`polydiv`."""

    q: NamedArray
    r: NamedArray


def polydiv(
    u: NamedArrayLike,
    v: NamedArrayLike,
    axis: AxisLike | None = None,
    *,
    trim_leading_zeros: bool = False,
) -> PolyDiv:
    """Returns the quotient and remainder of polynomial division.

    Jax implementation of :func:`numpy.polydiv`.

    Args:
        u: Rank-1 array of coefficients of the dividend, highest degree first.
        v: Rank-1 array of coefficients of the divisor, highest degree first.
        axis: Optional name (or :class:`~lynqx.Axis`) used for the coefficient axes of both ``q`` and ``r``.
        trim_leading_zeros: If ``True``, leading zeros are stripped from the inputs, as :func:`numpy.polydiv` does.
            This makes the output shape data dependent, so it must be ``False`` (the default) for compatibility
            with :func:`jax.jit` and other transformations.

    Returns:
        A :class:`PolyDiv` named tuple ``(q, r)`` of rank-1 coefficient arrays, such that ``u = q * v + r``.
    """
    u = _coeffs("polydiv", "u", u)
    v = _coeffs("polydiv", "v", v)
    q, r = jnp.polydiv(u.array, v.array, trim_leading_zeros=trim_leading_zeros)
    return PolyDiv(
        _coeff_result(q, axis, "polydiv", "u"),
        _coeff_result(r, axis, "polydiv", "u"),
    )


# Calculus


def polyder(p: NamedArrayLike, m: int = 1, axis: AxisLike | None = None) -> NamedArray:
    """Returns the ``m``-th derivative of a polynomial.

    Jax implementation of :func:`numpy.polyder`.

    Args:
        p: Rank-1 array of polynomial coefficients, highest degree first.
        m: Order of the derivative. It must be a static, positive integer smaller than ``len(p)`` since named axes
            cannot be empty.
        axis: Optional name (or :class:`~lynqx.Axis`) of the output coefficient axis.

    Returns:
        Rank-1 array of the derivative's coefficients, with size ``len(p) - m``.
    """
    p = _coeffs("polyder", "p", p)
    if m < 0:
        raise ValueError(f"order of derivative `m` for `polyder` must be non-negative, got {m}")
    if m >= p.shape[0]:
        raise ValueError(
            f"order of derivative `m`={m} for `polyder` must be smaller than the number of coefficients "
            f"({p.shape[0]}); the result would be an empty axis"
        )
    return _coeff_result(jnp.polyder(p.array, m), axis, "polyder", "p")


def polyint(
    p: NamedArrayLike,
    m: int = 1,
    k: NamedArrayLike | None = None,
    axis: AxisLike | None = None,
) -> NamedArray:
    """Returns the ``m``-th antiderivative of a polynomial.

    Jax implementation of :func:`numpy.polyint`.

    Args:
        p: Rank-1 array of polynomial coefficients, highest degree first.
        m: Order of the antiderivative. It must be a static, positive integer.
        k: Integration constants. Either a scalar (used for every integration) or a rank-1 array of length ``m``,
            lowest order integration first. Default is zero.
        axis: Optional name (or :class:`~lynqx.Axis`) of the output coefficient axis.

    Returns:
        Rank-1 array of the antiderivative's coefficients, with size ``len(p) + m``.
    """
    p = _coeffs("polyint", "p", p)
    if m < 0:
        raise ValueError(f"order of integration `m` for `polyint` must be non-negative, got {m}")

    k_array = None
    if k is not None:
        k_named = util.ensure_named("polyint", k)
        if k_named.ndim > 1:
            raise ValueError(f"integration constants `k` for `polyint` must be scalar or rank-1, got {k_named.ndim}")
        k_array = k_named.array

    return _coeff_result(jnp.polyint(p.array, m, k_array), axis, "polyint", "p")


# Evaluation


def polyval(p: NamedArrayLike, x: NamedArrayLike, *, unroll: int = 16) -> NamedArray:
    """Evaluates a polynomial at specific values.

    Jax implementation of :func:`numpy.polyval`, evaluated with Horner's method.

    Args:
        p: Rank-1 array of polynomial coefficients, highest degree first.
        x: Array (of any rank and axes) of points at which to evaluate the polynomial.
        unroll: Static loop unroll factor for the Horner recursion. Default is 16.

    Returns:
        A `NamedArray` with the same axes as ``x`` containing ``p(x)``.
    """
    p, x = _coeffs("polyval", "p", p), util.ensure_named("polyval", x)
    return constructors.array(jnp.polyval(p.array, x.array, unroll=unroll), x.axes)


class PolyfitFull(NamedTuple):
    """Result of :func:`polyfit` with ``full=True``."""

    p: NamedArray
    resids: NamedArray
    rank: NamedArray
    s: NamedArray
    rcond: NamedArray


class PolyfitCov(NamedTuple):
    """Result of :func:`polyfit` with ``cov=True`` or ``cov='unscaled'`` (and ``full=False``)."""

    p: NamedArray
    C: NamedArray


@overload
def polyfit(
    x: NamedArrayLike,
    y: NamedArrayLike,
    deg: int,
    rcond: float | None = None,
    *,
    full: Literal[False] = False,
    w: NamedArrayLike | None = None,
    cov: Literal[False] = False,
    axis: AxisLike | None = None,
) -> NamedArray: ...


@overload
def polyfit(
    x: NamedArrayLike,
    y: NamedArrayLike,
    deg: int,
    rcond: float | None = None,
    *,
    full: Literal[True],
    w: NamedArrayLike | None = None,
    cov: bool | Literal["unscaled"] = False,
    axis: AxisLike | None = None,
) -> PolyfitFull: ...


@overload
def polyfit(
    x: NamedArrayLike,
    y: NamedArrayLike,
    deg: int,
    rcond: float | None = None,
    *,
    full: Literal[False] = False,
    w: NamedArrayLike | None = None,
    cov: Literal[True, "unscaled"],
    axis: AxisLike | None = None,
) -> PolyfitCov: ...


def polyfit(
    x: NamedArrayLike,
    y: NamedArrayLike,
    deg: int,
    rcond: float | None = None,
    *,
    full: bool = False,
    w: NamedArrayLike | None = None,
    cov: bool | Literal["unscaled"] = False,
    axis: AxisLike | None = None,
) -> NamedArray | PolyfitCov | PolyfitFull:
    r"""Least squares polynomial fit to data.

    Jax implementation of :func:`numpy.polyfit`.

    Given a set of data points ``(x, y)`` and degree of polynomial ``deg``, the
    function finds a polynomial equation of the form:

    .. math::

        y = p(x) = p[0] x^{deg} + p[1] x^{deg - 1} + ... + p[deg]

    Args:
        x: Rank-1 array of the ``M`` sample points.
        y: Array of the ``M`` sample values, either rank-1 or rank-2. For rank-2 input, the sample axis is the one
            matching the axis of ``x`` by name (or, if ``x``'s axis is anonymous or absent from ``y``, the first
            axis); every other axis is an independent set of ``K`` data series, fitted simultaneously.
        deg: Degree of the polynomial. It must be specified statically.
        rcond: Relative condition number of the fit. Default value is ``len(x) * eps``. It must be specified
            statically.
        full: Switch that controls the return value. Default is ``False``, which returns only the coefficients ``p``
            (or ``(p, C)`` if ``cov`` is set). If ``True``, a :class:`PolyfitFull` named tuple
            ``(p, resids, rank, s, rcond)`` is returned. It must be specified statically.
        w: Rank-1 array of weights of shape ``(M,)``. If ``None``, all data points are considered to have equal
            weight. If not ``None``, the weight :math:`w_i` is applied to the unsquared residual of
            :math:`y_i - \widehat{y}_i` at :math:`x_i`, where :math:`\widehat{y}_i` is the fitted value of
            :math:`y_i`. Default is ``None``.
        cov: Boolean or string. If ``True``, returns the covariance matrix scaled by ``resids / (M - deg - 1)`` along
            with the polynomial coefficients. If ``cov='unscaled'``, returns the unscaled version of the covariance
            matrix. Default is ``False``. ``cov`` is ignored if ``full=True``. It must be specified statically.
        axis: Optional name (or :class:`~lynqx.Axis`) of the coefficient axis, of size ``deg + 1``. Defaults to an
            anonymous axis.

    Returns:
        - A `NamedArray` of polynomial coefficients ``p`` if ``full=False`` and ``cov=False``.

        - A :class:`PolyfitFull` named tuple ``(p, resids, rank, s, rcond)`` if ``full=True``, where

          - ``p`` has axes ``(coefficients, *K)``, where ``K`` are the non-sample axes of ``y`` (none for rank-1
            ``y``), i.e. shape ``(deg + 1,)`` or ``(deg + 1, K)``.
          - ``resids`` is the sum of squared residuals, of shape ``(K,)``; for rank-1 ``y`` it has a single
            anonymous axis of size 1.
          - ``rank`` is the scalar rank of the scaled Vandermonde matrix.
          - ``s`` holds the singular values of the scaled Vandermonde matrix, along an anonymous axis.
          - ``rcond`` is the scalar condition number used for the fit.

        - A :class:`PolyfitCov` named tuple ``(p, C)`` if ``full=False`` and ``cov`` is set, where

          - ``p`` is as above.
          - ``C`` is the covariance matrix of the coefficients, with axes ``(coefficients, anonymous, *K)``, i.e.
            shape ``(deg + 1, deg + 1)`` for rank-1 ``y`` and ``(deg + 1, deg + 1, K)`` for rank-2 ``y``. (The
            second coefficient axis is anonymous so that axis names stay unique.) For ``cov='unscaled'`` and
            rank-2 ``y``, the trailing axis is an anonymous axis of size 1, as in :func:`jax.numpy.polyfit`.

    Note:
        Unlike :func:`numpy.polyfit`, this function will not warn on rank reduction, which indicates an ill
        conditioned matrix.
    """
    x, y = util.ensure_named("polyfit", x, y)
    if x.ndim != 1:
        raise ValueError(f"`x` for `polyfit` must be 1-dimensional, got {x.ndim}")
    if y.ndim not in (1, 2):
        raise ValueError(f"`y` for `polyfit` must be 1- or 2-dimensional, got {y.ndim}")
    if cov not in (True, False, "unscaled"):
        raise ValueError(f"`cov` for `polyfit` must be a bool or 'unscaled', got {cov!r}")

    m = x.shape[0]

    # Locate the sample axis of y and move it to the front, as jnp.polyfit expects.
    y_array = y.array
    if y.ndim == 1:
        sample = 0
    else:
        x_axis = x.axes[0]
        sample = axis_index(y.axes, x_axis.name)
        if sample is None:
            sample = 0
        y_array = jnp.moveaxis(y_array, sample, 0)
    k_axes = tuple(ax for i, ax in enumerate(y.axes) if i != sample)
    if y.shape[sample] != m:
        raise ValueError(
            f"`x` and `y` for `polyfit` must have the same number of samples, got {m} and {y.shape[sample]}"
        )

    w_array = None
    if w is not None:
        w = _coeffs("polyfit", "w", w)
        if w.shape[0] != m:
            raise ValueError(f"`w` for `polyfit` must have shape ({m},), got {w.shape}")
        w_array = w.array

    coef_axis = util.resolve_new_axis(axis, deg + 1, "polyfit", "y")

    if full:
        p, resids, rank, s, rc = jnp.polyfit(x.array, y_array, deg, rcond, full=True, w=w_array)
        return PolyfitFull(
            constructors.array(p, (coef_axis, *k_axes)),
            constructors.array(resids, k_axes if y.ndim == 2 else Axis(resids.shape[0])),
            constructors.array(rank, ()),
            constructors.array(s, Axis(s.shape[0])),
            constructors.array(rc, ()),
        )

    if cov:
        p, c = jnp.polyfit(x.array, y_array, deg, rcond, full=False, w=w_array, cov=cov)  # pyrefly: ignore[bad-argument-type]  # JAX stubs omit 'unscaled'
        if c.ndim == 2:
            c_tail: tuple[Axis, ...] = ()
        elif cov == "unscaled":
            c_tail = (Axis(c.shape[2]),)
        else:
            c_tail = k_axes
        return PolyfitCov(
            constructors.array(p, (coef_axis, *k_axes)),
            constructors.array(c, (coef_axis, Axis(deg + 1), *c_tail)),
        )

    p = jnp.polyfit(x.array, y_array, deg, rcond, full=False, w=w_array)
    return constructors.array(p, (coef_axis, *k_axes))
