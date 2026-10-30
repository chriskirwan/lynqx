import jax
import jax.numpy as jnp
import numpy as np
import pytest
from lynqx._src.axis import Axis
from lynqx._src.named.constructors import array
from lynqx._src.named.polynomials import (
    poly,
    polyadd,
    polyder,
    PolyDiv,
    polydiv,
    polyfit,
    PolyfitCov,
    PolyfitFull,
    polyint,
    polymul,
    polysub,
    polyval,
    roots,
)


A = jnp.array([1.0, -2.0, 3.0, 4.0])
B = jnp.array([2.0, 5.0])


def na(a, axis="c"):
    return array(a, Axis(a.shape[0], axis))


class TestCoefficientArithmetic:
    @pytest.mark.parametrize("fn, jfn", [(polyadd, jnp.polyadd), (polysub, jnp.polysub), (polymul, jnp.polymul)])
    def test_matches_jnp(self, fn, jfn):
        got = fn(na(A), na(B))
        np.testing.assert_allclose(got.array, jfn(A, B))
        assert got.axes == (Axis(got.array.shape[0]),)

    def test_axis_naming_and_raw_input_rank_check(self):
        assert polyadd(na(A), na(B), "out").axes[0].name == "out"
        with pytest.raises(ValueError, match="1-dimensional"):
            polyadd(array(jnp.ones((2, 2)), {"a": 2, "b": 2}), na(B))

    def test_polydiv_namedtuple(self):
        got = polydiv(na(A), na(B), "d")
        assert isinstance(got, PolyDiv)
        q, r = jnp.polydiv(A, B)
        np.testing.assert_allclose(got.q.array, q)
        np.testing.assert_allclose(got.r.array, r)
        assert got.q.axes[0].name == got.r.axes[0].name == "d"

    def test_jit(self):
        f = jax.jit(lambda a, b: polymul(a, b))
        np.testing.assert_allclose(f(na(A), na(B)).array, jnp.polymul(A, B))


class TestCalculus:
    def test_der_int_roundtrip(self):
        d = polyder(polyint(na(A), 2), 2)
        np.testing.assert_allclose(d.array, A)
        assert polyint(na(A), 2, k=na(jnp.array([1.0, 2.0]), "k")).array.shape == (6,)

    def test_polyder_too_high(self):
        with pytest.raises(ValueError, match="smaller"):
            polyder(na(A), 4)


class TestEvaluation:
    def test_polyval_keeps_x_axes(self):
        x = array(jnp.linspace(0, 1, 6).reshape(2, 3), {"i": 2, "j": 3})
        got = polyval(na(A), x)
        assert got.axes == x.axes
        np.testing.assert_allclose(got.array, jnp.polyval(A, x.array))

    def test_poly_roots_roundtrip(self):
        r = jnp.array([1.0, 2.0, 3.0])
        c = poly(na(r), "c")
        assert c.axes == (Axis(4, "c"),)
        np.testing.assert_allclose(c.array, jnp.poly(r))
        np.testing.assert_allclose(jnp.sort(roots(c).array.real), r, atol=1e-4)

    def test_poly_matrix(self):
        m = jnp.diag(jnp.array([1.0, 2.0]))
        np.testing.assert_allclose(poly(array(m, {"a": 2, "b": 2})).array, jnp.poly(m))
        with pytest.raises(ValueError):
            poly(array(jnp.ones((2, 3)), {"a": 2, "b": 3}))


class TestPolyfit:
    x = jnp.linspace(0, 1, 8)
    y = jnp.sin(x)
    Y = jnp.stack([y, 2 * y], 1)

    def xn(self):
        return array(self.x, Axis(8, "t"))

    def test_plain(self):
        got = polyfit(self.xn(), array(self.y, Axis(8, "t")), 3, axis="coef")
        assert got.axes == (Axis(4, "coef"),)
        np.testing.assert_allclose(got.array, jnp.polyfit(self.x, self.y, 3), atol=1e-5)

    def test_2d_y_keeps_other_axes_and_finds_sample_axis_by_name(self):
        yt = array(self.Y.T, {"k": 2, "t": 8})  # sample axis is *last*
        got = polyfit(self.xn(), yt, 3, axis="coef")
        assert got.axes == (Axis(4, "coef"), Axis(2, "k"))
        np.testing.assert_allclose(got.array, jnp.polyfit(self.x, self.Y, 3), atol=1e-5)

    def test_full(self):
        got = polyfit(self.xn(), array(self.Y, {"t": 8, "k": 2}), 3, full=True)
        assert isinstance(got, PolyfitFull)
        ref = jnp.polyfit(self.x, self.Y, 3, full=True)
        for g, r in zip(got, ref):
            np.testing.assert_allclose(g.array, r, atol=1e-5)
        assert got.resids.axes == (Axis(2, "k"),)
        assert got.rank.axes == () and got.rcond.axes == ()

    def test_full_1d_resids(self):
        got = polyfit(self.xn(), array(self.y, Axis(8, "t")), 3, full=True)
        assert got.resids.shape == (1,)

    @pytest.mark.parametrize("cov", [True, "unscaled"])
    @pytest.mark.parametrize("two_d", [False, True])
    def test_cov(self, cov, two_d):
        yy = array(self.Y, {"t": 8, "k": 2}) if two_d else array(self.y, Axis(8, "t"))
        got = polyfit(self.xn(), yy, 3, cov=cov, axis="coef")
        assert isinstance(got, PolyfitCov)
        ref = jnp.polyfit(self.x, self.Y if two_d else self.y, 3, cov=cov)
        np.testing.assert_allclose(got.C.array, ref[1], atol=1e-5)
        assert got.C.axes[0] == Axis(4, "coef")

    def test_weights_and_errors(self):
        w = array(jnp.ones(8), Axis(8))
        np.testing.assert_allclose(
            polyfit(self.xn(), array(self.y, Axis(8)), 2, w=w).array, jnp.polyfit(self.x, self.y, 2, w=self.x * 0 + 1)
        )
        with pytest.raises(ValueError, match="same number"):
            polyfit(self.xn(), array(self.y[:5], Axis(5)), 2)
        with pytest.raises(ValueError, match="cov"):
            polyfit(self.xn(), array(self.y, Axis(8)), 2, cov="bad")

    def test_jit(self):
        f = jax.jit(lambda x, y: polyfit(x, y, 3))
        np.testing.assert_allclose(
            f(self.xn(), array(self.y, Axis(8, "t"))).array, jnp.polyfit(self.x, self.y, 3), atol=1e-5
        )
