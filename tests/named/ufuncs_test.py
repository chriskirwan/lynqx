from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import jax.test_util as jtu
import numpy as np
import pytest
from lynqx._src.array import NamedArrayImpl
from lynqx._src.axis import Axis
from lynqx._src.named import ufuncs


jax.config.update("jax_enable_x64", True)

FLOAT_DTYPES = (jnp.float32, jnp.float64)
INT_DTYPES = (jnp.int32, jnp.int64)
BOOL_DTYPES = (jnp.bool_,)
DEFAULT_TOL = {jnp.float32: 1e-5, jnp.float64: 1e-12}

# Fixed axis shape shared by every table-driven test, so a failure always reports
# against the same, easy-to-reason-about geometry: a 3x4 array over axes "batch","x".
AXIS_SHAPE = (("batch", 3), ("x", 4))


@dataclass(frozen=True)
class UfuncRecord:
    """One row of the op table: everything needed to test a single ufunc generically."""

    name: str  # attribute shared by `ufuncs` and `jnp`
    arity: int
    dtypes: tuple = FLOAT_DTYPES
    tol: dict = field(default_factory=lambda: dict(DEFAULT_TOL))
    test_grad: bool = True
    domain: tuple[float, float] = (-2.0, 2.0)  # sampling range; keep poles/branch cuts out


UNARY_RECORDS = [
    UfuncRecord("negative", 1),
    UfuncRecord("positive", 1),
    UfuncRecord("abs", 1),
    UfuncRecord("sign", 1, test_grad=False),
    UfuncRecord("exp", 1),
    UfuncRecord("expm1", 1),
    UfuncRecord("log", 1, domain=(0.1, 5.0)),
    UfuncRecord("log1p", 1, domain=(-0.5, 5.0)),
    UfuncRecord("sqrt", 1, domain=(0.1, 5.0)),
    UfuncRecord("square", 1),
    UfuncRecord("reciprocal", 1, domain=(0.5, 5.0)),
    UfuncRecord("sin", 1),
    UfuncRecord("cos", 1),
    UfuncRecord("tan", 1, domain=(-1.0, 1.0)),
    UfuncRecord("tanh", 1),
    UfuncRecord("sinh", 1),
    UfuncRecord("cosh", 1),
    UfuncRecord("arcsinh", 1),
    UfuncRecord("floor", 1, test_grad=False),
    UfuncRecord("ceil", 1, test_grad=False),
    UfuncRecord("around", 1, test_grad=False),
    UfuncRecord("isnan", 1, dtypes=FLOAT_DTYPES, test_grad=False),
    UfuncRecord("isfinite", 1, dtypes=FLOAT_DTYPES, test_grad=False),
    UfuncRecord("logical_not", 1, dtypes=BOOL_DTYPES, test_grad=False),
    UfuncRecord("bitwise_not", 1, dtypes=INT_DTYPES, test_grad=False),
]

BINARY_RECORDS = [
    UfuncRecord("add", 2),
    UfuncRecord("subtract", 2),
    UfuncRecord("multiply", 2),
    UfuncRecord("true_divide", 2, domain=(0.5, 5.0)),
    UfuncRecord("float_power", 2, domain=(0.5, 3.0)),
    UfuncRecord("power", 2, domain=(0.5, 3.0)),
    UfuncRecord("maximum", 2, test_grad=False),
    UfuncRecord("minimum", 2, test_grad=False),
    UfuncRecord("hypot", 2),
    UfuncRecord("logaddexp", 2),
    UfuncRecord("atan2", 2),
    UfuncRecord("equal", 2, test_grad=False),
    UfuncRecord("not_equal", 2, test_grad=False),
    UfuncRecord("greater", 2, test_grad=False),
    UfuncRecord("greater_equal", 2, test_grad=False),
    UfuncRecord("less", 2, test_grad=False),
    UfuncRecord("less_equal", 2, test_grad=False),
    UfuncRecord("logical_and", 2, dtypes=BOOL_DTYPES, test_grad=False),
    UfuncRecord("logical_or", 2, dtypes=BOOL_DTYPES, test_grad=False),
    UfuncRecord("bitwise_and", 2, dtypes=INT_DTYPES, test_grad=False),
    UfuncRecord("bitwise_or", 2, dtypes=INT_DTYPES, test_grad=False),
    UfuncRecord("bitwise_xor", 2, dtypes=INT_DTYPES, test_grad=False),
    UfuncRecord("floor_divide", 2, dtypes=INT_DTYPES, test_grad=False, domain=(1, 10)),
    UfuncRecord("mod", 2, dtypes=INT_DTYPES, test_grad=False, domain=(1, 10)),
]

ALL_DTYPES = FLOAT_DTYPES + INT_DTYPES + BOOL_DTYPES


def _sample(key: jax.Array, shape: tuple[int, ...], dtype, domain: tuple[float, float]):
    """Random values of `dtype` within `domain`."""
    if jnp.issubdtype(dtype, jnp.integer):
        lo, hi = int(domain[0]), int(domain[1])
        return jax.random.randint(key, shape, lo, hi + 1, dtype=dtype)
    if dtype == jnp.bool_:
        return jax.random.bernoulli(key, 0.5, shape)
    lo, hi = domain
    return jax.random.uniform(key, shape, dtype=dtype, minval=lo, maxval=hi)


def _make_named(key: jax.Array, dtype, domain: tuple[float, float]) -> NamedArrayImpl:
    axes = tuple(Axis(size, name) for name, size in AXIS_SHAPE)
    shape = tuple(size for _, size in AXIS_SHAPE)
    return NamedArrayImpl(_sample(key, shape, dtype, domain), axes)


def _tol_for(record: UfuncRecord, dtype) -> float:
    return record.tol.get(dtype, 1e-6)


@pytest.mark.parametrize("dtype", ALL_DTYPES, ids=str)
@pytest.mark.parametrize("record", UNARY_RECORDS, ids=lambda r: r.name)
def test_unary_matches_reference(record: UfuncRecord, dtype) -> None:
    if dtype not in record.dtypes:
        pytest.skip(f"{record.name} not defined for {dtype}")

    lynqx_fn = getattr(ufuncs, record.name)
    ref_fn = getattr(jnp, record.name)

    x = _make_named(jax.random.key(0), dtype, record.domain)
    got = lynqx_fn(x)
    want = ref_fn(x.array)

    tol = _tol_for(record, dtype)
    np.testing.assert_allclose(got.array, want, rtol=tol, atol=tol)
    assert got.axes == x.axes, "unary ops must preserve axes exactly, unchanged"


@pytest.mark.parametrize("dtype", ALL_DTYPES, ids=str)
@pytest.mark.parametrize("record", BINARY_RECORDS, ids=lambda r: r.name)
def test_binary_matches_reference(record: UfuncRecord, dtype) -> None:
    if dtype not in record.dtypes:
        pytest.skip(f"{record.name} not defined for {dtype}")

    lynqx_fn = getattr(ufuncs, record.name)
    ref_fn = getattr(jnp, record.name)

    k1, k2 = jax.random.split(jax.random.key(0))
    x = _make_named(k1, dtype, record.domain)
    y = _make_named(k2, dtype, record.domain)

    got = lynqx_fn(x, y)
    want = ref_fn(x.array, y.array)

    tol = _tol_for(record, dtype)
    np.testing.assert_allclose(got.array, want, rtol=tol, atol=tol)


@pytest.mark.parametrize(
    "record", [r for r in UNARY_RECORDS if r.test_grad and jnp.float64 in r.dtypes], ids=lambda r: r.name
)
def test_unary_grad(record: UfuncRecord) -> None:
    lynqx_fn = getattr(ufuncs, record.name)
    x = _make_named(jax.random.key(1), jnp.float64, record.domain)

    def f(arr):
        return lynqx_fn(NamedArrayImpl(arr, x.axes)).array.sum()

    jtu.check_grads(f, (x.array,), order=1, modes=["rev"])


@pytest.mark.parametrize(
    "record", [r for r in BINARY_RECORDS if r.test_grad and jnp.float64 in r.dtypes], ids=lambda r: r.name
)
def test_binary_grad(record: UfuncRecord) -> None:
    lynqx_fn = getattr(ufuncs, record.name)
    k1, k2 = jax.random.split(jax.random.key(1))
    x = _make_named(k1, jnp.float64, record.domain)
    y = _make_named(k2, jnp.float64, record.domain)

    def f(a, b):
        return lynqx_fn(NamedArrayImpl(a, x.axes), NamedArrayImpl(b, y.axes)).array.sum()

    jtu.check_grads(f, (x.array, y.array), order=1, modes=["rev"])


def test_binary_broadcasts_over_missing_axis() -> None:
    """A binary op between a 2-axis and a 1-axis NamedArray should broadcast the
    missing axis by name, the same way NumPy broadcasts a missing trailing dim."""
    batch, x_axis = Axis(3, "batch"), Axis(4, "x")
    lhs = NamedArrayImpl(jnp.arange(12.0).reshape(3, 4), (batch, x_axis))
    rhs = NamedArrayImpl(jnp.arange(3.0), (batch,))

    got = ufuncs.add(lhs, rhs)
    want = lhs.array + rhs.array[:, None]

    np.testing.assert_allclose(got.array, want)
    assert [a.name for a in got.axes] == ["batch", "x"]


def test_binary_broadcasts_size_one_axis() -> None:
    """Size-1 axes broadcast against any size, matching NumPy's size-1 broadcast rule."""
    batch, x_axis = Axis(3, "batch"), Axis(4, "x")
    lhs = NamedArrayImpl(jnp.arange(12.0).reshape(3, 4), (batch, x_axis))
    rhs = NamedArrayImpl(jnp.ones((1,)), (Axis(1, "x"),))

    got = ufuncs.multiply(lhs, rhs)
    np.testing.assert_allclose(got.array, lhs.array)
    assert got.axes == (batch, x_axis)


def test_binary_incompatible_named_axes_raise() -> None:
    """Two differently-sized axes sharing a name must fail loudly, not silently
    broadcast the wrong data together."""
    lhs = NamedArrayImpl(jnp.ones((3,)), (Axis(3, "x"),))
    rhs = NamedArrayImpl(jnp.ones((4,)), (Axis(4, "x"),))

    with pytest.raises(ValueError):
        ufuncs.add(lhs, rhs)


@pytest.mark.parametrize("op_name", ["frexp", "modf", "divmod"])
def test_multi_output_ops_preserve_axes(op_name: str) -> None:
    """Tuple-returning ops (frexp/modf/divmod) must propagate axes to *both* outputs."""
    axes = (Axis(3, "batch"), Axis(4, "x"))
    x = NamedArrayImpl(jax.random.uniform(jax.random.key(2), (3, 4), minval=1.0, maxval=8.0), axes)

    if op_name == "divmod":
        y = NamedArrayImpl(jnp.full((3, 4), 2.0), axes)
        out0, out1 = ufuncs.divmod(x, y)
        want0, want1 = jnp.divmod(x.array, y.array)
    else:
        fn = getattr(ufuncs, op_name)
        out0, out1 = fn(x)
        want0, want1 = getattr(jnp, op_name)(x.array)

    np.testing.assert_allclose(out0.array, want0)
    np.testing.assert_allclose(out1.array, want1)
    assert out0.axes == axes
    assert out1.axes == axes
