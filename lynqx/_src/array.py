import operator as op

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import wadler_lindig as wl
from jax import Array
from jax.sharding import Sharding

from lynqx._src.axis_util import check_unique_axis_names
from lynqx._src.typing import Axis, NamedArray


def _validate_shape(array: Array, axes: tuple[Axis, ...]) -> None:
    if array.ndim != len(axes):
        raise ValueError(
            f"Array has {array.ndim} dimensions but {len(axes)} axes were provided: {[ax.label for ax in axes]}"
        )

    for i, (size, ax) in enumerate(zip(array.shape, axes)):
        if size != ax.size:
            raise ValueError(
                f"Axis {ax.label!r} at position {i} has declared size {ax.size} but array has size {size}"
            )

    check_unique_axis_names(axes)


class NamedArrayImpl(NamedArray, eqx.Module):
    array: Array
    axes: tuple[Axis, ...] = eqx.field(static=True)

    def __check_init__(self):
        _validate_shape(self.array, self.axes)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape

    @property
    def dtype(self) -> jnp.dtype:
        return self.array.dtype

    @property
    def itemsize(self) -> int:
        return self.dtype.itemsize

    @property
    def nbytes(self) -> int:
        return self.array.nbytes

    @property
    def ndim(self) -> int:
        return self.array.ndim

    @property
    def size(self) -> int:
        return self.array.size

    @property
    def sharding(self) -> Sharding:
        return self.array.sharding

    def __pdoc__(self, **kwargs) -> wl.AbstractDoc:
        """Wadler-Lindig pretty-printer for NamedArrayImpl.

        Default (compact):
            f32[x:32, y:32]

        Verbose (pass ``verbose=True`` through ``wl.pdoc``/``wl.pformat`` kwargs):
            NamedArray(
            array=f32[32,32], axes=(Axis(name='x', size=32), Axis(name='y', size=32))
            )

        Handles ``array=None`` gracefully (equinox filtered/partitioned pytrees).
        """
        indent: int = kwargs.get("indent", 2)
        verbose: bool = kwargs.get("verbose", False)

        array = self.array  # may be None after eqx.filter / partition
        axes = self.axes  # always a tuple[Axis, ...]

        if verbose or array is None:
            return wl.bracketed(
                begin=wl.TextDoc("NamedArray("),
                docs=wl.named_objs([("array", array), ("axes", axes)], **kwargs),
                sep=wl.comma,
                end=wl.TextDoc(")"),
                indent=indent,
            )

        # compact form: dtype[ax0_name:ax0_size, ax1_name:ax1_size, …]
        dtype_str = (
            str(array.dtype).replace("float", "f").replace("uint", "u").replace("int", "i").replace("complex", "c")
        )

        axis_docs = [wl.TextDoc(f"{ax.label}:{ax.size}") for ax in axes]

        return wl.TextDoc(dtype_str) + wl.bracketed(
            begin=wl.TextDoc("["),
            docs=axis_docs,
            sep=wl.comma,
            end=wl.TextDoc("]"),
            indent=indent,
        )

    def __len__(self):
        try:
            return self.shape[0]
        except IndexError as err:
            raise TypeError("len() of unsized object") from err  # same as numpy error

    # matplotlib compatibility
    def __array__(self, dtype: np.dtype | None = None, copy: bool | None = None):
        kwds = {} if copy is None else {"copy": copy}
        return np.asarray(self.array, dtype=dtype, **kwds)  # pyrefly: ignore[no-matching-overload]

    def __bool__(self):
        return bool(self.array)

    def __float__(self):
        return float(self.array)

    def __int__(self):
        return int(self.array)

    def __complex__(self):
        return complex(self.array)

    def __hex__(self):
        return hex(self.array)

    def __oct__(self):
        return oct(self.array)

    def __index__(self):
        return op.index(self.array)

    def __repr__(self) -> str:
        return wl.pformat(self)

    def __str__(self) -> str:
        return wl.pformat(self)

    def __iter__(self):
        return self.array.__iter__()


def _short_dtype(array) -> str:
    """Contract dtype the same way wl.array_summary does internally."""
    s = str(array.dtype)
    return s.replace("float", "f").replace("uint", "u").replace("int", "i").replace("complex", "c")
