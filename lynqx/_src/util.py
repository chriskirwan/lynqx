from collections.abc import Callable
from functools import lru_cache
from typing import cast, ParamSpec, Protocol, TypeVar

from jax.extend.backend import register_backend_cache


S = ParamSpec("S")
T = TypeVar("T", covariant=True)


class _CachedFunction(Protocol[S, T]):
    def __call__(self, *args: S.args, **kwargs: S.kwargs) -> T: ...
    def cache_info(self): ...
    def cache_clear(self): ...


def cache(maxsize: int = 128) -> Callable[[Callable[S, T]], _CachedFunction[S, T]]:
    def decorator(func: Callable[S, T]) -> _CachedFunction[S, T]:
        return cast(_CachedFunction[S, T], lru_cache(maxsize=maxsize)(func))

    register_backend_cache(decorator, "cache")
    return decorator
