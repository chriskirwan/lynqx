import os


os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax
import pytest
from hypothesis import HealthCheck, settings


# JIT compilation can comfortably blow past hypothesis's default 200ms deadline -- that's a compile-time cost, not a
# regression, and treating it as one just makes the suite flaky.
settings.register_profile("default", deadline=None, suppress_health_check=[HealthCheck.too_slow])
settings.register_profile("ci", deadline=None, max_examples=200)
settings.register_profile("dev", deadline=None, max_examples=25)
settings.load_profile("default")


@pytest.fixture(scope="session")
def mesh_1d():
    """A 1D mesh named ``x`` spanning every (simulated) device."""
    return jax.make_mesh((len(jax.devices()),), ("x",))


@pytest.fixture(scope="session")
def mesh_2d():
    """A 2D ``(x, y)`` mesh, factored as evenly as the device count allows."""
    n = len(jax.devices())
    for x in range(int(n**0.5), 0, -1):
        if n % x == 0:
            return jax.make_mesh((x, n // x), ("x", "y"))
    return jax.make_mesh((1, n), ("x", "y"))
