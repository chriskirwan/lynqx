import os


os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax  # noqa: E402
import pytest  # noqa: E402


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
