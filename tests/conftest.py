import pytest

from cad_d4d.geometry.builders import build_cube_complex, sphere_map
from cad_d4d.targets.synthetic import TargetConfig, compose, make_synthetic_target, perturb_dofs, scale_dofs


@pytest.fixture(scope="session")
def coarse_sphere():
    return build_cube_complex()


@pytest.fixture(scope="session")
def reachable_target(coarse_sphere):
    """A target inside the coarse representation's reachable space."""
    return make_synthetic_target(
        coarse_sphere, perturb=compose(scale_dofs((1.2, 0.9, 0.8)), perturb_dofs(0.05)),
        cfg=TargetConfig(sdf_res=48, n_coverage=1500))


@pytest.fixture(scope="session")
def fine_sphere_target():
    """Uniformly refined cube-sphere: a close approximation of the unit sphere."""
    s = build_cube_complex(sphere_map(), n_interior_knots=3)
    from cad_d4d.targets.synthetic import Target
    return Target.from_state(s, TargetConfig(sdf_res=56, n_coverage=1500))
