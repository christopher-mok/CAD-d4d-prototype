import numpy as np

from cad_d4d.geometry.builders import build_cube_complex, ellipsoid_map


def perturbed_state(seed=0, sigma=0.05, n_knots=0):
    s = build_cube_complex(ellipsoid_map((1.0, 0.85, 0.7)), n_interior_knots=n_knots)
    rng = np.random.default_rng(seed)
    s.set_values(s.values() + sigma * rng.normal(size=s.values().shape))
    return s


def grid(n=41):
    t = np.linspace(0, 1, n)
    U, V = np.meshgrid(t, t, indexing="ij")
    return np.stack([U.ravel(), V.ravel()], 1)


def same_face_deviation(a, b, fids, n=41):
    uv = grid(n)
    return max(float(np.max(np.linalg.norm(a.evaluate(f, uv) - b.evaluate(f, uv), axis=1))) for f in fids)


def dense_surface_points(state, n=25):
    uv = grid(n)
    return np.concatenate([state.evaluate(f, uv) for f in sorted(state.cx.faces)])


def surface_distance(a_pts, b_pts):
    """Max over a_pts of distance to the nearest of b_pts (one-sided, sample-based)."""
    from scipy.spatial import cKDTree
    d, _ = cKDTree(b_pts).query(a_pts)
    return float(d.max())
