"""B-spline basis mathematics (NumPy, float64).

Everything here is structural: it depends only on knot vectors and degrees,
never on control-point values. All refinement operations are expressed as
explicit linear maps ``P_new = A @ P_old`` so they can be composed into the
global DOF map and reasoned about exactly.

Conventions
-----------
* Knot vectors are clamped (first/last knot repeated ``degree + 1`` times).
* A knot vector of length ``m + 1`` with degree ``p`` has ``n = m - p``
  control points / basis functions.
* Knot values that differ by less than ``KNOT_TOL`` are treated as equal.
"""
from __future__ import annotations

import numpy as np

KNOT_TOL = 1e-9


# ---------------------------------------------------------------------------
# Knot-vector utilities
# ---------------------------------------------------------------------------

def clamped_uniform_knots(n_interior: int, degree: int = 3, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """Clamped knot vector with ``n_interior`` uniformly spaced interior knots."""
    inner = np.linspace(lo, hi, n_interior + 2)[1:-1]
    return np.concatenate([np.full(degree + 1, lo), inner, np.full(degree + 1, hi)])


def num_basis(knots: np.ndarray, degree: int) -> int:
    return len(knots) - degree - 1


def interior_knots(knots: np.ndarray, degree: int) -> np.ndarray:
    return np.asarray(knots[degree + 1: len(knots) - degree - 1], dtype=float)


def multiplicity(knots: np.ndarray, t: float, tol: float = KNOT_TOL) -> int:
    return int(np.sum(np.abs(np.asarray(knots) - t) <= tol))


def snap_knots(knots: np.ndarray, tol: float = KNOT_TOL) -> np.ndarray:
    """Snap near-equal knots to a common value and clamp to [knots[0], knots[-1]].

    Rescaling knot vectors (e.g. after a face split) introduces floating-point
    drift; snapping keeps multiplicities well defined.
    """
    k = np.array(knots, dtype=float)
    lo, hi = k[0], k[-1]
    k = np.clip(k, lo, hi)
    k[np.abs(k - lo) <= tol] = lo
    k[np.abs(k - hi) <= tol] = hi
    for i in range(1, len(k)):
        if abs(k[i] - k[i - 1]) <= tol:
            k[i] = k[i - 1]
    return k


def knot_multiset_difference(a: np.ndarray, b: np.ndarray, tol: float = KNOT_TOL) -> np.ndarray:
    """Return the elements of sorted multiset ``a`` not matched in sorted multiset ``b``."""
    out = []
    i = j = 0
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    while i < len(a):
        if j < len(b) and abs(a[i] - b[j]) <= tol:
            i += 1
            j += 1
        elif j < len(b) and b[j] < a[i]:
            j += 1
        else:
            out.append(a[i])
            i += 1
    return np.array(out, dtype=float)


def is_knot_subset(small: np.ndarray, big: np.ndarray, tol: float = KNOT_TOL) -> bool:
    """True if multiset ``small`` is contained in multiset ``big``."""
    return len(knot_multiset_difference(small, big, tol)) == 0


def affine_map_knots(knots: np.ndarray, a: float, b: float, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """Map knot values from interval [a, b] to [lo, hi]."""
    k = lo + (np.asarray(knots, dtype=float) - a) * (hi - lo) / (b - a)
    return snap_knots(k)


def reverse_knots(knots: np.ndarray) -> np.ndarray:
    """Knot vector of the reversed curve on the same parameter interval."""
    k = np.asarray(knots, dtype=float)
    return snap_knots((k[0] + k[-1]) - k[::-1])


def greville(knots: np.ndarray, degree: int) -> np.ndarray:
    n = num_basis(knots, degree)
    return np.array([np.mean(knots[i + 1: i + degree + 1]) for i in range(n)])


# ---------------------------------------------------------------------------
# Basis evaluation (The NURBS Book, algorithms A2.1 / A2.3)
# ---------------------------------------------------------------------------

def find_span(knots: np.ndarray, degree: int, u: float) -> int:
    """Index k with knots[k] <= u < knots[k+1] (the last non-empty span at the end)."""
    n = num_basis(knots, degree) - 1
    k = int(np.searchsorted(knots, u, side="right")) - 1
    return int(min(max(k, degree), n))


def basis_funs_ders(knots: np.ndarray, degree: int, u: float, n_ders: int = 0):
    """Non-zero basis functions and derivatives at ``u``.

    Returns ``(span, ders)`` with ``ders[k, j]`` the k-th derivative of
    ``N_{span - degree + j}`` at ``u``.
    """
    p = degree
    span = find_span(knots, p, u)
    ndu = np.zeros((p + 1, p + 1))
    left = np.zeros(p + 1)
    right = np.zeros(p + 1)
    ndu[0, 0] = 1.0
    for j in range(1, p + 1):
        left[j] = u - knots[span + 1 - j]
        right[j] = knots[span + j] - u
        saved = 0.0
        for r in range(j):
            ndu[j, r] = right[r + 1] + left[j - r]
            temp = ndu[r, j - 1] / ndu[j, r]
            ndu[r, j] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        ndu[j, j] = saved
    n_ders_eff = min(n_ders, p)
    ders = np.zeros((n_ders + 1, p + 1))
    ders[0, :] = ndu[:, p]
    a = np.zeros((2, p + 1))
    for r in range(p + 1):
        s1, s2 = 0, 1
        a[0, 0] = 1.0
        for k in range(1, n_ders_eff + 1):
            d = 0.0
            rk, pk = r - k, p - k
            if r >= k:
                a[s2, 0] = a[s1, 0] / ndu[pk + 1, rk]
                d = a[s2, 0] * ndu[rk, pk]
            j1 = 1 if rk >= -1 else -rk
            j2 = k - 1 if r - 1 <= pk else p - r
            for j in range(j1, j2 + 1):
                a[s2, j] = (a[s1, j] - a[s1, j - 1]) / ndu[pk + 1, rk + j]
                d += a[s2, j] * ndu[rk + j, pk]
            if r <= pk:
                a[s2, k] = -a[s1, k - 1] / ndu[pk + 1, r]
                d += a[s2, k] * ndu[r, pk]
            ders[k, r] = d
            s1, s2 = s2, s1
    r = p
    for k in range(1, n_ders_eff + 1):
        ders[k, :] *= r
        r *= p - k
    return span, ders


def basis_matrix(knots: np.ndarray, degree: int, params, deriv: int = 0) -> np.ndarray:
    """Dense matrix ``B[i, a] = d^deriv N_a / du^deriv (params[i])``.

    Vectorized Cox-de Boor over all parameters (degree-raising recursion on
    full arrays); derivatives use  N'_{i,q} = q/(u_{i+q}-u_i) N_{i,q-1}
    - q/(u_{i+q+1}-u_{i+1}) N_{i+1,q-1}  applied ``deriv`` times. Parameters at
    the right end use the last non-empty span (as ``find_span``).
    """
    U = np.asarray(knots, dtype=float)
    u = np.atleast_1d(np.asarray(params, dtype=float))[:, None]
    p = degree
    n = num_basis(U, p)
    if deriv > p:
        return np.zeros((len(u), n))
    span = np.clip(np.searchsorted(U, u[:, 0], side="right") - 1, p, n - 1)
    N = np.zeros((len(u), len(U) - 1))
    N[np.arange(len(u)), span] = 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        for q in range(1, p - deriv + 1):
            nf = len(U) - 1 - q
            ld = U[q: q + nf] - U[:nf]
            rd = U[q + 1: q + 1 + nf] - U[1: 1 + nf]
            a = np.where(ld > 0, (u - U[:nf]) / np.where(ld > 0, ld, 1.0), 0.0)
            b = np.where(rd > 0, (U[q + 1: q + 1 + nf] - u) / np.where(rd > 0, rd, 1.0), 0.0)
            N = a * N[:, :nf] + b * N[:, 1: nf + 1]
        for q in range(p - deriv + 1, p + 1):
            nf = len(U) - 1 - q
            ld = U[q: q + nf] - U[:nf]
            rd = U[q + 1: q + 1 + nf] - U[1: 1 + nf]
            c1 = np.where(ld > 0, q / np.where(ld > 0, ld, 1.0), 0.0)
            c2 = np.where(rd > 0, q / np.where(rd > 0, rd, 1.0), 0.0)
            N = c1 * N[:, :nf] - c2 * N[:, 1: nf + 1]
    return N[:, :n]


def basis_matrix_reference(knots: np.ndarray, degree: int, params, deriv: int = 0) -> np.ndarray:
    """Per-point evaluation with The NURBS Book A2.3 (reference for tests)."""
    knots = np.asarray(knots, dtype=float)
    params = np.atleast_1d(np.asarray(params, dtype=float))
    n = num_basis(knots, degree)
    B = np.zeros((len(params), n))
    for i, u in enumerate(params):
        span, ders = basis_funs_ders(knots, degree, float(u), deriv)
        B[i, span - degree: span + 1] = ders[deriv]
    return B


def evaluate_curve(knots: np.ndarray, degree: int, cps: np.ndarray, params, deriv: int = 0) -> np.ndarray:
    return basis_matrix(knots, degree, params, deriv) @ np.asarray(cps)


# ---------------------------------------------------------------------------
# Exact refinement as linear maps
# ---------------------------------------------------------------------------

def insertion_matrix(knots: np.ndarray, degree: int, t: float):
    """Boehm single knot insertion as a linear map.

    Returns ``(A, new_knots)`` with ``P_new = A @ P_old`` and the curve
    unchanged. Works for any current multiplicity of ``t`` below ``degree + 1``.
    """
    knots = np.asarray(knots, dtype=float)
    p = degree
    n = num_basis(knots, p)
    k = find_span(knots, p, t)
    A = np.zeros((n + 1, n))
    for i in range(n + 1):
        if i <= k - p:
            A[i, i] = 1.0
        elif i >= k + 1:
            A[i, i - 1] = 1.0
        else:
            denom = knots[i + p] - knots[i]
            alpha = 0.0 if denom == 0.0 else (t - knots[i]) / denom
            A[i, i] = alpha
            A[i, i - 1] = 1.0 - alpha
    new_knots = np.insert(knots, k + 1, t)
    return A, new_knots


def insert_times(knots: np.ndarray, degree: int, t: float, times: int):
    """Insert ``t`` ``times`` times. Returns ``(A, new_knots)`` with ``P_new = A @ P_old``."""
    A = np.eye(num_basis(knots, degree))
    k = np.asarray(knots, dtype=float)
    for _ in range(times):
        Ai, k = insertion_matrix(k, degree, t)
        A = Ai @ A
    return A, k


def refinement_matrix(knots_old: np.ndarray, knots_new: np.ndarray, degree: int) -> np.ndarray:
    """Linear map from control points on ``knots_old`` to ``knots_new`` (a superset)."""
    knots_old = np.asarray(knots_old, dtype=float)
    knots_new = np.asarray(knots_new, dtype=float)
    extra = knot_multiset_difference(knots_new, knots_old)
    if len(knots_new) - len(knots_old) != len(extra):
        raise ValueError("knots_new is not a refinement of knots_old")
    A = np.eye(num_basis(knots_old, degree))
    k = knots_old
    for t in extra:
        Ai, k = insertion_matrix(k, degree, float(t))
        A = Ai @ A
    if len(k) != len(knots_new) or np.max(np.abs(k - knots_new)) > 10 * KNOT_TOL:
        raise ValueError("refinement produced unexpected knot vector")
    return A


def interpolating_index(knots: np.ndarray, degree: int, t: float) -> int:
    """Index of the control point the curve interpolates at ``t``.

    Requires ``t`` to be an end knot or an interior knot of multiplicity
    exactly ``degree``.
    """
    knots = np.asarray(knots, dtype=float)
    n = num_basis(knots, degree)
    if abs(t - knots[0]) <= KNOT_TOL:
        return 0
    if abs(t - knots[-1]) <= KNOT_TOL:
        return n - 1
    first = int(np.argmax(np.abs(knots - t) <= KNOT_TOL))
    if multiplicity(knots, t) != degree:
        raise ValueError(f"knot {t} must have multiplicity {degree}")
    return first - 1


def insert_to_multiplicity(knots: np.ndarray, degree: int, t: float, mult: int):
    """Insert ``t`` until its multiplicity is ``mult``. Returns ``(A, new_knots)``."""
    knots = np.asarray(knots, dtype=float)
    A = np.eye(num_basis(knots, degree))
    k = knots
    while multiplicity(k, t) < mult:
        Ai, k = insertion_matrix(k, degree, t)
        A = Ai @ A
    return A, k


def segment_matrix(knots: np.ndarray, degree: int, a: float, b: float):
    """Exact extraction of the curve restricted to [a, b], reparameterized to [0, 1].

    Returns ``(A, seg_knots)`` with segment control points ``A @ P``.
    """
    knots = np.asarray(knots, dtype=float)
    p = degree
    A = np.eye(num_basis(knots, p))
    k = knots
    for t in (a, b):
        if knots[0] + KNOT_TOL < t < knots[-1] - KNOT_TOL:
            Ai, k = insert_to_multiplicity(k, p, t, p)
            A = Ai @ A
    ia = interpolating_index(k, p, a)
    ib = interpolating_index(k, p, b)
    inner = [x for x in k if a + KNOT_TOL < x < b - KNOT_TOL]
    seg = np.concatenate([np.full(p + 1, a), inner, np.full(p + 1, b)])
    if ib - ia + 1 != num_basis(seg, p):
        raise RuntimeError("segment extraction index mismatch")
    return A[ia: ib + 1], affine_map_knots(seg, a, b)
