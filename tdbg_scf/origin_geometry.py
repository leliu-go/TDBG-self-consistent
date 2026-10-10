"""Periodic interpolation and explicitly labelled C3 plotting completion."""
from __future__ import annotations

import numpy as np
import warnings
from scipy.interpolate import RegularGridInterpolator
from matplotlib.figure import Figure


def first_mbz_boundary(geom):
    """Closed Wigner-Seitz hexagon of the triangular moire reciprocal lattice."""
    b1, b2 = np.asarray(geom.b1m), np.asarray(geom.b2m)
    vertices = np.array([b1-b2, 2*b1+b2, b1+2*b2,
                         -b1+b2, -2*b1-b2, -b1-2*b2]) / 3
    return np.vstack([vertices, vertices[0]])


def square_mesh(geom, points=145, padding=1.2):
    if isinstance(points, bool) or int(points) != points or points < 3:
        raise ValueError("square_points must be an integer >= 3")
    if not np.isfinite(padding) or padding <= 1:
        raise ValueError("square padding must exceed one")
    half_width = padding * np.max(np.abs(first_mbz_boundary(geom)))
    axis = np.linspace(-half_width, half_width, int(points))
    x, y = np.meshgrid(axis, axis, indexing="xy")
    return x, y, np.column_stack([x.ravel(), y.ravel()])


class UndersampledReciprocalGrid(ValueError):
    """A valid cell grid with too few samples for a two-dimensional surface."""


class PeriodicBandSurface:
    """Bilinear interpolation of an existing complete reciprocal midpoint grid.

    C3 completion averages the three rotated periodic plotting interpolants.
    It is a plotting approximation for the non-C3-closed finite square cutoff.
    The ordinary call preserves every original sample and uses no C3 completion.
    """

    def __init__(self, geom, kpts, values):
        self.geom = geom
        self.basis = np.column_stack([geom.b1m, geom.b2m])
        self.inverse_basis = np.linalg.inv(self.basis)
        kpts, values = np.asarray(kpts, float), np.asarray(values, float)
        if (kpts.ndim != 2 or kpts.shape[1] != 2 or values.shape[0] != len(kpts)
                or values.ndim not in (1, 2) or not np.isfinite(kpts).all()
                or not np.isfinite(values).all()):
            raise ValueError("require finite kpts (Nk,2) and matching band values")
        uv = kpts @ self.inverse_basis.T
        rounded = np.round(uv, 13)
        u, v = np.unique(rounded[:,0]), np.unique(rounded[:,1])
        if len(u)*len(v) != len(kpts):
            raise ValueError("contour samples must form a complete rectangular reciprocal grid")
        for axis in (u, v):
            if not np.allclose(np.diff(axis), 1/len(axis), rtol=1e-9, atol=1e-12):
                raise ValueError("contour grid must span exactly one uniform reciprocal cell")
        i, j = np.searchsorted(u, rounded[:,0]), np.searchsorted(v, rounded[:,1])
        if len(np.unique(i*len(v)+j)) != len(kpts):
            raise ValueError("contour grid contains duplicate samples")
        if len(u) < 2 or len(v) < 2:
            raise UndersampledReciprocalGrid("contour interpolation needs at least 2 samples on each reciprocal axis")
        grid = np.empty((len(u),len(v)) + values.shape[1:])
        grid[i,j] = values
        expanded = np.concatenate([grid,grid[:1]], axis=0)
        expanded = np.concatenate([expanded,expanded[:,:1]], axis=1)
        self.lower = np.array([u[0],v[0]])
        self.interpolator = RegularGridInterpolator(
            (np.r_[u,u[0]+1],np.r_[v,v[0]+1]), expanded,
            method="linear", bounds_error=True)

    def __call__(self, points, c3_complete=False):
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError("surface queries must be finite with shape (N,2)")
        if c3_complete:
            angle = 2*np.pi/3
            rotation = np.array([[np.cos(angle),-np.sin(angle)],
                                 [np.sin(angle),np.cos(angle)]])
            return (self(points)+self(points @ rotation.T)
                    +self(points @ (rotation @ rotation).T))/3
        uv = points @ self.inverse_basis.T
        wrapped = (uv-self.lower) % 1 + self.lower
        return self.interpolator(wrapped)

    def c3_mismatch(self, points):
        """Mismatch of the raw periodic interpolation, not a convergence test."""
        angle = 2*np.pi/3
        rotation = np.array([[np.cos(angle),-np.sin(angle)],
                             [np.sin(angle),np.cos(angle)]])
        error = np.abs(self(points)-self(np.asarray(points) @ rotation.T))
        return {"maximum_meV": float(np.max(error)),
                "median_meV": float(np.median(error)),
                "includes_grid_interpolation_error": True}


def _refine_line_vertices(lines, x, y, energy_function, tolerance):
    """Refine crossings on their original mesh edges, preserving connectivity."""
    if not lines:
        return lines
    points = np.vstack(lines).copy()
    x_axis, y_axis = np.asarray(x[0], float), np.asarray(y[:, 0], float)
    dx, dy = x_axis[1]-x_axis[0], y_axis[1]-y_axis[0]
    if (dx <= 0 or dy <= 0 or not np.allclose(np.diff(x_axis), dx)
            or not np.allclose(np.diff(y_axis), dy)):
        raise ValueError("line refinement requires a uniform increasing square mesh")
    ix = np.clip(np.rint((points[:, 0]-x_axis[0])/dx).astype(int), 0, len(x_axis)-1)
    iy = np.clip(np.rint((points[:, 1]-y_axis[0])/dy).astype(int), 0, len(y_axis)-1)
    distance_x = np.abs(points[:, 0]-x_axis[ix])/dx
    distance_y = np.abs(points[:, 1]-y_axis[iy])/dy
    near_vertical = distance_x <= max(1e-8, 1e-14/dx)
    near_horizontal = distance_y <= max(1e-8, 1e-14/dy)
    if not np.all(near_vertical | near_horizontal):
        raise ValueError("contour vertices must lie on the original mesh edges")
    # A horizontal crossing can be extremely close to a corner. Choose the
    # actually nearest mesh axis instead of giving the vertical edge priority.
    vertical = near_vertical & (~near_horizontal | (distance_x <= distance_y))
    ilo = np.clip(np.searchsorted(x_axis, points[:, 0], side="right")-1, 0, len(x_axis)-2)
    jlo = np.clip(np.searchsorted(y_axis, points[:, 1], side="right")-1, 0, len(y_axis)-2)
    lower = np.column_stack([np.where(vertical, x_axis[ix], x_axis[ilo]),
                             np.where(vertical, y_axis[jlo], y_axis[iy])])
    upper = np.column_stack([np.where(vertical, x_axis[ix], x_axis[ilo+1]),
                             np.where(vertical, y_axis[jlo+1], y_axis[iy])])

    def evaluate(vertices):
        values = np.asarray(energy_function(vertices), float)
        if values.shape != (len(vertices),) or not np.isfinite(values).all():
            raise ValueError("line energy callback must return one finite value per vertex")
        return values

    f_lower, f_upper = evaluate(lower), evaluate(upper)
    active = np.abs(evaluate(points)) > tolerance
    for endpoint, values in ((lower, f_lower), (upper, f_upper)):
        at_endpoint = active & (np.abs(values) <= tolerance)
        points[at_endpoint] = endpoint[at_endpoint]
        active[at_endpoint] = False
    if np.any(active & (f_lower*f_upper > 0)):
        raise ValueError("Fermi crossing is not bracketed on its original mesh edge")
    for _ in range(60):
        ids = np.flatnonzero(active)
        if not len(ids):
            break
        midpoint = (lower[ids]+upper[ids])/2
        values = evaluate(midpoint)
        points[ids] = midpoint
        converged = np.abs(values) <= tolerance
        active[ids[converged]] = False
        remaining = ids[~converged]
        values = values[~converged]
        midpoint = midpoint[~converged]
        same_sign = np.signbit(values) == np.signbit(f_lower[remaining])
        lower[remaining[same_sign]] = midpoint[same_sign]
        f_lower[remaining[same_sign]] = values[same_sign]
        upper[remaining[~same_sign]] = midpoint[~same_sign]
    if np.any(np.abs(evaluate(points)) > tolerance):
        raise ValueError("Fermi line vertex refinement did not meet the energy tolerance")
    boundaries = np.cumsum([len(line) for line in lines])[:-1]
    return list(np.split(points, boundaries))


def zero_level_lines(x, y, energy_relative_mu, *, energy_function=None, tolerance_meV=1e-7):
    """Each returned array is one connected polyline; no joins across pockets."""
    z = np.asarray(energy_relative_mu, float)
    if z.shape != x.shape or z.shape != y.shape or not np.isfinite(z).all():
        raise ValueError("contour mesh must be finite and match its energy surface")
    minimum,maximum = float(np.min(z)),float(np.max(z))
    if not minimum <= 0 <= maximum or minimum == maximum:
        return []
    # Matplotlib handles a touching minimum, but substitutes the minimum for a
    # touching maximum. Flip the latter without changing the zero-level set.
    if maximum == 0:
        z = -z
    figure = Figure()
    axes = figure.subplots()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",message="No contour levels were found within the data range",category=UserWarning)
        contours = axes.contour(x,y,z,levels=[0.])
    lines,seen = [],set()
    for segment in contours.allsegs[0]:
        segment = np.asarray(segment,float)
        if len(segment) < 2 or not np.any(np.linalg.norm(np.diff(segment,axis=0),axis=1)>1e-14):
            continue
        rounded = np.round(segment,12)
        key = min(tuple(rounded.ravel()),tuple(rounded[::-1].ravel()))
        if key not in seen:
            lines.append(segment)
            seen.add(key)
    figure.clear()
    if energy_function is not None:
        if not np.isfinite(tolerance_meV) or tolerance_meV <= 0:
            raise ValueError("contour vertex tolerance must be positive and finite")
        lines = _refine_line_vertices(lines, x, y, energy_function, tolerance_meV)
    return lines


def pack_lines(segments):
    """Pad independent XY column pairs with NaN, preserving their order."""
    if not segments:
        return np.empty((0,2))
    result = np.full((max(len(segment) for segment in segments),2*len(segments)),np.nan)
    for i, segment in enumerate(segments):
        result[:len(segment),2*i:2*i+2] = segment
    return result
