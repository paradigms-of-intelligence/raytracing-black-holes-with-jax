# Copyright 2026 Google LLC
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Minimalistic self-contained JAX-based General Relativity Raytracer.

Many functions in this module take or return arguments that themselves
are functions with specific rule in General Relativity. In order to
eliminate repetitiveness in the documentation, these are described
upfront here, with individual functions referring to the
module-docstring where needed. The key definitions are:

  L2_func: Callable `f(coords_pos, coords_vec) -> len_sq` which
    computes the length-squared of a local (at position given by
    the `[dim]-jax.Array` `coords_pos`) vector with
    'spatial coordinate rate-of-change per curve-parameter rate-of-change'
    coordinates `coords_vec`, given as another `[dim]-jax.Array`.

  metric_func: Callable `g_func(coords_pos) -> coords_metric`
    that maps position-coordinates (given as a `float64 [dim]-jax.Array`)
    to metric-tensor coordinates (as a `float64 [dim, dim]-jax.Array`).

  christoffel_func: Callable `christoffel(coords_pos) -> christoffel_symbols`
    that maps position-coordinates  (given as a `[dim]-jax.Array`) to a
    `[dim, dim, dim]-jax.Array` - henceforth `Gamma` - containing the
    Christoffel symbols of the 2nd kind at that point. In conventional
    notation: `Gamma[m, n, p] = \Gamma^m_np`.

  ode_y_func: Callable `F(y) -> rate_of_change_of_y` for numerical
    ODE-integration. Here, `y` is a `jax.Array` such that
    `tuple(y.reshape(2, dim))` is the tuple `(coords_pos, coords_tangent)`
    containing position and tangent vector data.

The term 'basis vector' refers to a tangent vector on a coordinate-line curve.
"""

import pdb  # For: `python3 -i CODE.py` and `pdb.pm()` inspection on `raise ...`
import itertools, matplotlib.pyplot, numpy, os, warnings

import jax
from jax import numpy as jnp
jax.config.update("jax_enable_x64", True)


def metric_func_from_L2_func(L2_func):
  "Computes `metric_func` function from `L2_func` function."
  d2_L2_by_dxi_dxj_func = jax.hessian(L2_func, argnums=1)
  def metric_func(coords_pos):
    """Maps X_i -> M_ij, with X=position, M=metric tensor."""
    return 0.5 * d2_L2_by_dxi_dxj_func(coords_pos, jnp.zeros_like(coords_pos))
  return metric_func


def christoffel_func_from_metric_func(metric_func):
  """Maps 'metric tensor' function to 'Christoffel symbols' function.

  Args:
    metric_func: See `metric_func` in module docstring.
  Returns:
    See `christoffel_func` in module docstring.
  """
  # {coords} -> d {metric} / d {coord}
  d_metric_d_coords_func = jax.jacobian(metric_func)
  def christoffel_func(coords_pos):
    """Maps X_i -> C_ijk, with X=position, C=Christoffel symbols of 2nd kind."""
    # These are the g_ij,k (or M_ij,k):
    d_metric_d_coords_pos = d_metric_d_coords_func(coords_pos)
    # We use jnp.einsum() rather than Array.transpose() and also
    # jnp.linalg.inv() rather than jnp.linalg.solve() for better
    # alignment with the text.
    return 0.5 * jnp.einsum('im,mkl->ikl',
                            jnp.linalg.inv(metric_func(coords_pos)),
                            d_metric_d_coords_pos
                            + jnp.einsum('mlk->mkl', d_metric_d_coords_pos)
                            - jnp.einsum('klm->mkl', d_metric_d_coords_pos))
  return christoffel_func


def geodesic_dy_dt_func_from_christoffel_func(christoffel_func):
  "Computes `ode_y_func` function from `christoffel_func` function."
  def dy_dt_func(motion_state_y):
    """Maps Y_i -> {rate-of-change of Y_i}."""
    c_pos, c_tangent = jnp.split(motion_state_y, 2)
    rate_tangent = -jnp.einsum('ijk,j,k->i', christoffel_func(c_pos),
                               c_tangent, c_tangent)
    return jnp.concatenate([c_tangent, rate_tangent], axis=0)
  return dy_dt_func


def geodesic_dy_dt_func_from_metric_func(metric_func):
  "Computes `ode_y_func` function from `metric_func` function."
  christoffel_func = christoffel_func_from_metric_func(metric_func)
  return geodesic_dy_dt_func_from_christoffel_func(christoffel_func)    


# This actually generalizes to a wider class of functions than `ode_y_func`.
def rk4_estimate(f, y0, ds):
  """Estimates average `dy/dt` over step `ds` via Runge-Kutta RK4.

  Args:
    f: See `ode_y_func` in module docstring.
    y0: `jax.Array` representing the current motion-state.
    ds: step-size parameter interval. We try to estimate the average `dy/dt`
        over a step of this size.

  Returns:
    RK4 estimate of average `dy/dt` when using `f` to propagate motion-state
    over interval `ds`.
  """
  k1 = f(y0)
  k2 = f(y0 + (0.5 * ds) * k1)
  k3 = f(y0 + (0.5 * ds) * k2)
  k4 = f(y0 + ds * k3)
  return (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def ode_solve(estimate_func, f, y0, s_samples):
  """Numerically integrates an ordinary differential equation.

  Args:
    estimate_func: mean-rate-of-change estimating function.
      Must have the same signature as `rk4_estimate`.
    f: See `ode_y_func` in module docstring.
    y0: `jax.Array` representing the current motion-state.
    s_samples: 1-axis `jax.Array` with coordinate-parameter sample-values.
      ODE-integration attributes motion-state `y0` to parameter-value
      `s_samples[0]` and then proceeds in steps of
      `s_samples[n+1] - s_samples[n]`.
  Returns:
    `[num_steps, *y0.shape()]-jax.Array` `result` with collected
    motion-state-at-end-of-every-step data. Here, `result[k]` corresponds to
    parameter-value `s_samples[k+1]`, and `num_steps + 1 == s_samples.size`.
  """
  def inner_func(y_now, s_step):
    rate = estimate_func(f, y_now, s_step)
    y_next = y_now + s_step * rate
    return y_next, y_next
  y_final, ys_collected = jax.lax.scan(inner_func, y0, jnp.diff(s_samples))
  return ys_collected


def riemann_func_from_christoffel_func(christoffel_func):
  """Computes Riemannian curvature function from Christoffel function.

  Args:
    christoffel_func: See `christoffel_func` in module docstring.

  Returns:
    Callable `R(coords_pos) -> curvature_tensor` that maps position to
    a `[dim, dim, dim, dim]-jax.Array` - henceforth `Riemann` - containing
    the Riemann curvature tensor. In conventional GR notation:
    `Riemann[m, n, p, q] == R^m_npq`.
  """
  jacobian_christoffel_func = jax.jacobian(christoffel_func)
  def riemann_func(coords_pos):
    """Maps X_i -> R^m_npq, where X=position, R=Riemann curvature tensor."""
    christoffel = christoffel_func(coords_pos)
    d_christoffel = jacobian_christoffel_func(coords_pos)
    # We use jnp.einsum() throughout, also for simple transposition.
    aux = (jnp.einsum('ijlk->ijkl', d_christoffel) +
           jnp.einsum('ikm,mjl->ijkl', christoffel, christoffel))
    # The whole expression is aux_ijkl - aux_ijlk.
    return aux - jnp.einsum('ijkl->ijlk', aux)
  return riemann_func


def einstein_func_from_L2_func(L2_func):
  """Computes Einstein Tensor function from Line Element function.

  Args:
    L2_func: See `L2_func` in module docstring.

  Returns:
    Callable `G(coords_pos) -> einstein_tensor` that maps position to
    a `[dim, dim]-jax.Array` - henceforth `Einstein` - containing
    the Einstein curvature tensor. In conventional GR notation:
    `Einstein[m, n] == G_mn`.
  """
  metric_func = metric_func_from_L2_func(L2_func)
  christoffel_func = christoffel_func_from_metric_func(metric_func)
  riemann_func = riemann_func_from_christoffel_func(christoffel_func)
  def einstein_func(coords):
    """Maps X_i -> G^mn, where X=position, G=Einstein tensor."""
    metric = metric_func(coords)
    riemann = riemann_func(coords)
    ricci = jnp.einsum('mimj->ij', riemann)
    return (ricci - 0.5 * jnp.einsum('ij,ij->', ricci, jnp.linalg.inv(metric))
            * metric)
  return einstein_func


def schwarzschild_L2_func(txyz, d_txyz, r_schwarzschild=1.0):
  """Computes the local-vector length-squared for Schwarzschild geometry.

  Schwarzschild geometry is given by:
  ```
  ds**2 = -F(r)*dt**2 + dr**2/F(r) +
           + r**2 d_theta**2 + r**2 * sin(theta)**2 d_phi**2
  ```

  ...where here, we have `F(r) = 1/(1-r_schwarzschild/r)`, `r*cos(theta)=z`,
  `r*sin(theta)*cos(phi)=x`, `r*sin(theta)*sin(phi)=y`, i.e. `theta`
  is the angle from the 3d "north" axis.
  
  Args:
    txyz: `[4]-jax.Array` position coordinate-vector `(t, x, y, z)`.
    d_txyz: `[4]-jax.Array`, tangent vector (w.r.t. local coordinate
      rate-of-change vector space basis).
    r_schwarzschild: The "Schwarzschild radius".

  Returns:
    The value of the "space-time" (local Minkowski) pseudo-Euclidean
    scalar product of the vector with coordinates `d_txyz` with itself,
    or `jax.numpy.nan` if `r <= r_schwarzschild`.
  """
  xyz = txyz[1:]
  r = jnp.linalg.norm(xyz)
  e_r = xyz / r
  part_e_r = (d_txyz[1:] @ e_r) * e_r
  part_perpendicular = d_txyz[1:] - part_e_r
  ds2_perpendicular = jnp.square(part_perpendicular).sum()
  s_factor = jnp.where(r > r_schwarzschild, 1 - r_schwarzschild / r, jnp.nan)
  ds2_radial = jnp.square(part_e_r).sum() / s_factor
  ds2_time_abs = jnp.square(d_txyz[0]) * s_factor
  return ds2_radial + ds2_perpendicular - ds2_time_abs


# Validation of the construction - can be removed:
schwarzschild_einstein_func = einstein_func_from_L2_func(schwarzschild_L2_func)
jax_schwarzschild_G = jax.jit(schwarzschild_einstein_func)
einstein1 = jax_schwarzschild_G(jnp.array([5.0, 2.0, 3.0, 1.0]))
print('=== Einstein tensor at an arbitrary point (should be zero) ===')
print(einstein1.round(12))

#### Scaffolding

def get_orthonormal_frame(metric_func, txyz, frame_raw):
  """Obtains an orthonormal coordinate frame from "future" / "forward" / "up".

  Here, `coords_dir_*` vectors are at point `txyz` and w.r.t. `txyz`
  coordinate-lines tangent vector basis.
  
  Args:
    metric_func: See `metric_func` in module docstring.
    txyz: Space-time coordinates of the camera.
    frame_raw: [4, 4]-ArrayLike `r` such that `r[:, 0]` will be proportional
      to the "forward-in-time" unit-vector of the resulting orthonormal frame,
      the projection of `r[:, 1]` to the subspace orthogonal to `r[:, 0]`
      will be proportional to the resulting orthonormal frame's "x" ("forward")
      direction, and the projection of `r[:, 3]` to the subspace orthogonal to
      the "t" and "x" frame-directions will be proportional to "z".
  """
  g = metric_func(txyz)
  sprod = lambda v1, v2: numpy.einsum('ij,i,j->', g, v1, v2)
  frame_raw_txzy = numpy.asarray(frame_raw,
                                 dtype=numpy.float64)[:, (0, 1, 3, 2)]
  basis_vecs = []
  for n, b in enumerate(frame_raw_txzy.T):
    b_now = b
    for b_earlier in basis_vecs:
      b_now = b_now - b_earlier * (sprod(b_now, b_earlier) /
                                   sprod(b_earlier, b_earlier))
    basis_vecs.append(b_now / abs(sprod(b_now, b_now))**.5)
  return numpy.stack(basis_vecs, axis=1)[:, (0, 1, 3, 2)]


def _get_ray_eye_ends(txyz_end, frame, dyzs, dx=1, dt=1, c=1):
  # Find trajectory end-state-vectors that reach us going
  # in the 3d direction (w.r.t. local frame) `-(-dx, y, z)`.
  # (We ODE-integrate rays "backwards in time").
  dyzs_flat = dyzs.reshape(-1, 2)  # indexed `[num_pixel, yz_coord]`.
  dxyzs_flat = jnp.pad(-dyzs_flat, ((0, 0), (1, 0)), constant_values=dx)
  # We are here working with spatial coordinates in a locally-orthonormal frame.
  normed_dxyzs_flat = dxyzs_flat / jnp.linalg.norm(
    dxyzs_flat, axis=-1, keepdims=True)
  dtxyzs_flat = jnp.pad(c * dt * normed_dxyzs_flat,
                        ((0, 0), (1, 0)), constant_values=dt)
  ode_states_flat = (
    # eye-point (txyz) coordinates, broadcastable across pixel-index,
    # padded to accomodate 4d tangent-coordinates.
    jnp.pad(txyz_end[jnp.newaxis, :], ((0, 0), (0, 4)))
    # per-pixel 4d tangent coordinates.
    + jnp.pad(jnp.einsum('cL,nL->nc', frame, dtxyzs_flat), ((0, 0), (4, 0))))
  return ode_states_flat.reshape(*dyzs.shape[:-1], 4 + 4)


#### Rendering

if 'DEMO-RENDER':
  CHECKER_SCALE = 8
  NUM_PATCHES_XY, PATCH_NUM_PIXELS_XY = 8, 72
  PIXELS_XY = PATCH_NUM_PIXELS_XY * NUM_PATCHES_XY
  z_displacement = 0.1  # How far to move off of the z=0 plane.
  color_ud, color_du = numpy.array([1, 1, 0, 1, 0.5, 0]).reshape(2, 1, 1, 3)  
  ode_steps = jnp.linspace(0, 100.0, 401, dtype=jnp.float64)
  jax_get_ray_eye_ends = jax.jit(_get_ray_eye_ends)
  r_disc_min, r_disc_max = 4, 8
  color_ud, color_du = numpy.array([1, 1, 0, 1, 0.5, 0]).reshape(2, 1, 1, 3)
  # Black Hole ("BH") Geometry:
  bh_metric_func = metric_func_from_L2_func(schwarzschild_L2_func)
  bh_dy_dt_func = geodesic_dy_dt_func_from_metric_func(bh_metric_func)
  def ray_trace(y0):
    return ode_solve(rk4_estimate, bh_dy_dt_func, y0, ode_steps)
  batch_ray_trace_flat = jax.vmap(ray_trace)
  @jax.jit
  def batch_ray_trace(y0s):
    return (batch_ray_trace_flat(y0s.reshape(-1, y0s.shape[-1]))
            .swapaxes(0, 1).reshape(-1, *y0s.shape))
  # Camera:
  eye_coords = 20.0 * jnp.array([0.0, 1.0, 0.0, z_displacement],
                                dtype=jnp.float64)
  camera_frame = get_orthonormal_frame(
    bh_metric_func, eye_coords,
    [(1,  0, 0, 0),       # t
     -eye_coords,         # forward (towards black hole)
     (0, 0, -1, 0),       # "y" - negative to make right-handed xyz.
     (0,  0, 0, 1)])      # "up"
  print('DDD Camera Frame\n', camera_frame.round(6).tolist())
  coord_steps = jnp.linspace(-0.75, 0.75, PIXELS_XY)
  # Target data:
  full_image = numpy.zeros((PIXELS_XY, PIXELS_XY, 3))
  image_blocks = full_image.reshape(
    *(NUM_PATCHES_XY, PATCH_NUM_PIXELS_XY)*2, 3).transpose(0, 2, 1, 3, 4)
  xy_coords = jnp.stack(
    jnp.meshgrid(coord_steps, coord_steps, indexing='xy'), axis=-1)
  xy_patches = xy_coords.reshape(
    # Splitting both i,j image-indices into "slow" block and fast
    # "pixel in block" indices - and pullung block-indices to the front:
    *(NUM_PATCHES_XY, PATCH_NUM_PIXELS_XY)*2, 2).transpose(0, 2, 1, 3, 4)
  for n_block_row, n_block_col in itertools.product(
      range(NUM_PATCHES_XY), repeat=2):
    xy_patch = xy_patches[n_block_row, n_block_col]
    ray_eye_ends = jax_get_ray_eye_ends(eye_coords, camera_frame, xy_patch)
    ray_traced_patch = numpy.array(batch_ray_trace(ray_eye_ends))
    # If the "far" end of the ray is not far behind the hole,
    # we color the current pixel white.
    ray_far_ends_x = ray_traced_patch[-1, ..., 1]
    ray_far_ends_dxyz = ray_traced_patch[-1, ..., -3:]
    y_cell_index, z_cell_index = (numpy.round(
      CHECKER_SCALE * ray_far_ends_dxyz[..., d] / ray_far_ends_dxyz[..., 0])
       for d in (1, 2))
    checkerboard_index = (y_cell_index + z_cell_index) % 2
    # Here, we expect to see three values: 0, 1, NaN.
    image_patch = numpy.where(
      # De-noising the interior uses some intermediate numpy trickery.
      numpy.isfinite(numpy.lib.stride_tricks.sliding_window_view(
        numpy.pad(ray_far_ends_x, ((1, 1), (1, 1))), (3, 3)
      )).all(axis=(-1, -2)),
      # Case: we did not hit the hole.
      numpy.where(
        (ray_far_ends_dxyz[..., 0] < 0) & (ray_far_ends_x < -10),              
        # We did pass the hole and see the checkerboard sky behind it.
        (0.25 + 0.5 * checkerboard_index),
        1.0),  # Case: We did not pass the hole - "white on the map".
      0.0)  # Case: We did hit the hole - "black on the map".
    image_blocks[n_block_row, n_block_col, ...] = image_patch[..., jnp.newaxis]
    # Adding an accretion disc:
    z_flat = ray_traced_patch[:, :, :, 3].reshape(ray_traced_patch.shape[0], -1)
    r_flat = numpy.linalg.norm(
      ray_traced_patch[:-1, :, :, 1:4], axis=-1).reshape(z_flat[:-1, :].shape)
    r_in_range = (r_disc_min < r_flat) & (r_flat < r_disc_max)
    z_transition = numpy.diff(numpy.sign(z_flat), axis=0)
    first_transition_index = numpy.argmax(
      r_in_range & z_transition.astype(bool), axis=0)
    z_transition_type = z_transition[
      first_transition_index,
      numpy.arange(first_transition_index.size)].reshape(*image_patch.shape, 1)
    image_blocks[n_block_row, n_block_col, :] = numpy.where(
      z_transition_type == 0,
      image_blocks[n_block_row, n_block_col, :],  # No change here.
      numpy.where(z_transition_type > 0, color_du, color_ud))
    print(f" - Rendered patch ({n_block_row}, {n_block_col})")
  matplotlib.pyplot.imshow(full_image)
  matplotlib.pyplot.show()
