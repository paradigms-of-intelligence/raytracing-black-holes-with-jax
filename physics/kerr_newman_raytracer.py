r"""Kerr-Newman black hole with a thin accretion disc.

The metric is the Kerr-Schild form g = eta + f k k with spin and charge
[1, 2]. Light rays are integrated with Dormand-Prince 8(5,3) [3] from
Hamilton's equations for H = g^{mu nu} p_mu p_nu / 2. All derivatives
are by jax. The disc radiates as a blackbody at the Novikov-Thorne
temperature [4, 5] with the redshift of each ray. Without arguments the
script renders the default scene. With --check, it runs the tests.

Units are G = c = 1 with the Schwarzschild radius 2 M as the unit of
length. Index 0 is time. A ray `state` is the `[8]-jax.Array` of the
position (t, x, y, z), followed by the covariant momentum p_mu.

Typical call:

  python3 -i kerr_newman_raytracer.py --charge=0.0 --width=256 --height=256 \
    --out=/tmp/bh.png

References:

  [1] M. Visser, "The Kerr spacetime: A brief introduction",
      arXiv:0706.0622 (2007). Eqs. (33) and (43)-(45) give the Kerr-Schild
      Cartesian coordinates.
  [2] G. C. Debney, R. P. Kerr, A. Schild, "Solutions of the Einstein
      and Einstein-Maxwell equations", J. Math. Phys. 10, 1842 (1969).
      The charged Kerr-Schild metric and its potential.
  [3] E. Hairer, S. P. Norsett, G. Wanner, "Solving Ordinary
      Differential Equations I: Nonstiff Problems", 2nd ed., Springer
      (1993), Chap. II, and the code dop853.f from
      www.unige.ch/~hairer/software.html.
  [4] I. D. Novikov, K. S. Thorne, "Astrophysics of black holes", in
      "Black Holes (Les Houches 1972)", Gordon and Breach (1973), p. 343.
  [5] D. N. Page, K. S. Thorne, "Disk-accretion onto a black hole.
      Time-averaged structure of accretion disk", ApJ 191, 499 (1974).
      Eq. (15n) gives the flux of the disc.
  [6] J. M. Bardeen, W. H. Press, S. A. Teukolsky, "Rotating black
      holes: Locally nonrotating frames, energy extraction, and scalar
      synchrotron radiation", ApJ 178, 347 (1972). Eq. (2.21) gives the
      ISCO.
  [7] C. T. Cunningham, "Returning radiation in accretion disks around
      black holes", ApJ 208, 534 (1976).
  [8] S. Chandrasekhar, "Radiative Transfer", Oxford University Press
      (1950). The Eddington limb darkening, used for discs in
      L.-X. Li, E. R. Zimmerman, R. Narayan, J. E. McClintock,
      ApJS 157, 335 (2005), Eq. (D20).
  [9] B. Carter, "Global structure of the Kerr family of gravitational
      fields", Phys. Rev. 174, 1559 (1968). The fourth constant of motion.
  [10] S. Chandrasekhar, "The Mathematical Theory of Black Holes",
      Oxford University Press (1983), Chap. 7. Spherical photon orbits.
  [11] C. Wyman, P.-P. Sloan, P. Shirley, "Simple analytic
      approximations to the CIE XYZ color matching functions",
      J. Computer Graphics Techniques 2(2), 1 (2013).
"""

import pdb  # For: interactive pdb.pm() debugging with `python3 -i ...`

import argparse
import functools
import sys
import time
import typing

import jax
from jax import numpy as jnp
import numpy
import PIL.Image

jax.config.update('jax_enable_x64', True)

ETA_DIAG = jnp.array([-1.0, 1.0, 1.0, 1.0], dtype=jnp.float64)
ETA = jnp.diag(ETA_DIAG)


# SI constants (used for disc temperature):

G_NEWTON = 6.6743e-11  # m**3 * kg**(-1) * s**(-2)
C_LIGHT = 2.99792458e8  # m / s
PROTON_MASS = 1.67262e-27  # kg
THOMSON_CROSS_SECTION = 6.65246e-29  # m**2
STEFAN_BOLTZMANN = 5.67037e-8  # J / K
SOLAR_MASS = 1.98892e30  # kg

#### Helpers

def unit_vec(v):
  """Maps a [..., n]-ArrayLike of vectors to length-normalized-to-1 Array."""
  return jnp.asarray(v) / jnp.linalg.norm(v, axis=-1, keepdims=True)


#### Geometry


class BlackHole(typing.NamedTuple):
  """Properties of the black hole at the coordinate origin.

  Attributes:
    mass: M. Half the Schwarzschild radius.
    spin: a = J / M.
    charge: Q in geometrized Gaussian units.
  """
  mass: float = 0.5
  spin: float = 0.0
  charge: float = 0.0

  @property
  def horizon(self):
    """The outer horizon radius M + sqrt(M**2 - a**2 - Q**2)."""
    m, a, q = self
    return m + jnp.sqrt(m * m - a * a - q * q)


def kerr_radius(xyz, spin, eps=1e-8):
  """Returns the Boyer-Lindquist radius of Kerr-Schild positions.

  The radius is the positive root of
  r**4 - (x**2 + y**2 + z**2 - a**2) r**2 - a**2 z**2 = 0 ([1], Eq. (33)).

  Args:
    xyz: `[..., 3]`-array of positions.
    spin: The spin a.
    eps: Floor for the radius. It is zero on the ring singularity.
  """
  half = 0.5 * (jnp.square(xyz).sum(axis=-1) - spin ** 2)
  # jnp.hypot loses precision here inside the jitted integrator (jax >= 0.7).
  root = jnp.sqrt(half * half + (spin * xyz[..., 2]) ** 2)
  return jnp.sqrt(jnp.maximum(half + root, eps * eps))


def kerr_schild(coord, bh):
  """Returns the scalar f and the null covector k of g = eta + f k k.

  The Kerr-Schild form of [1] (Eqs. (43)-(44)) with the charge term of
  [2], f = r**2 (2 M r - Q**2) / (r**4 + a**2 z**2). Both t and phi are
  reversed relative to [1]. The metric is then regular on the past
  horizon that the rays traced back from the camera cross and the sense
  of rotation is kept.

  Args:
    coord: `[4]`-array (t, x, y, z).
    bh: The `BlackHole`.
  """
  m, a, q = bh
  x, y, z = coord[1:4]
  r = kerr_radius(coord[1:], a)
  r2 = r * r
  f = r2 * (2.0 * m * r - q * q) / (r2 ** 2 + (a * z) ** 2)
  d = r2 + a * a
  k = jnp.array([1.0, -(r * x - a * y) / d, -(r * y + a * x) / d, -z / r])
  return f, k


def metric(coord, bh):
  """Returns g_{mu nu} at `coord`.

  Args:
    coord: `[4]`-array (t, x, y, z) for evaluating the metric.
    bh: `BlackHole` determining the geometry.
  """
  f, k = kerr_schild(coord, bh)
  return ETA + f * jnp.outer(k, k)


def inverse_metric(coord, bh):
  """Returns g^{mu nu} = eta - f k k at `coord` ([1], Eq. (45)).

  Args:
    coord: `[4]`-array (t, x, y, z) for evaluating the metric.
    bh: `BlackHole` determining the geometry.
  """
  f, k = kerr_schild(coord, bh)
  k_up = ETA @ k
  return ETA - f * jnp.outer(k_up, k_up)


#### Light rays


def hamiltonian_basic(coord, p_cov, bh):
  """Returns H = g^{mu nu} p_mu p_nu / 2."""
  return 0.5 * p_cov @ inverse_metric(coord, bh) @ p_cov


# This variant inlines the computation of the inverse metric,
# avoiding some matrix intermediates. This speeds up the main
# computation by about 10%.
def hamiltonian_fast(coord, p_cov, bh):
  """Returns H = g^{mu nu} p_mu p_nu / 2."""
  f, k = kerr_schild(coord, bh)
  k_up = k.at[0].set(-k[0])
  return 0.5 * ((jnp.square(p_cov) * ETA_DIAG).sum() -
                f * jnp.square((p_cov @ k_up)))


hamiltonian = hamiltonian_fast


_grad01_hamiltonian = jax.grad(hamiltonian, argnums=(0, 1))


def geodesic_rhs(state, bh):
  """Returns d(state)/d(affine parameter) for tracing back in time.

  Args:
    state: `[8]`-array (x_mu, p_mu).
    bh: The `BlackHole`.
  """
  # Hamilton's equations dx = dH/dp and dp = -dH/dx with reversed signs
  x, p = jnp.split(state, 2)
  dh_dx, dh_dp = _grad01_hamiltonian(x, p, bh)
  return jnp.concatenate([-dh_dp, dh_dx])


# Dormand and Prince 8(5,3) with the coefficients of Hairer's DOP853 [3].
# B is the 8th order solution and E5 and E3 are the two embedded error
# estimates.
DOP853_A = (
    (),
    (0.05260015195876773,),
    (0.0197250569845379, 0.0591751709536137),
    (0.02958758547680685, 0.0, 0.08876275643042054),
    (0.2413651341592667, 0.0, -0.8845494793282861, 0.924834003261792),
    (0.037037037037037035, 0.0, 0.0, 0.17082860872947386,
     0.12546768756682242),
    (0.037109375, 0.0, 0.0, 0.17025221101954405, 0.06021653898045596,
     -0.017578125),
    (0.03709200011850479, 0.0, 0.0, 0.17038392571223998,
     0.10726203044637328, -0.015319437748624402, 0.008273789163814023),
    (0.6241109587160757, 0.0, 0.0, -3.3608926294469414, -0.868219346841726,
     27.59209969944671, 20.154067550477894, -43.48988418106996),
    (0.47766253643826434, 0.0, 0.0, -2.4881146199716677, -0.590290826836843,
     21.230051448181193, 15.279233632882423, -33.28821096898486,
     -0.020331201708508627),
    (-0.9371424300859873, 0.0, 0.0, 5.186372428844064, 1.0914373489967295,
     -8.149787010746927, -18.52006565999696, 22.739487099350505,
     2.4936055526796523, -3.0467644718982196),
    (2.273310147516538, 0.0, 0.0, -10.53449546673725, -2.0008720582248625,
     -17.9589318631188, 27.94888452941996, -2.8589982771350235,
     -8.87285693353063, 12.360567175794303, 0.6433927460157636))
DOP853_B = (0.054293734116568765, 0.0, 0.0, 0.0, 0.0, 4.450312892752409,
            1.8915178993145003, -5.801203960010585, 0.3111643669578199,
            -0.1521609496625161, 0.20136540080403034, 0.04471061572777259)
DOP853_E5 = (0.01312004499419488, 0.0, 0.0, 0.0, 0.0, -1.2251564463762044,
             -0.4957589496572502, 1.6643771824549864, -0.35032884874997366,
             0.3341791187130175, 0.08192320648511571, -0.022355307863886294)
DOP853_E3 = (-0.18980075407240762, 0.0, 0.0, 0.0, 0.0, 4.450312892752409,
             1.8915178993145003, -5.801203960010585, -0.4226823213237919,
             -0.1521609496625161, 0.20136540080403034, 0.02265179219836082)


def weighted(row, ks):
  """Returns the sum of row[i] * ks[i] over the nonzero row entries."""
  return sum(a * k for a, k in zip(row, ks) if a)


def dop853_step(f, y, h, k1):
  """Returns one Dormand-Prince 8(5,3) step with the error estimates.

  Args:
    f: Right-hand side `[n, d]`-array to `[n, d]`-array.
    y: `[n, d]`-array of states.
    h: `[n, 1]`-array of step lengths or a scalar.
    k1: `f(y)`.

  Returns:
    `(y_next, err5, err3, k_next)` with the 8th order step, its
    differences to the embedded 5th and 3rd order steps and `f(y_next)`.
  """
  ks = [k1]
  for row in DOP853_A[1:]:
    ks.append(f(y + h * weighted(row, ks)))
  y_next = y + h * weighted(DOP853_B, ks)
  return (y_next, h * weighted(DOP853_E5, ks), h * weighted(DOP853_E3, ks),
          f(y_next))


def hermite(y0, y1, s0, s1, t):
  """Returns the cubic through y0 and y1 with end slopes s0 and s1.

  Args:
    y0: `[..., d]`-array of values at t = 0.
    y1: `[..., d]`-array of values at t = 1.
    s0: `[..., d]`-array of slopes at t = 0. That is `h * f(y0)` for a
      step h.
    s1: `[..., d]`-array of slopes at t = 1. That is `h * f(y1)`.
    t: Step fraction in [0, 1] of shape `[..., 1]` for one fraction per
      row of `y0`.
  """
  t0, t1 = 1.0 - t, t
  t0_sq, t1_sq = jnp.square(t0), jnp.square(t1)
  return ((1.0 + 2.0 * t1) * t0_sq * y0 + t * t0_sq * s0
          + t1_sq * (3.0 - 2.0 * t) * y1 - t1_sq * t0 * s1)


def plane_crossing(y0, y1, s0, s1):
  """Returns the state and the fraction of the step where z = 0.

  Args:
    y0: `[n, 8]`-array of states at the start of the step.
    y1: `[n, 8]`-array of states at the end.
    s0: `[n, 8]`-array `h * f(y0)`.
    s1: `[n, 8]`-array `h * f(y1)`.

  Returns:
    `(state, t_fraction)` with the `[n, 8]`-array of Hermite
    interpolants at the crossing and the `[n]`-array of step fractions.
    For a step without a crossing the result is undefined.
  """
  fn_z = lambda t: hermite(y0[..., 3], y1[..., 3], s0[..., 3], s1[..., 3], t)
  never_zero = lambda d: jnp.where(d == 0.0, 1.0, d)
  # We refine this iteratively via a few Newton steps.
  t_fraction = y0[:, 3] / never_zero(y0[:, 3] - y1[:, 3])
  for _ in range(3):
    z_t, dz_t = jax.jvp(fn_z, (t_fraction,), (jnp.ones_like(t_fraction),))
    t_fraction = jnp.clip(t_fraction - z_t / never_zero(dz_t), 0.0, 1.0)
  t_column = t_fraction[:, jnp.newaxis]
  return hermite(y0, y1, s0, s1, t_column), t_fraction


RUNNING, HIT_DISC, CAPTURED, ESCAPED = range(4)


class TraceResult(typing.NamedTuple):
  """Rays after `trace()`.

  Attributes:
    states: `[n, 8]`-array of final ray states.
    tags: `[n]`-int-array of `RUNNING`, `HIT_DISC`, `CAPTURED` or `ESCAPED`.
    radiance: `[n, ...]`-array of the radiance collected at the disc
      crossings.
    transmittance: `[n]`-array of the transmittance left for the sky.
  """
  states: jax.Array
  tags: jax.Array
  radiance: jax.Array
  transmittance: jax.Array


@functools.partial(jax.jit,
                   static_argnames=('emission', 'max_steps', 'reverse'))
def trace(states, bh, r_far, emission, *, args=(), tol=1e-8, max_steps=4000,
          reverse=False):
  """Traces rays back in time to the disc, the horizon or the sky.

  Args:
    states: `[n, 8]`-array of initial ray states.
    bh: The `BlackHole`.
    r_far: Boyer-Lindquist radius of the sky.
    emission: Callable `emission(state, bh, *args)` returning
      `(radiance, opacity)` at a crossing of the plane z = 0. The
      radiance is added to the ray with the transmittance so far and
      the transmittance is multiplied by `1 - opacity`.
    args: Tuple of further arguments of `emission`.
    tol: Absolute and relative tolerance of the step size control.
    max_steps: Maximum number of steps per call.
    reverse: If `True` the loop is a `jax.lax.scan` of `max_steps` steps
      with a checkpoint per step for reverse-mode derivatives.

  Returns:
    A `TraceResult`. A ray ends `HIT_DISC` once its transmittance is
    below 1e-3, `CAPTURED` inside the horizon or `ESCAPED` beyond
    `r_far`. After `max_steps` it is still `RUNNING`.
  """
  f = jax.vmap(functools.partial(geodesic_rhs, bh=bh))
  emit = jax.vmap(lambda state: emission(state, bh, *args))

  # Indices into `carry`, for readability:
  I_Y, I_H, I_K, I_TAG, I_RADIANCE, I_TRANSMITTANCE, I_N = range(7)

  def step(carry_now):
    y, h, k1, tag, radiance_now, transmittance_now, num_step = carry_now
    y_next, err5, err3, k_next = dop853_step(f, y, h[:, jnp.newaxis], k1)
    scale = tol * (1.0 + jnp.maximum(jnp.abs(y), jnp.abs(y_next)))
    n5 = jnp.square((err5 / scale)).sum(axis=-1)
    n3 = jnp.square((err3 / scale)).sum(axis=-1)
    # Hairer's combination of the two estimates. A NaN trial step counts
    # as infinite error.
    error_raw = n5 / jnp.sqrt(jnp.maximum(n5 + 0.01 * n3, 1e-300) * y.shape[-1])
    error = jnp.where(jnp.isfinite(error_raw), error_raw, jnp.inf)
    accept = (error <= 1.0) & (tag == RUNNING)
    # A crossing inside a step is redone with a step that ends there.
    # The cubic is accurate near its end points.
    y_hit, t = plane_crossing(y, y_next,
                              h[:, jnp.newaxis] * k1,
                              h[:, jnp.newaxis] * k_next)
    light, alpha = emit(y_hit)
    crossed = accept & (y[:, 3] * y_next[:, 3] < 0.0)
    aim = crossed & (alpha > 0.0) & (0.005 < t) & (t < 0.995)
    accept = accept & ~aim
    alpha = jnp.where(crossed & ~aim, alpha, 0.0)
    # The emission can be non-finite deep inside the horizon.
    added = (transmittance_now * alpha)[:, jnp.newaxis] * light
    radiance_next = radiance_now + jnp.where(
      (alpha > 0.0)[:, jnp.newaxis], added, 0.0)
    transmittance_next = transmittance_now * (1.0 - alpha)
    y = jnp.where(accept[:, jnp.newaxis], y_next, y)
    k_next = jnp.where(accept[:, jnp.newaxis], k_next, k1)
    factor = jnp.clip(0.9 * jnp.maximum(error, 1e-12) ** (-0.125), 0.2, 5.0)
    h = jnp.where(aim, t * h, h * factor)
    # Zero step for finished rays. Their trial steps are then finite.
    h = jnp.where(tag == RUNNING, h, 0.0)
    r = kerr_radius(y[:, 1:4], bh.spin)
    blocked = (tag == RUNNING) & (transmittance_next < 1e-3)
    tag1 = jnp.where(blocked, HIT_DISC, tag)
    tag2 = jnp.where((tag1 == RUNNING) & (r < bh.horizon), CAPTURED, tag1)
    tag_next = jnp.where((tag2 == RUNNING) & (r > r_far), ESCAPED, tag2)
    return (y, h, k_next, tag_next,
            radiance_next, transmittance_next, num_step + 1)

  n = states.shape[0]
  carry_now = (
    states, jnp.full((n,), 0.1), f(states), jnp.full((n,), RUNNING),
    jnp.zeros_like(emit(states)[0]), jnp.ones((n,)), 0)
  if reverse:
    # With a checkpoint per step, the backward pass stores one carry per step.
    carry_now, _ = jax.lax.scan(lambda c, _: (jax.checkpoint(step)(c), None),
                                carry_now, None, length=max_steps)
  else:
    carry_now = jax.lax.while_loop(
      (lambda c: (c[I_TAG] == RUNNING).any() & (c[I_N] < max_steps)),
      step, carry_now)
  y, _, _, tags, radiance, transmittance, _ = carry_now
  return TraceResult(y, tags, radiance, transmittance)


#### Camera


def orthonormal_frame(g, vectors):
  """Returns the Gram-Schmidt orthonormalization of `vectors` in the metric `g`.

  Args:
    g: `[4, 4]`-array of the metric.
    vectors: Four independent `[4]`-arrays with a timelike first one.

  Returns:
    `[4, 4]`-array with the vectors as columns. `frame.T @ g @ frame` is
    `diag(-1, 1, 1, 1)`.
  """
  frame = []
  for v in vectors:
    v_current = v
    for w in frame:
      v_current = v_current - w * (w @ g @ v_current) / (w @ g @ w)
    frame.append(v_current / jnp.sqrt(jnp.abs(v_current @ g @ v_current)))
  return jnp.stack(frame, axis=1)


def ray_states(coord, frame, g, directions):
  """Returns unit energy ray states at `coord` from `directions`.

  Args:
    coord: `[4]`-array of the observer position.
    frame: `[4, 4]`-array of `orthonormal_frame()` with the observer
      4-velocity as column 0.
    g: `[4, 4]`-array of the metric at `coord`.
    directions: `[n, 3]`-array of unit viewing directions in the frame.
  """
  local = jnp.concatenate([jnp.ones((directions.shape[0], 1)), -directions], 1)
  p_cov = local @ frame.T @ g
  return jnp.concatenate([jnp.broadcast_to(coord, p_cov.shape), p_cov], 1)


def camera_frame(bh, distance, inclination):
  """Returns `(position, metric, tetrad)` of a static camera.

  The camera is aimed at the origin with the spin axis up.

  Args:
    bh: The `BlackHole` setting the geometry.
    distance: Coordinate distance of the camera from the origin.
    inclination: Angle of the camera position from the spin axis.

  Returns:
    The `[4]`-array position in the x-z plane, the `[4, 4]`-array metric
    there and the `[4, 4]`-array tetrad with the columns 4-velocity,
    forward, right and up.

  Raises:
    ValueError: Inside the ergosphere.
  """
  si, ci = jnp.sin(inclination), jnp.cos(inclination)
  position = jnp.array([0.0, distance * si, 0.0, distance * ci])
  g = metric(position, bh)
  if not g[0, 0] < 0.0:
    raise ValueError('No static camera inside the ergosphere.')
  id4 = jnp.eye(4)
  frame = orthonormal_frame(g, [
      id4[0],
      (-position).at[0].set(0.0),
      id4[2],
      jnp.array([0.0, -ci, 0.0, si])])
  return position, g, frame


def camera_rays(bh, distance, inclination, fov, width, height,
                rows=slice(None)):
  """Returns one ray per pixel of the image rows `rows` as `[n, 8]` states.

  Args:
    bh: The `BlackHole`.
    distance: Coordinate distance of the camera from the origin.
    inclination: Angle of the camera position from the spin axis.
    fov: Vertical field of view.
    width: Image width in pixels.
    height: Image height in pixels.
    rows: Slice of the rows to produce rays for.

  Returns:
    `[n, 8]`-array of ray states in row-major order.
  """
  position, g, frame = camera_frame(bh, distance, inclination)
  half = jnp.tan(0.5 * fov)
  v = half * (height - 2.0 * jnp.arange(height)[rows] - 1.0) / height
  u = half * (2.0 * jnp.arange(width) + 1.0 - width) / height
  vv, uu = jnp.meshgrid(v, u, indexing='ij')
  # The row at v = 0 is moved off the disc plane for an edge-on view.
  vv = jnp.where(vv == 0.0, 1e-9 * half, vv)
  direction = jnp.stack([jnp.ones_like(uu), uu, vv], axis=-1).reshape(-1, 3)
  return ray_states(position, frame, g, unit_vec(direction))


#### Disc


def circular_orbit(r, bh):
  """Returns `(E, L_z, Omega)` of the prograde circular orbit at r.

  The orbit is a geodesic where the x derivative of the norm of
  xi = d_t + Omega d_phi is zero. That is a quadratic in Omega.

  Args:
    r: Boyer-Lindquist radius in the plane z = 0.
    bh: The `BlackHole`.

  Returns:
    Energy and angular momentum per unit rest mass and the angular
    velocity d phi / d t. Inside the photon orbit the energy is `inf`.
  """
  # jnp.hypot loses precision here inside the jitted integrator (jax >= 0.7).
  x = jnp.sqrt(r * r + bh.spin ** 2)
  point = jnp.array([0.0, x, 0.0, 0.0])
  e0 = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float64)

  def norm(x, omega):
    xi = jnp.array([1.0, 0.0, omega * x, 0.0])
    return xi @ metric(jnp.array([0.0, x, 0.0, 0.0]), bh) @ xi

  d_norm = jax.grad(norm)
  c0 = d_norm(x, 0.0)
  c1 = jax.grad(d_norm, argnums=1)(x, 0.0)
  c2 = jax.grad(jax.grad(d_norm, argnums=1), argnums=1)(x, 0.0)
  # Inside the photon orbit, there is no other orbit.
  # The values are floored for the backward pass.
  root = c1 * c1 - 2.0 * c0 * c2
  omega = (-c1 + jnp.sqrt(jnp.maximum(root, 0.0))) / c2
  xi = e0.at[2].set(omega * x)
  norm_u = norm(x, omega)
  u_cov = metric(point, bh) @ xi / jnp.sqrt(jnp.maximum(-norm_u, 1e-30))
  energy = jnp.where((root > 0.0) & (norm_u < 0.0), -u_cov[0], jnp.inf)
  return energy, x * u_cov[2], omega


@jax.jit
def isco(bh):
  """Returns the ISCO radius where E(r) is minimal.

  Args:
    bh: The `BlackHole`.
  """
  energy = lambda r: circular_orbit(r, bh)[0]
  grid = jnp.geomspace(bh.horizon, 20.0 * bh.mass, 1000)
  r = grid[jnp.argmin(jax.vmap(energy)(grid))]
  # Newton from the grid minimum. The step is skipped without
  # positive curvature of E (the horizon at a* = 1).
  fn_d_energy = jax.grad(energy)
  fn_d2_energy = jax.grad(fn_d_energy)
  for _ in range(4):
    curve = fn_d2_energy(r)
    r = jnp.where(curve > 0.0, r - fn_d_energy(r) / curve, r)
  return r


def get_disc_flux(bh, r_isco, r_out, n=1000):
  """Returns radii and the Page-Thorne flux at unit accretion rate.

  Eq. (15n) of [5] with zero torque at the inner edge. With E, L and
  Omega of `circular_orbit()` and ' for d/dr it is
  F = -Omega' / (4 pi r (E - Omega L)**2) * integral of (E - Omega L) L'
  from r_isco to r.

  Args:
    bh: The `BlackHole`.
    r_isco: Inner edge of the disc.
    r_out: Outer edge of the disc.
    n: Number of radii.

  Returns:
    `(radii, flux)` with `[n]`-arrays of Boyer-Lindquist radii from
    `r_isco` to `r_out` in geometric progression and of the flux from
    each face of the disc per unit proper area and time in units of
    Mdot / (2 M)**2.
  """
  radii = jnp.geomspace(r_isco, r_out, n)
  per_radius = functools.partial(jax.vmap, in_axes=(0, None))
  energy, ang_mom, omega = per_radius(circular_orbit)(radii, bh)
  _, d_ang_mom, d_omega = per_radius(jax.jacfwd(circular_orbit))(radii, bh)
  binding = energy - omega * ang_mom
  integrand = binding * d_ang_mom
  # The trapezoid rule averages the integrand at the ends of each interval
  # and takes the width from the geometric spacing, which is finest near
  # the ISCO. With n = 1000 the flux is accurate to 1e-4 of its peak.
  steps = 0.5 * jnp.diff(radii) * (integrand[1:] + integrand[:-1])
  integral = jnp.pad(jnp.cumsum(steps), ((1, 0),))
  return radii, -d_omega * integral / (4.0 * jnp.pi * radii * binding ** 2)


def temperature_scale(bh, r_isco, mass_in_m_solar, eddington_ratio):
  """Returns T**4 per unit of `get_disc_flux()` at the given luminosity.

  Args:
    bh: The `BlackHole`.
    r_isco: Inner edge of the disc.
    mass_in_m_solar: Mass of the black hole in solar masses.
    eddington_ratio: Disc luminosity over the Eddington luminosity
      4 pi G M m_p c / sigma_T.
  """
  efficiency = 1.0 - circular_orbit(r_isco, bh)[0]
  mass_in_kg = mass_in_m_solar * SOLAR_MASS
  eddington = (4.0 * numpy.pi * G_NEWTON * mass_in_kg * PROTON_MASS * C_LIGHT
               / THOMSON_CROSS_SECTION)
  r_s = 2.0 * G_NEWTON * mass_in_kg / C_LIGHT ** 2
  flux_si = eddington_ratio * eddington / (efficiency * r_s ** 2)
  return flux_si / STEFAN_BOLTZMANN


def disc_temperature(bh, r_out, mass_solar, eddington_ratio, eps=1e-30):
  """Returns `(disc_radii, temperature_in_Kelvin)`.

  Args:
    bh: The `BlackHole`.
    r_out: Outer edge of the disc.
    mass_solar: Mass of the black hole in solar masses.
    eddington_ratio: Disc luminosity over the Eddington luminosity.
    eps: Floor for the flux before the fourth root. It gives a finite
      derivative where the flux is zero.

  Returns:
    `[n]`-arrays of the radii from the ISCO to `r_out` and of the
    effective temperature there.
  """
  r_isco = isco(bh)
  radii, flux_disc = get_disc_flux(bh, r_isco, r_out)
  flux_total = flux_disc + returned_flux(bh, radii, flux_disc)
  flux_capped = jnp.maximum(flux_total, eps)
  scale = temperature_scale(bh, r_isco, mass_solar, eddington_ratio)
  return radii, (scale * flux_capped) ** 0.25


def disc_velocity(coord, bh, eps=1e-30):
  """Returns the 4-velocity of the disc at `coord` in the plane z = 0.

  Args:
    coord: `[4]`-array (t, x, y, 0).
    bh: The `BlackHole`.
    eps: Floor for -g(xi, xi) of xi = d_t + Omega d_phi for a finite
      result where there is no timelike orbit.
  """
  omega = circular_orbit(kerr_radius(coord[1:], bh.spin), bh)[2]
  xi = jnp.array([1.0, -omega * coord[2], omega * coord[1], 0.0])
  return xi / jnp.sqrt(jnp.maximum(-(xi @ metric(coord, bh) @ xi), eps))


# Opaque inside (1 - DISC_EDGE) r_out. The column density then falls
# exponentially to e^(-DISC_EXP_MAX) at r_out.
DISC_EDGE = 0.3
DISC_OPTICAL_DEPTH = 4.0
DISC_EXP_MAX = 9.0

def disc_crossing(state, bh, r_in, r_out, eps=1e-12):
  """Returns radius, redshift, angle and opacity of a disc crossing.

  Args:
    state: `[8]`-array of a ray state on the plane z = 0.
    bh: The `BlackHole`.
    r_in: Inner edge of the disc.
    r_out: Outer edge of the disc.
    eps: Floor for the divisors. It keeps the result finite at states
      off the disc.

  Returns:
    `(r, g, mu, opacity)`, where:
      - `r` is the Boyer-Lindquist radius of the crossing.
      - `g` is the redshift factor between the disc and the unit energy
        of the ray at the camera.
      - `mu` is the cosine of the ray to the disc normal in the fluid frame.
      - `opacity` is `1 - exp(-tau / mu)` of the slab and 0 outside
        `[r_in, r_out]`.
  """
  coord, p_cov = jnp.split(state, 2)
  r = kerr_radius(coord[1:], bh.spin)
  g = metric(coord, bh)
  u = disc_velocity(coord, bh)
  energy = jnp.maximum(-(p_cov @ u), eps)
  normal0 = jnp.array([0.0, 0.0, 0.0, 1.0])
  normal1 = normal0 + (normal0 @ g @ u) * u
  normal = normal1 / jnp.sqrt(jnp.maximum(normal1 @ g @ normal1, eps))
  mu = jnp.maximum(jnp.abs(p_cov @ normal) / energy, eps)
  span = DISC_EDGE * r_out
  column = jnp.exp(-DISC_EXP_MAX * jnp.maximum(r - r_out + span, 0.0) / span)
  alpha0 = -jnp.expm1(-DISC_OPTICAL_DEPTH * column / mu)
  alpha = jnp.where((r_in <= r) & (r <= r_out), alpha0, 0.0)
  return r, 1.0 / energy, mu, alpha


def limb(mu):
  """Returns the Eddington limb darkening of a grey atmosphere.

  Args:
    mu: Cosine of the angle to the surface normal.

  Returns:
    `(2 + 3 mu) / 4` normalized to the flux of an isotropic radiance of
    1 ([8] and Li et al. Eq. (D20)).
  """
  return 0.25 * (2.0 + 3.0 * mu)


def disc_emission(state, bh, radii, kelvin):
  """Returns the limb darkened blackbody radiance at g T and the opacity.

  Args:
    state: `[8]`-array of a ray state on the plane z = 0.
    bh: The `BlackHole`.
    radii: `[n]`-array of disc radii.
    kelvin: `[n]`-array of disc temperatures at `radii`.

  Returns:
    `(rgb, opacity)` with the `[3]`-array of linear sRGB radiance and
    the opacity of `disc_crossing()`.
  """
  r, g_factor, mu, alpha = disc_crossing(state, bh, radii[0], radii[-1])
  rgb = blackbody_rgb(g_factor * jnp.interp(r, radii, kelvin)) * limb(mu)
  return rgb, alpha


def disc_landing(state, bh, radii, rings):
  """Returns g**4 times limb darkening times the ring weights at a crossing.

  The `emission` of the rays of `returned_flux()`.

  Args:
    state: `[8]`-array of a ray state on the plane z = 0.
    bh: The `BlackHole`.
    radii: `[n]`-array of disc radii of which only the edges are used.
    rings: `[RETURN_RINGS]`-array of ring radii in geometric progression.

  Returns:
    `(weights, opacity)` with the `[RETURN_RINGS]`-array of the radiance
    received per unit flux of each ring and the opacity of
    `disc_crossing()`.
  """
  r, g_factor, mu, alpha = disc_crossing(state, bh, radii[0], radii[-1])
  # Linear weights in log r between the two nearest rings.
  s = jnp.log(r / rings[0]) / jnp.log(rings[1] / rings[0])
  weight = jnp.maximum(1.0 - jnp.abs(jnp.arange(rings.size) - s), 0.0)
  return g_factor ** 4 * limb(mu) * weight, alpha


# Rings and directions of the returning radiation map. With 48 azimuths
# the returned fraction at a* = 0.9 is 9%. With 192, it is 0.1%.
RETURN_RINGS = 32
RETURN_MU = 16
RETURN_PHI = 192

LG_NODE, LG_WEIGHT = numpy.polynomial.legendre.leggauss(RETURN_MU)


def returned_flux(bh, radii, flux):
  """Returns the flux at `radii` from the disc onto itself [7].

  The transfer matrix between rings is the hemisphere quadrature of g**4
  times the limb darkening over the rays traced back from each ring to
  the disc. The total flux is the self-consistent solution of F = F_0 + T F.

  Args:
    bh: The `BlackHole`.
    radii: `[n]`-array of disc radii.
    flux: `[n]`-array of the viscous flux F_0 at `radii`.

  Returns:
    `[n]`-array of F - F_0 at `radii` in the units of `flux`.
  """
  # The first ring is one spacing above the ISCO so that the landing
  # weight is continuous.
  rings = jnp.geomspace(radii[0], radii[-1], RETURN_RINGS + 1)[1:]
  phi = (numpy.arange(RETURN_PHI) + 0.5) * 2.0 * numpy.pi / RETURN_PHI
  mu, phi = [v.ravel() for v in numpy.meshgrid(0.5 * (LG_NODE + 1.0), phi,
                                               indexing='ij')]
  sine = numpy.sqrt(1.0 - jnp.square(mu))
  directions = jnp.stack([mu, sine * numpy.cos(phi), sine * numpy.sin(phi)],
                         axis=-1)
  # Quadrature weights mu dOmega / pi over the hemisphere.
  weights = jnp.asarray(numpy.repeat(LG_WEIGHT, RETURN_PHI) * mu / RETURN_PHI)

  def launch(ring):
    x_ring = jnp.sqrt(ring * ring + bh.spin ** 2)
    coord = jnp.array([0.0, x_ring, 0.0, 0.0])
    g = metric(coord, bh)
    i4 = jnp.eye(4)
    frame = orthonormal_frame(
      g, [disc_velocity(coord, bh), i4[3], i4[1], i4[2]])
    return ray_states(coord, frame, g, directions)

  states = jax.vmap(launch)(rings).reshape(-1, 8)
  landed = trace(states, bh, 8.0 * radii[-1], disc_landing,
                 args=(radii, rings), tol=1e-6, max_steps=1000).radiance
  transfer = jnp.einsum('k,ikj->ij', weights,
                        landed.reshape(RETURN_RINGS, -1, RETURN_RINGS))
  seed = jnp.interp(rings, radii, flux)
  total = jnp.linalg.solve(jnp.eye(RETURN_RINGS) - transfer, seed)
  return jnp.interp(radii, rings, total - seed)


#### Color

# Gaussian lobe fit of [11] to the CIE 1931 functions x y z.
# A lobe is weight, then centre in nm and then the widths below and above it.
CIE_LOBES = (
    ((1.056, 599.8, 37.9, 31.0), (0.362, 442.0, 16.0, 26.7),
     (-0.065, 501.1, 20.4, 26.2)),
    ((0.821, 568.8, 46.9, 40.5), (0.286, 530.9, 16.3, 31.1)),
    ((1.217, 437.0, 11.8, 36.0), (0.681, 459.0, 26.0, 13.8)))
XYZ_FROM_SRGB = jnp.array([[+3.2406, -1.5372, -0.4986],
                           [-0.9689, +1.8758, +0.0415],
                           [+0.0557, -0.2040, +1.0570]]).T
WAVELENGTHS_NM = jnp.arange(380.0, 781.0, 5.0)


def cie_bar(lobes):
  """Returns one colour matching function on WAVELENGTHS_NM.

  Args:
    lobes: One entry of `CIE_LOBES`.
  """
  total = 0.0
  for weight, centre, below, above in lobes:
    width = jnp.where(WAVELENGTHS_NM < centre, below, above)
    total += weight * jnp.exp(-0.5 * ((WAVELENGTHS_NM - centre) / width) ** 2)
  return total


CIE_BARS = jnp.stack([cie_bar(lobes) for lobes in CIE_LOBES], axis=-1)


def blackbody_rgb(kelvin):
  """Returns the linear sRGB radiance of a blackbody.

  Args:
    kelvin: `[...]`-array of temperatures.

  Returns:
    `[..., 3]`-array in a fixed arbitrary unit.
  """
  # Planck's law in the wavelength with hc/k = 1.4388e7 nm K.
  x = 1.4388e7 / (WAVELENGTHS_NM * jnp.asarray(kelvin)[..., jnp.newaxis])
  planck = (500.0 / WAVELENGTHS_NM) ** 5 / jnp.expm1(jnp.minimum(x, 700.0))
  return jnp.maximum(planck @ CIE_BARS @ XYZ_FROM_SRGB, 0.0)


#### Sky


# Milky Way surface brightness in units of the median star per map cell.
GLOW = 0.5

SKY_DTYPE = jnp.float32


@functools.partial(jax.jit, static_argnames=('rows', 'count'))
def star_sky(rows, count=100_000):
  """Returns an equirectangular star map of `rows` by 2 `rows` cells.

  Half of the stars are uniform on the sphere. Half are in a band of
  10 degrees around the equator with a bulge at the centre of the map.
  A smooth glow of the same shape with a dust lane is the unresolved
  stars. Fluxes are drawn from the Euclidean count law N(>F) proportional
  to F**(-3/2) and temperatures are log-normal around 5600 K. The unit is
  the median star in one cell of a 2048 row map.

  Args:
    rows: Number of rows of the map.
    count: Number of stars.

  Returns:
    `[rows, 2 rows, 3]`-array of `SKY_DTYPE`. Row 0 is the pole +z and
    the middle column the direction -x behind the black hole.
  """
  shape = (rows, 2 * rows)
  k = jax.random.split(jax.random.key(0), 7)
  n = count // 2
  even = jnp.arcsin(jax.random.uniform(k[0], (n,), minval=-1.0, maxval=1.0))
  band = jnp.deg2rad(10.0) * jax.random.laplace(k[1], (n,))
  lat = jnp.clip(jnp.concatenate([even, band]), -0.5 * jnp.pi, 0.5 * jnp.pi)
  lon = jax.random.uniform(k[2], (count,), minval=-jnp.pi, maxval=jnp.pi)
  bulge = jnp.pi + 0.5 * jax.random.normal(k[3], (count,))
  in_bulge = jax.random.uniform(k[4], (count,)) < 0.3
  lon = jnp.where(in_bulge & (jnp.arange(count) >= n), bulge, lon)
  flux = jax.random.pareto(k[5], 1.5, (count,))
  kelvin = 10.0 ** (3.75 + 0.12 * jax.random.normal(k[6], (count,)))
  rgb_bb = blackbody_rgb(kelvin)
  rgb_scaled = rgb_bb * (flux / jnp.max(rgb_bb, axis=-1))[:, jnp.newaxis]
  # A star of fixed flux has a radiance proportional to the cell count.
  rgb = rgb_scaled / jnp.median(flux) * (rows / 2048.0) ** 2
  row = jnp.clip(((0.5 - lat / jnp.pi) * rows).astype(int), 0, rows - 1)
  col = (-lon / (2.0 * jnp.pi) % 1.0 * shape[1]).astype(int)
  sky = (jnp.zeros((*shape, 3), SKY_DTYPE)
         .at[row, col].add(rgb.astype(SKY_DTYPE)))
  lat = (0.5 - (jnp.arange(rows, dtype=SKY_DTYPE) + 0.5) / rows) * jnp.pi
  lon = 2.0 * jnp.pi * (
    (jnp.arange(shape[1], dtype=SKY_DTYPE) + 0.5) / shape[1])
  glow = (jnp.exp(-jnp.abs(lat) / jnp.deg2rad(5.0))[:, jnp.newaxis]
          * (1.0 + 2.0 * jnp.exp(-2.0 * (lon[None, :] - jnp.pi) ** 2))
          * (1.0
             - 0.6 * jnp.exp(-(lat / jnp.deg2rad(1.5)) ** 2))[:, jnp.newaxis])
  color_bb = blackbody_rgb(6000.0)
  color = (GLOW * color_bb / jnp.max(color_bb)).astype(SKY_DTYPE)
  return sky + glow[:, :, jnp.newaxis] * color


def image_sky(filename):
  """Returns content of equirectangular sRGB image as linear radiance.

  Args:
    filename: Path of the image.

  Returns:
    `[rows, cols, 3]`-array of `SKY_DTYPE` in the layout of `star_sky()`.
  """
  image = PIL.Image.open(filename).convert('RGB')
  srgb = numpy.asarray(image, numpy.float32) / 255.0
  return jnp.array(
    numpy.where(srgb <= 0.04045, srgb / 12.92,
                ((srgb + 0.055) / 1.055) ** 2.4), dtype=SKY_DTYPE)


def sky_radiance(sky, direction):
  """Returns the bilinear lookup of `sky` in the given unit directions.

  Args:
    sky: `[rows, cols, 3]`-array in the layout of `star_sky()`.
    direction: `[n, 3]`-array of unit vectors.

  Returns:
    `[n, 3]`-array of float64.
  """
  rows, cols = sky.shape[:2]
  # The map is read from inside the sphere. The middle column is the -x
  # direction behind the black hole. The azimuth increases to the left.
  azimuth = jnp.arctan2(direction[:, 1], direction[:, 0])
  x = -azimuth / (2.0 * jnp.pi) * cols - 0.5
  y = jnp.arccos(jnp.clip(direction[:, 2], -1.0, 1.0)) / jnp.pi * rows - 0.5
  x0, y0 = jnp.floor(x), jnp.floor(y)
  fx, fy = (x - x0)[:, jnp.newaxis], (y - y0)[:, jnp.newaxis]
  x0i, y0i = x0.astype(int), y0.astype(int)
  rgb = 0.0
  for dx, wx in ((0, 1.0 - fx), (1, fx)):
    for dy, wy in ((0, 1.0 - fy), (1, fy)):
      rgb += wx * wy * sky[jnp.clip(y0i + dy, 0, rows - 1), (x0i + dx) % cols]
  return rgb


#### Rendering


@jax.jit
def sky_behind(states, tags, bh, sky):
  """Returns the sky radiance for escaped rays and zero for the rest.

  Args:
    states: `[n, 8]`-array of final ray states.
    tags: `[n]`-array of their tags.
    bh: The `BlackHole`.
    sky: `[rows, cols, 3]`-array of the sky map.

  Returns:
    `[n, 3]`-array of linear sRGB radiance.
  """
  coord, p_cov = jnp.split(states, 2, axis=-1)
  # The sky direction is opposite to the momentum.
  g_up = jax.vmap(functools.partial(inverse_metric, bh=bh))(coord)
  spatial = jnp.einsum('nij,nj->ni', g_up, p_cov)[:, 1:]
  direction = -unit_vec(spatial)
  # The sky blueshift is the same for every ray and is in the sky gain.
  escaped = (tags == ESCAPED)[:, jnp.newaxis]
  return jnp.where(escaped, sky_radiance(sky, direction), 0.0)


def render(bh, states, radii, kelvin, sky, width, height,
           chunk_default=1<<16, file=None):
  """Returns the `[height, width, 3]` image of the rays.

  Args:
    bh: The `BlackHole`.
    states: `[height * width, 8]`-array of ray states in image order.
    radii: `[n]`-array of disc radii.
    kelvin: `[n]`-array of disc temperatures.
    sky: `[rows, cols, 3]`-array of the sky map scaled to the scene.
    width: Image width.
    height: Image height.
    chunk_default: Number of rays traced at once.
    file: `file=` parameter of `print()` for the step limit report.

  Returns:
    `[height, width, 3]`-numpy-array of linear sRGB radiance.
  """
  # Sorted by impact parameter so that the slow rays share batches.
  axis = numpy.cross(states[:, 1:4], states[:, 5:8])
  order = numpy.argsort(numpy.linalg.norm(axis, axis=-1))
  # Padded with the outermost ray so that every batch has the same shape.
  chunk = min(chunk_default, states.shape[0])
  pad = -states.shape[0] % chunk
  filler = jnp.repeat(states[order[-1:]], pad, axis=0)
  padded = jnp.concatenate([states[order], filler])
  r_far = 8.0 * kerr_radius(states[0, 1:4], bh.spin).item()
  rgb, tags = [], []
  for batch in jnp.split(padded, padded.shape[0] // chunk):
    traced = trace(batch, bh, r_far, disc_emission, args=(radii, kelvin))
    tags.append(traced.tags)
    sky_light = sky_behind(traced.states, traced.tags, bh, sky)
    rgb.append(traced.radiance
               + traced.transmittance[:, jnp.newaxis] * sky_light)
  unfinished = int((jnp.concatenate(tags)[:states.shape[0]] == RUNNING).sum())
  if unfinished:
    print(f'rays that reached the step limit: {unfinished} ', file=file)
  image = numpy.empty((height * width, 3))
  image[order] = numpy.concatenate(rgb)[:height * width]
  return image.reshape(height, width, 3)


def save_png(path, image):
  """Writes linear RGB as 8-bit sRGB with the brightest 0.1% clipped.

  Args:
    path: Output path.
    image: `[height, width, 3]`-array of linear sRGB radiance.

  Raises:
    SystemExit: If no pixel is lit.
  """
  lit = image[image > 0.0]
  if lit.size == 0:
    sys.exit('No ray reached the disc or the sky.')
  clipped = numpy.clip(image / numpy.quantile(lit, 0.999), 0.0, 1.0)
  srgb = numpy.where(clipped <= 0.0031308,
                     12.92 * clipped,
                     1.055 * clipped ** (1 / 2.4) - 0.055)
  PIL.Image.fromarray((srgb * 255.0 + 0.5).astype(numpy.uint8)).save(path)


#### Checks


def curvature(coord, bh):
  """Returns `(G_{mu nu}, Kretschmann_scalar)` at `coord`.

  Args:
    coord: `[4]`-array (t, x, y, z).
    bh: The `BlackHole`.
  """
  def christoffel(x):
    d = jax.jacfwd(metric)(x, bh)
    return 0.5 * jnp.einsum(
        'im,mkl->ikl', inverse_metric(x, bh),
        d + jnp.einsum('mlk->mkl', d) - jnp.einsum('klm->mkl', d))

  gamma = christoffel(coord)
  d_gamma = jax.jacfwd(christoffel)(coord)
  aux = (jnp.einsum('ijlk->ijkl', d_gamma)
         + jnp.einsum('ikm,mjl->ijkl', gamma, gamma))
  riemann = aux - aux.swapaxes(-1, -2)
  ricci = jnp.einsum('mimj->ij', riemann)
  g, g_inv = metric(coord, bh), inverse_metric(coord, bh)
  einstein = ricci - 0.5 * g * jnp.einsum('ij,ij->', g_inv, ricci)
  return (einstein,
          jnp.einsum('Ijkl,iJKL,Ii,jJ,kK,lL->',
                     riemann, riemann, g, g_inv, g_inv, g_inv,
                     optimize='greedy'))


def maxwell_source(coord, bh):
  """Returns 8 pi T_{mu nu} of the electromagnetic field of the black hole.

  The potential is A = Q r**3 / (r**4 + a**2 z**2) k with the k of the
  metric [2].

  Args:
    coord: `[4]`-array (t, x, y, z).
    bh: The `BlackHole`.
  """
  def potential(x):
    r = kerr_radius(x[1:], bh.spin)
    scale = bh.charge * r ** 3 / (r ** 4 + (bh.spin * x[3]) ** 2)
    return scale * kerr_schild(x, bh)[1]

  d = jax.jacfwd(potential)(coord)
  field = d.T - d
  g, g_inv = metric(coord, bh), inverse_metric(coord, bh)
  invariant = (field * (g_inv @ field @ g_inv)).sum()
  return 2.0 * (field @ g_inv @ field.T - 0.25 * g * invariant)


def carter(state, bh):
  """Returns Carter's constant of a light ray [9].

  Args:
    state: `[8]`-array of the ray state.
    bh: The `BlackHole`.

  Returns:
    Q = p_theta**2 + cos(theta)**2 (L_z**2 / sin(theta)**2 - a**2 E**2)
    in Boyer-Lindquist angles.
  """
  (_, x, y, z), (_, p_x, p_y, p_z) = jnp.split(state, 2)
  r = kerr_radius(state[1:4], bh.spin)
  sine = jnp.sqrt((x * x + y * y) / (r * r + bh.spin ** 2))
  p_theta = z / (r * sine) * (x * p_x + y * p_y) - r * sine * p_z
  energy, ang_mom = -state[4], x * p_y - y * p_x
  return p_theta ** 2 + (z / r) ** 2 * (
      (ang_mom / sine) ** 2 - (bh.spin * energy) ** 2)


def check(file=None):
  """Prints the test results.

  Args:
    file: `file=` parameter to forward to `print()`, for redirecting output.
  """
  def fprint(*args):
    return print(*args, file=file)

  points = (jnp.array([0.0, 4.3, 2.1, 1.7]),
            jnp.array([0.0, -3.0, 5.5, -2.2]))

  def residual(bh, source_bh):
    """Returns max |G - 8 pi T| and max sqrt(K) over the test points."""
    worst = scale = 0.0
    for x in points:
      einstein, kretschmann = curvature(x, bh)
      off = jnp.abs(einstein - maxwell_source(x, source_bh)).max()
      worst = max(worst, off.item())
      scale = max(scale, jnp.sqrt(kretschmann).item())
    return worst, scale

  fprint('=== Einstein-Maxwell residual and the curvature scale sqrt(K) ===')
  for a_star, q_star in ((0.0, 0.0), (0.9, 0.0), (0.6, 0.5), (0.3, 0.9)):
    bh = BlackHole(spin=0.5 * a_star, charge=0.5 * q_star)
    worst, scale = residual(bh, bh)
    fprint(f'  a*={a_star} Q*={q_star}  {worst:.1e} of {scale:.1e}')
  worst, scale = residual(bh, BlackHole(spin=bh.spin, charge=0.9 * bh.charge))
  fprint(f'  control with 0.9 Q  {worst:.1e} of {scale:.1e}')

  def closed_isco(a_star):
    """Returns the Kerr ISCO radius in units of M ([6], Eq. (2.21))."""
    z1 = 1.0 + (1.0 - a_star ** 2) ** (1 / 3) * (
        (1.0 + a_star) ** (1 / 3) + (1.0 - a_star) ** (1 / 3))
    z2 = jnp.sqrt(3.0 * a_star ** 2 + z1 * z1)
    return 3.0 + z2 - jnp.sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2))

  fprint('=== ISCO radius and spin derivative against Bardeen et al. 1972 ===')
  solver = lambda a_star: isco(BlackHole(spin=0.5 * a_star))
  worst = max(abs(float(solver(a_star) - 0.5 * closed_isco(a_star)))
              for a_star in (0.0, 0.6, 0.9, 0.998))
  slope = jax.jvp(solver, (0.6,), (1.0,))[1]
  gap = abs(float(slope - 0.5 * jax.grad(closed_isco)(0.6)))
  fprint(f'  max radius difference {worst:.1e} for a* up to 0.998')
  fprint(f'  derivative difference {gap:.1e} at a* = 0.6')

  fprint('=== Disc luminosity against 1 - E_isco ===')
  for a_star in (0.0, 0.9):
    bh = BlackHole(spin=0.5 * a_star)
    radii, flux = get_disc_flux(bh, isco(bh), 1e5, n=40000)
    energy = jax.vmap(circular_orbit, in_axes=(0, None))(radii, bh)[0]
    shine = jnp.trapezoid(4.0 * jnp.pi * radii * energy * flux, radii)
    want = 1.0 - float(energy[0])
    fprint(f'  a*={a_star}  {float(shine):.6f} against {want:.6f}')

  fprint('=== Returning radiation as a fraction of the viscous flux ===')
  for a_star in (0.0, 0.9):
    bh = BlackHole(spin=0.5 * a_star)
    radii, flux = get_disc_flux(bh, isco(bh), 15.0)
    back = returned_flux(bh, radii, flux)
    fraction = (jnp.trapezoid(radii * back, radii) /
                jnp.trapezoid(radii * flux, radii))
    fprint(f'  a*={a_star}  {float(fraction):.4f}')

  fprint('=== Fixed-step error of DOP853 and of RK4 at equal cost ===')
  bh = BlackHole(spin=0.45, charge=0.15)
  states = camera_rays(bh, 30.0, numpy.deg2rad(80.0), numpy.deg2rad(25.0),
                       5, 5)
  f = jax.vmap(functools.partial(geodesic_rhs, bh=bh))

  def dop853(f, y, h, k1):
    y, _, _, k1 = dop853_step(f, y, h, k1)
    return y, k1

  def rk4(f, y, h, k1):
    """Returns `(y, f(y))` after a RK4 step of `h`."""
    k2 = f(y + 0.5 * h * k1)
    k3 = f(y + 0.5 * h * k2)
    k4 = f(y + h * k3)
    y = y + h / 6.0 * (k1 + 2.0 * (k2 + k3) + k4)
    return y, f(y)

  @functools.partial(jax.jit, static_argnames=('count', 'scheme'))
  def march(count, scheme):
    h = jnp.full((states.shape[0], 1), 26.0 / count)
    body = lambda _, y_k1: scheme(f, y_k1[0], h, y_k1[1])
    return jax.lax.fori_loop(0, count, body, (states, f(states)))[0]

  truth, previous = march(4096, dop853), None
  for count in (6, 12, 24, 48):
    error = float(jnp.abs(march(count, dop853) - truth).max())
    same_work = float(jnp.abs(march(3 * count, rk4) - truth).max())
    rate = '' if previous is None else (
        f'  order {numpy.log2(previous / error):.2f}')
    fprint(f'  {count:4d} steps  {error:.1e}  RK4 {same_work:.1e}{rate}')
    previous = error

  fprint('=== Interpolation error inside a step ===')
  h = jnp.full((truth.shape[0], 1), 0.5)
  k0 = f(truth)
  y1, _, _, k1 = dop853_step(f, truth, h, k0)
  worst = 0.0
  for fraction in (0.3, 0.7):
    direct = dop853_step(f, truth, fraction * h, k0)[0]
    t = jnp.full(h.shape[:1], fraction)
    between = hermite(truth, y1, h * k0, h * k1, t[:, jnp.newaxis])
    worst = max(worst, jnp.abs(between - direct).max().item())
  moved = float(jnp.abs(y1 - truth).max())
  fprint(f'  {worst:.1e} over a step of length {moved:.1e}')

  fprint('=== Drift of the conserved quantities over a traced ray ===')
  states = camera_rays(bh, 30.0, numpy.deg2rad(80.0), numpy.deg2rad(30.0),
                       15, 15)
  no_disc = jnp.zeros(2)
  traced = trace(states, bh, 240.0, disc_emission, args=(no_disc, no_disc))
  final, escaped = traced.states, traced.tags == ESCAPED
  norm = jax.vmap(functools.partial(hamiltonian, bh=bh))(
    *jnp.split(final, 2, axis=1))
  ang_mom = lambda y: y[:, 1] * y[:, 6] - y[:, 2] * y[:, 5]
  invariants = jax.vmap(functools.partial(carter, bh=bh))
  drifts = [jnp.abs(norm), jnp.abs(ang_mom(final) - ang_mom(states)),
            jnp.abs(invariants(final) - invariants(states))]
  null, l_z, q = (d[escaped].max().item() for d in drifts)
  fprint(f'  max |g(p, p)| {null:.1e}  max |dL_z| {l_z:.1e}  max |dQ| {q:.1e} '
         f'over {int(escaped.sum())} escaped rays with max Q '
         f'{float(invariants(states).max()):.1f}')

  fprint('=== Image derivative in spin by jvp against a finite difference ===')

  def disc_light(a_star):
    hole = BlackHole(spin=0.5 * a_star, charge=bh.charge)
    radii, kelvin = disc_temperature(hole, 15.0, 1e9, 1e-5)
    return trace(states, hole, 240.0, disc_emission,
                 args=(radii, kelvin)).radiance

  slope = jax.jvp(disc_light, (0.9,), (1.0,))[1]
  finite = (disc_light(0.9 + 1e-5) - disc_light(0.9 - 1e-5)) / 2e-5
  gap = jnp.abs(slope - finite) / jnp.abs(slope).max()
  fprint(f'  median {jnp.median(gap).item():.1e} and max '
         f'{gap.max().item():.1e} relative to the largest jvp entry over '
         f'{states.shape[0]} rays')

  fprint('=== Spin fitted to a disc image by Gauss-Newton ===')
  target = disc_light(0.6)
  a_star = 0.4
  for _ in range(6):
    light, slope = jax.jvp(disc_light, (a_star,), (1.0,))
    a_star -= float(jnp.vdot(slope, light - target) / jnp.vdot(slope, slope))
  fprint(f'  start 0.4  fit {a_star:.6f}  truth 0.6')

  fprint('=== Transpose test of the vjp against the jvp ===')
  radii, kelvin = disc_temperature(bh, 15.0, 1e9, 1e-5)
  loop = lambda k: trace(states, bh, 240.0, disc_emission,
                         args=(radii, k)).radiance
  scan = lambda k: trace(states, bh, 240.0, disc_emission, args=(radii, k),
                         max_steps=600, reverse=True).radiance
  tangent = kelvin * jax.random.normal(jax.random.key(1), kelvin.shape)
  cotangent = jax.random.normal(jax.random.key(2), (states.shape[0], 3))
  forward = jnp.vdot(cotangent, jax.jvp(loop, (kelvin,), (tangent,))[1])
  backward = jnp.vdot(jax.vjp(scan, kelvin)[1](cotangent)[0], tangent)
  fprint(f'  <c, J t> = {float(forward):.12e} against <J^T c, t> = '
         f'{backward.item():.12e}')

  fprint('=== Data types ===')
  sky = star_sky(64, count=1_000)
  traced = trace(states, bh, 240.0, disc_emission, args=(radii, kelvin))
  arrays = (('rays', states), ('disc', kelvin), ('traced', traced.states),
            ('radiance', traced.radiance), ('sky map', sky),
            ('sky', sky_behind(traced.states, traced.tags, bh, sky)))
  fprint('  ' + '  '.join(f'{name} {array.dtype}' for name, array in arrays))

  fprint('=== Shadow edge against the critical curve of the photon shell ===')
  # Bisection between captured and escaped rays in 8 directions. The
  # edge ray has the xi = L_z / E and eta = Q / E**2 of a spherical photon
  # orbit ([10], Chap. 7). xi is monotonic in r so r follows from xi and
  # eta is the test.
  position, g, frame = camera_frame(bh, 30.0, numpy.deg2rad(80.0))
  angles = jnp.arange(8) * (0.25 * jnp.pi) + 0.3

  def edge_rays(s):
    direction = jnp.stack([jnp.ones(8), s * jnp.cos(angles),
                           s * jnp.sin(angles)], axis=-1)
    return ray_states(position, frame, g, unit_vec(direction))

  lo, hi = jnp.zeros(8), jnp.full(8, 0.25)
  for _ in range(36):
    mid = 0.5 * (lo + hi)
    tags = trace(edge_rays(mid), bh, 240.0, disc_emission,
                 args=(no_disc, no_disc)).tags
    lo = jnp.where(tags == CAPTURED, mid, lo)
    hi = jnp.where(tags == CAPTURED, hi, mid)
  edge = edge_rays(0.5 * (lo + hi))
  energy = -edge[:, 4]
  xi_ray, eta_ray = ang_mom(edge) / energy, invariants(edge) / energy ** 2
  delta = lambda r: r * r - 2.0 * bh.mass * r + bh.spin ** 2 + bh.charge ** 2
  shell = lambda r: 4.0 * r * delta(r) / jax.grad(delta)(r)
  xi = lambda r: (r * r + bh.spin ** 2 - shell(r)) / bh.spin
  eta = lambda r: shell(r) ** 2 / delta(r) - (xi(r) - bh.spin) ** 2
  grid = jnp.linspace(bh.horizon, 4.5 * bh.mass, 2000)
  worst = 0.0
  for xi_k, eta_k in zip(xi_ray, eta_ray):
    r = grid[jnp.argmin(jnp.abs(jax.vmap(xi)(grid) - xi_k))]
    for _ in range(6):
      r = r - (xi(r) - xi_k) / jax.grad(xi)(r)
    worst = max(worst, abs(float(eta(r) - eta_k)))
  fprint(f'  max |eta - eta(r)| {worst:.1e} with max eta '
         f'{eta_ray.max().item():.1f} over 8 directions')

  fprint('=== Schwarzschild deflection against the exact integral ===')
  schwarzschild = BlackHole()
  rays = camera_rays(schwarzschild, 30.0, numpy.deg2rad(80.0),
                     numpy.deg2rad(16.0), 7, 7)
  traced = trace(rays, schwarzschild, 240.0, disc_emission,
                 args=(no_disc, no_disc))
  mass = schwarzschild.mass
  node, weight = numpy.polynomial.legendre.leggauss(400)

  def sweep(b, r_turn, r_max):
    """Returns the azimuth swept between the turning point and r_max."""
    # The substitution r = r_turn + s**2 removes the root of the integrand
    # at the turning point.
    s = 0.5 * numpy.sqrt(r_max - r_turn) * (node + 1.0)
    r = r_turn + s * s
    root = numpy.sqrt(1.0 - b * b / (r * r) * (1.0 - 2.0 * mass / r))
    return numpy.sqrt(r_max - r_turn) * (weight @ (s * b / (r * r * root)))

  escaped = numpy.asarray(traced.tags) == ESCAPED
  final = numpy.asarray(traced.states)
  pairs = zip(numpy.asarray(rays)[escaped], final[escaped])
  worst = 0.0
  for y0, y1 in pairs:
    # The path is traced opposite to the momentum.
    x0, x1, axis = y0[1:4], y1[1:4], numpy.cross(y0[1:4], -y0[5:8])
    b = numpy.linalg.norm(axis) / -y0[4]
    r_turn = max(numpy.roots([1.0, 0.0, -b * b, 2.0 * mass * b * b]).real)
    exact = sum(sweep(b, r_turn, numpy.linalg.norm(x)) for x in (x0, x1))
    u0, u1 = x0 / numpy.linalg.norm(x0), x1 / numpy.linalg.norm(x1)
    normal = axis / numpy.linalg.norm(axis)
    swept = numpy.arctan2(numpy.cross(u0, u1) @ normal, u0 @ u1)
    worst = max(worst, abs(swept % (2.0 * numpy.pi) - exact))
  fprint(f'  max difference {worst:.1e} rad over {int(escaped.sum())} '
         'escaped rays')


#### Command line


def main():
  """Renders an image or runs the tests."""
  parser = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
  parser.add_argument('--spin', type=float, default=0.6, help='a / M')
  parser.add_argument('--charge', type=float, default=0.3, help='Q / M')
  parser.add_argument('--inclination', type=float, default=84.0,
                      help='camera angle from the spin axis in degrees')
  parser.add_argument('--distance', type=float, default=50.0,
                      help='camera coordinate distance in Schwarzschild radii')
  parser.add_argument('--fov', type=float, default=45.0,
                      help='vertical field of view in degrees')
  parser.add_argument('--width', type=int, default=1024)
  parser.add_argument('--height', type=int, default=1024)
  parser.add_argument('--mass', type=float, default=1e9, help='solar masses')
  parser.add_argument('--eddington', type=float, default=1e-5,
                      help='disc luminosity over the Eddington limit')
  parser.add_argument('--disc-radius', type=float, default=15.0,
                      help='outer disc radius in Schwarzschild radii')
  parser.add_argument('--no-disc', action='store_true')
  parser.add_argument('--sky', type=float, default=None,
                      help='sky brightness relative to the disc peak. '
                           'Default 0.03 for stars and 1.0 for an image')
  parser.add_argument('--background', type=str, default=None,
                      help='equirectangular photo of the sky')
  parser.add_argument('--supersample', type=int, default=1,
                      help='rays per pixel edge')
  parser.add_argument('--out', type=str, default='black_hole.png')
  parser.add_argument('--check', action='store_true',
                      help='run the tests instead of rendering')
  args = parser.parse_args()
  if args.check:
    check()
    return

  started = time.monotonic()
  if not args.spin ** 2 + args.charge ** 2 <= 1.0:
    sys.exit('(a*)**2 + (Q*)**2 is larger than 1. '
             'That is a naked singularity.')
  if not 0.0 < args.fov < 180.0:
    sys.exit('fov has to be between 0 and 180 degrees.')
  if args.supersample < 1:
    sys.exit('supersample has to be at least 1.')
  bh = BlackHole(spin=0.5 * args.spin, charge=0.5 * args.charge)
  the_isco = isco(bh).item()
  if not args.no_disc and not args.disc_radius > the_isco:
    sys.exit('The disc has to reach past the ISCO at '
             f'{the_isco:.4g}.')
  if args.no_disc:
    radii = kelvin = jnp.zeros(2)
    disc_peak = 1.0
  else:
    radii, kelvin = disc_temperature(bh, args.disc_radius, args.mass,
                                     args.eddington)
    disc_peak = float(blackbody_rgb(kelvin.max()).max())
    print(f'r+ {bh.horizon:.4f}  ISCO {radii[0].item():.4f}  '
          f'T_peak {kelvin.max().item():.0f} K')
  if args.background:
    sky = image_sky(args.background)
  else:
    # Cells the size of a pixel so that no star is lost between pixels.
    sky = star_sky(min(8192, int(args.height * 180.0 / args.fov)))
  if args.sky is None:
    args.sky = 1.0 if args.background else 0.03
  sky = sky * (args.sky * disc_peak)
  ss = args.supersample
  width, height = ss * args.width, ss * args.height
  # The memory is bounded by bands of two million rays.
  band = max(1, (1 << 21) // (ss * width))
  image = []
  for start in range(0, args.height, band):
    stop = min(start + band, args.height)
    states = camera_rays(bh, args.distance, numpy.deg2rad(args.inclination),
                         numpy.deg2rad(args.fov), width, height,
                         slice(ss * start, ss * stop))
    part = render(bh, states, radii, kelvin, sky, width, ss * (stop - start))
    image.append(part.reshape(stop - start, ss, args.width, ss, 3).mean(
        axis=(1, 3)))
  save_png(args.out, numpy.concatenate(image))
  print(f'{args.width}x{args.height}  {time.monotonic() - started:.0f} s'
        f'  {args.out}')


if __name__ == '__main__':
  main()
