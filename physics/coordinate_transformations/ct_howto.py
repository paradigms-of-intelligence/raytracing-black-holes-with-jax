import jax
from jax import numpy as jnp
import numpy


def tprint(a, name='', round_to=6, file=None):
  """Pretty-prints multi-index array contents."""
  a = numpy.asarray(a).round(round_to)
  num_items = (a.ravel() != 0).sum().item()
  print(f'=== array {name}{a.shape}: {num_items} non-small entries ===',
        file=file)
  for indices, value in numpy.ndenumerate(a):
    if value != 0:
      print(f'{str(indices)[1:-1]:20s}: {value}', file=file)


def get_connection_func(*,
                        connection_func_x,
                        x_from_y_func,
                        y_from_x_func=None):
  """Obtains 'y-coordinate-system' connection coefficients from 'x-coords' C.C.

  Henceforth, the "orginal" coordinate-system will be denoted "`x`"
  and the "new" coordinate system "`y`". In General Relativity,
  the connection coefficients will be the Christoffel symbols.

  Args:
    connection_func_x: Either `None`, indicating that the
      `x`-coordinate-system comes with trivial parallel transport
      (in a more efficient way than a constantly-zero function),
      or a function mapping an `x`-coordinates vector to the
      [dim, dim, dim]-Array of `Gamma^i_jk` connection coefficients,
      index-order being `i,j,k`.
    x_from_y_func: A jax-differentiable function computing
      `x`-coordinates from `y`-coordinates.
    y_from_x_func: Either `None`, or a jax-differentiable function
      computing `y`-coordinates from `x`-coordinates.

  Returns:
    A connection-function with the same signature as `connection_func_y`
    that computes the connection-coefficients for the y-coordinate-system.
  """
  # TODO: allow an option to replace matrix-inversion with linear
  # equation solving. There are numerics subtleties around which
  # approach is preferred under what circumstances.
  # For now, we simply go with matrix inversion.
  dx_dy_from_x_func = jax.jacfwd(x_from_y_func)
  if y_from_x_func is not None:
    dy_dx_from_x_func = jax.jacfwd(y_from_x_func)
    d2y_dxdx_from_x_func = jax.jacfwd(dy_dx_from_x_func)
    d2x_dydy_from_y_func = None  # We will not need that here.
  else:
    dx_dy_from_y_func = jax.jacfwd(x_from_y_func)
    d2y_dxdx_from_x_func = None  # We will not need that here.
    d2x_dydy_from_y_func = jax.jacfwd(dx_dy_from_y_func)
  @jax.jit
  def get_derivatives(coords_y):
    coords_x = x_from_y_func(coords_y)
    dx_dy = dx_dy_from_y_func(coords_y)
    if y_from_x_func is not None:
      dy_dx = dy_dx_from_x_func(coords_x)
      d2y_dxdx = d2y_dxdx_from_x_func(coords_x)
      coord_accel_term = -jnp.einsum(
          'inp,nj,pk->ijk', d2y_dxdx, dx_dy, dx_dy, optimize='greedy')
    else:
      dy_dx = jnp.linalg.inv(dx_dy)
      d2x_dydy = d2x_dydy_from_y_func(coords_y)
      coord_accel_term = jnp.einsum('mjk,im->ijk', d2x_dydy, dy_dx)
    return dy_dx, dx_dy, coord_accel_term
  @jax.jit
  def connection_func_y(coords_y):
    coords_x = x_from_y_func(coords_y)
    dy_dx, dx_dy, coord_accel_term = get_derivatives(coords_y)
    if connection_func_x is None:
      connection_x_in_y_coordinates = jnp.zeros_like(coord_accel_term)
    else:
      connection_x_in_y_coordinates = jnp.einsum(
          'mnp,im,nj,pk->ijk',
          connection_func_x(coords_x),
          dy_dx, dx_dy, dx_dy, optimize='greedy')
    return connection_x_in_y_coordinates + coord_accel_term
  return connection_func_y


if 'DEMO':
  omega = 10.0
  # This gets us the rotating-system coordinates given the
  # not-rotating-system-coordinates.
  @jax.jit
  def fn_rotating_x_from_y(txyz_ycoords):   # ycoords are "rotating"
    ty, xy, yy, zy = jnp.unstack(txyz_ycoords)
    ct, st = jnp.cos(omega * ty), jnp.sin(omega * ty)
    return jnp.stack([ty, xy*ct - yy*st, xy*st + yy*ct, zy], axis=0)
  #
  fn_gamma = get_connection_func(connection_func_x=None,
                                 x_from_y_func=fn_rotating_x_from_y)
  gamma = fn_gamma(jnp.array([7, 12.0, 0.0, 3.0]))  # (t, x, y, z)
  tprint(gamma, 'gamma_rotating', 3)
  # Let's see what happens under a round trip.
  @jax.jit
  def fn_rotating_y_from_x(txyz_ycoords):   # ycoords are "rotating"
    ty, xy, yy, zy = jnp.unstack(txyz_ycoords)
    ct, st = jnp.cos(-omega * ty), jnp.sin(-omega * ty)
    return jnp.stack([ty, xy*ct - yy*st, xy*st + yy*ct, zy], axis=0)
  # Slightly confusing naming, since for the round trip, "x is y now".
  fn_gamma_roundtrip = get_connection_func(
      connection_func_x=fn_gamma,
      x_from_y_func=fn_rotating_y_from_x)
  gamma_re = fn_gamma_roundtrip(jnp.array([12, 3.0, 1.0, 1.0]))  # (t, x, y, z)
  print('# After Round trip: ')
  # TODO: After a round trip, we are again describing the parallel-transport
  # corrections for a "perfectly-Cartesian" coordinate system, so it is a
  # nontrivial check that these must come out as zero.
  tprint(gamma_re, 'gamma_nonrotating', 3)
  assert numpy.allclose(gamma_re, 0, atol=1e-3)


# This prints:
#
# === array gamma_rotating(4, 4, 4): 5 non-small entries ===
# 1, 0, 0             : -1200.0
# 1, 0, 2             : -10.0
# 1, 2, 0             : -10.0
# 2, 0, 1             : 10.0
# 2, 1, 0             : 10.0
# # After Round trip: 
# === array gamma_nonrotating(4, 4, 4): 0 non-small entries ===
