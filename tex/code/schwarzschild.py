def schwarzschild_L2_func(txyz, d_txyz, r_schwarzschild=1.0):
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
