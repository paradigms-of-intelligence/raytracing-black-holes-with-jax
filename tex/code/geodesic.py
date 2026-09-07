def geodesic_dy_dt_func_from_christoffel_func(christoffel_func):
  def dy_dt_func(motion_state_y):
    c_pos, c_tangent = jnp.unstack(motion_state_y.reshape(2, -1), axis=0)
    rate_tangent = -jnp.einsum('ijk,j,k->i', christoffel_func(c_pos),
                               c_tangent, c_tangent)
    return jnp.concatenate([c_tangent, rate_tangent], axis=0)
  return dy_dt_func


def geodesic_dy_dt_func_from_metric_func(metric_func):
  christoffel_func = christoffel_func_from_metric_func(metric_func)
  return geodesic_dy_dt_func_from_christoffel_func(christoffel_func)    
