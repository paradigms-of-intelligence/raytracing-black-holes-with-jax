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
import jax
from jax import numpy as jnp

R_E = 6371e3  # [m], Earth's radius.
rad_from_deg = jnp.pi / 180

# The specific geometry of interest:
@jax.jit
def jax_L2_geography(coords_lat_lon, rates):
  lat, lon = jnp.asarray(coords_lat_lon, dtype=jnp.float64)
  v_theta, v_phi = (jnp.asarray(rates, dtype=jnp.float64)
                    * rad_from_deg)
  r_ring = R_E * jnp.cos(lat * rad_from_deg)
  return R_E**2 * v_theta**2 + r_ring**2 * v_phi**2

# Some generic helpers for obtaining algorithmic descriptions of
# geometry from other such algorithmic descriptions.

def scalar_product_func_from_L2_func(L2_func):
  def sp_func(coords, v1, v2):
    v1a = jnp.asarray(v1); v2a = jnp.asarray(v2)
    return 0.25 * (L2_func(coords, v1a + v2a) -
                   L2_func(coords, v1a - v2a))
  return sp_func

def metric_func_from_L2_func(L2_func):
  d2_L2_by_dxi_dxj_func = jax.hessian(L2_func, argnums=1)
  def metric_func(coords_pos):
    return 0.5 * d2_L2_by_dxi_dxj_func(coords_pos, jnp.zeros_like(coords_pos))
  return metric_func

# {pos_coords}, {v1_coords}, {v2_coords} -> {scalar product}
jax_sprod_geography = jax.jit(
  scalar_product_func_from_L2_func(jax_L2_geography))

# Applying the generic tools:

jax_metric_geography = jax.jit(
  metric_func_from_L2_func(jax_L2_geography))

# >>> print(jax_metric_geography(jnp.array((60.0, 25.0))))
# [[1.23643117e+10 0.00000000e+00]
#  [0.00000000e+00 3.09107793e+09]]
