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
