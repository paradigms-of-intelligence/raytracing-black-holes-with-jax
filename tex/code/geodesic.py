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
