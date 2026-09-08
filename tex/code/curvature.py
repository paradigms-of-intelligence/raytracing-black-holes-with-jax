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
def riemann_func_from_christoffel_func(christoffel_func):
  jacobian_christoffel_func = jax.jacobian(christoffel_func)
  def riemann_func(coords_pos):
    christoffel = christoffel_func(coords_pos)
    d_christoffel = jacobian_christoffel_func(coords_pos)
    # We use jnp.einsum() throughout, also for simple transposition.
    aux = (jnp.einsum('ijlk->ijkl', d_christoffel) +
           jnp.einsum('ikm,mjl->ijkl', christoffel, christoffel))
    # The whole expression is aux_ijkl - aux_ijlk.
    return aux - jnp.einsum('ijkl->ijlk', aux)
  return riemann_func


def einstein_func_from_L2_func(L2_func):
  metric_func = metric_func_from_L2_func(L2_func)
  christoffel_func = christoffel_func_from_metric_func(metric_func)
  riemann_func = riemann_func_from_christoffel_func(christoffel_func)
  def einstein_func(coords):
    metric = metric_func(coords)
    riemann = riemann_func(coords)
    ricci = jnp.einsum('mimj->ij', riemann)
    return (ricci - 0.5 * jnp.einsum('ij,ij->', ricci, jnp.linalg.inv(metric))
            * metric)
  return einstein_func
