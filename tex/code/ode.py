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
def rk4_estimate(f, y0, ds):
  k1 = f(y0)
  k2 = f(y0 + (0.5 * ds) * k1)
  k3 = f(y0 + (0.5 * ds) * k2)
  k4 = f(y0 + ds * k3)
  return (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

def simple_estimate(f, y0, ds): return f(y0)

def ode_solve(estimate_func, f, y0, s_samples):
  def inner_func(y_now, s_step):
    rate = estimate_func(f, y_now, s_step)
    y_next = y_now + s_step * rate
    return y_next, y_next
  y_final, ys_collected = jax.lax.scan(inner_func, y0, jnp.diff(s_samples))
  return ys_collected

jax_ode_solve = jax.jit(ode_solve,
                        static_argnames=('f', 'estimate_func'))

circular = jax_ode_solve(
  rk4_estimate,  # Alternative: simple_estimate
  (lambda x: jnp.concatenate([-x[1:2], x[:1]], axis=0)),
  jnp.array([1.0, 0.0]), jnp.linspace(0.0, 10.0, 201))
