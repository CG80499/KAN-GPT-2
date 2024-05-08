from flax import linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import jax.random as random
from functools import partial
from jax import jit, vmap

# NOT USED

# for testing
def compute_b_splines(x: float, i: int, k: int, grid: list[float]) -> float:
    # x is real number
    # k is degree of the polynomial
    # i is the index of the polynomial
    # grid is 1D array of knots
    # check bounds
    if i + k + 1 >= len(grid):
        return 0.0
    if k == 0:
        return 1.0 if grid[i] <= x < grid[i + 1] else 0.0
    b_i_km1 = compute_b_splines(x, i, k - 1, grid)
    b_i1_km1 = compute_b_splines(x, i + 1, k - 1, grid)
    return (x - grid[i]) / (grid[i + k] - grid[i]) * b_i_km1 + (grid[i + k + 1] - x) / (grid[i + k + 1] - grid[i + 1]) * b_i1_km1

def compute_b_splines_array(x: jnp.ndarray, i: int, k: int, grid: jnp.ndarray) -> jnp.ndarray:
    # x a 1D array of real numbers
    # k is degree of the polynomial
    # i is the index of the polynomial
    # grid is 1D array of knots
    if k == 0:
        return jnp.where((grid[i] <= x) & (x < grid[i + 1]), 1.0, 0.0)
    b_i_k_minus_one = compute_b_splines_array(x, i, k - 1, grid)
    b_i_plus_k_minus_one = compute_b_splines_array(x, i + 1, k - 1, grid)
    return (x - grid[i]) / (grid[i + k] - grid[i]) * b_i_k_minus_one + (grid[i + k + 1] - x) / (grid[i + k + 1] - grid[i + 1]) * b_i_plus_k_minus_one

@partial(jit, static_argnums=(1,))
def extend_grid(grid: jnp.ndarray, k: int) -> jnp.ndarray:
    h = (grid[-1] - grid[0]) / (len(grid) - 1)
    right_extension = jnp.arange(1, k + 1) * h + grid[-1]
    left_extension = jnp.arange(1, k + 1) * h + grid[0]
    return jnp.concatenate([left_extension, grid, right_extension])

_compute_all_b_splines = jit(vmap(compute_b_splines_array, in_axes=(None, 0, None, None)), static_argnums=(2,))

def compute_all_b_splines(x: jnp.ndarray, indices: jnp.ndarray, k: int, grid: jnp.ndarray) -> jnp.ndarray:
    extended_grid = extend_grid(grid, k)
    return _compute_all_b_splines(x, indices, k, extended_grid)