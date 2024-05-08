from flax import linen as nn
import jax.numpy as jnp
from spline import compute_all_b_splines
import jax

# NOT USED
class KAN(nn.Module):
    in_features: int
    out_features: int
    k: int # degree of the basis polynomials
    num_grid_points: int
    grid_range: tuple[float, float] = (-2.0, 2.0)

    def setup(self):
        # Initialize the grid as a non-trainable parameter
        self.grid = self.variable(
            "params", "grid", 
            lambda rng: jnp.linspace(self.grid_range[0], self.grid_range[1], self.num_grid_points),  # Removed the shape argument as it's not used
            None
        )
        self.num_basis = self.num_grid_points + self.k - 1
        self.coefficients = self.param("coefficients", jax.random.normal, (self.out_features, self.in_features, self.num_basis))
        self.w = self.param("w", jax.random.normal, (self.out_features, self.in_features))

    def __call__(self, inputs: jnp.ndarray):
        assert inputs.shape[1] == self.in_features, f"Expected input shape ({inputs.shape[0]}, {self.in_features}), got {inputs.shape}"
        # x has shape (batch_size, in_features)
        grid = self.grid.value
        indices = jnp.arange(self.num_grid_points + self.k - 1)

        def run_for_single_input(i):
            x = inputs[i]
            b_spline_values = compute_all_b_splines(x, indices, self.k, grid).T # shape (in_features, num_basis)

            def phi_i_j(i, j):
                return self.w[i, j] * (b_spline_values[j] @ self.coefficients[i, j] / self.num_basis ** 0.5 + jax.nn.silu(x[j]))
            
            def phi_i(i):
                return jnp.sum(jax.vmap(lambda j: phi_i_j(i, j))(jnp.arange(self.in_features))) / self.in_features ** 0.5

            return jax.vmap(lambda i: phi_i(i))(jnp.arange(self.out_features))
        
        return jax.vmap(run_for_single_input)(jnp.arange(inputs.shape[0]))
    
    def num_params(self):
        num_basis = self.num_grid_points + self.k - 1
        return self.in_features * self.out_features * num_basis + self.out_features * self.in_features
        

# model = KNN(in_features=2, out_features=3, k=2, num_grid_points=10)

# key = jax.random.PRNGKey(0)

# x = jax.random.normal(key, (3, 2))

# params = model.init(key, x)

# y = model.apply(params, x)

# fast_apply = jax.jit(model.apply)

# import time

# start = time.time()

# fast_apply(params, x)

# for i in range(1000):
#     fast_apply(params, x)

# print("Time taken for 1000 iterations:", time.time() - start)

# print(y.shape)

# print("Number of parameters:", model.num_params()) # 60 + 6 = 66