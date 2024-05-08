from flax import linen as nn
import jax.numpy as jnp
import jax

# Inspired by https://github.com/SynodicMonth/ChebyKAN/blob/main/ChebyKANLayer.py

class ChebyKAN(nn.Module):
    in_features: int
    out_features: int
    degree: int # degree of the basis polynomials

    def setup(self):
        assert self.degree > 0, "Degree of the Chebyshev polynomials must be greater than 0"
        mean, std = 0.0, 1/ (self.in_features * (self.degree + 1))
        self.coefficients = self.param("coefficients", lambda key, shape: mean + std * jax.random.normal(key, shape), (self.in_features, self.out_features, self.degree+1))

    def __call__(self, x):
        # x: (batch_size, in_features)
        # normalize x between -1 and 1
        x = jnp.tanh(x)
        cheby_values = jnp.ones((x.shape[0], self.in_features, self.degree+1))
        cheby_values = cheby_values.at[:, :, 1].set(x)
        for i in range(2, self.degree+1):
            next_value = 2 * x * cheby_values[:, :, i-1] - cheby_values[:, :, i-2]
            cheby_values = cheby_values.at[:, :, i].set(next_value)
        # cheby_values: (batch_size, in_features, degree+1)
        # multiply by coefficients (in_features, out_features, degree+1)
        return jnp.einsum('bid,ijd->bj', cheby_values, self.coefficients)