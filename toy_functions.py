import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from flax.training import train_state
import optax
import numpy as np

from chebykan_layer import ChebyKAN

np.random.seed(314)

def generate_data(batch_size: int):
    
    while True:
        key = jax.random.PRNGKey(0)  # Initialize a random number generator key
        key, subkey = jax.random.split(key)  # Split key to maintain randomness across batches
        x = jax.random.uniform(subkey, shape=(batch_size, 1), minval=-1.0, maxval=1.0)  # Uniform distribution over [-1, 1]
        
        key, subkey = jax.random.split(key)
        y = jax.random.uniform(subkey, shape=(batch_size, 1), minval=-1.0, maxval=1.0)  # Uniform distribution over [-1, 1]
        
        # f(x, y) = exp(sin(pix) + y^2)
        f_xy = jnp.exp(jnp.sin(jnp.pi * x) + y**2)
        
        yield jnp.hstack((x, y)), f_xy


# Model Definitions
class MLPModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x

class KANModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = ChebyKAN(in_features=2, out_features=8, degree=8)(x)
        x = ChebyKAN(in_features=8, out_features=1, degree=8)(x)
        return x

# Initialize models
def create_train_state(rng, model: MLPModel | KANModel, learning_rate=0.01):
    params = model.init(rng, jnp.ones([1, 2]))  # dummy input for initialization
    optimizer = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

def param_count(params):
    return sum(x.size for x in jax.tree_util.tree_leaves(params))

@jax.jit
def train_step(state: train_state.TrainState, inputs, targets) :
    def loss_fn(params):
        model_output = state.apply_fn(params, inputs)
        loss = jnp.mean((model_output - targets)**2) ** 0.5
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

max_steps = 10000_000
model = MLPModel()
rng = random.PRNGKey(0)

params = model.init(rng, jnp.ones([1, 2]))
print(param_count(params))

state = create_train_state(rng, model, learning_rate=0.001)

# Train the model
for step, (inputs, targets) in enumerate(generate_data(batch_size=256)):
    if step >= max_steps:
        break
    state, loss = train_step(state, inputs, targets)
    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss}")