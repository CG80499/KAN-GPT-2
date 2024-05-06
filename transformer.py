import numpy as np
from flax import linen as nn
import jax
import jax.numpy as jnp
from typing import TypedDict
import numpy as np
from tiny_stories import TinyStoriesDataset
from tiny_stories import TOKENIZER_SIZE
from flax.training import train_state
import wandb
import optax

D_TYPE = jnp.float32

MAX_LEN = 64

class MLP(nn.Module):
    d_inner: int

    @nn.compact
    def __call__(self, x):
        d_outer = x.shape[-1]
        x = nn.Dense(features=self.d_inner, param_dtype=D_TYPE)(x)
        x = nn.gelu(x)
        x = nn.Dense(features=d_outer, param_dtype=D_TYPE)(x)
        return x
    
class MLPBlock(nn.Module):
    d_inner: int

    @nn.compact
    def __call__(self, x):
        y = nn.LayerNorm(param_dtype=D_TYPE)(x)
        y = MLP(self.d_inner)(y)
        return x + y
    
class SelfAttentionBlock(nn.Module):
    d_model: int
    n_heads: int

    @nn.compact
    def __call__(self, x):
        # Shape (batch_size, seq_len, d_model)
        n_heads, d_model = self.n_heads, self.d_model
        assert d_model % n_heads == 0, 'n_heads must divide d_model'
        # Shape (batch_size, num_heads, seq_len, seq_len)
        mask = jnp.ones((x.shape[0], n_heads, x.shape[1], x.shape[1]))
        # Create diagonal mask
        mask = jnp.tril(mask)
        y = nn.LayerNorm(param_dtype=D_TYPE)(x)
        attn = nn.MultiHeadDotProductAttention(
            num_heads=n_heads, qkv_features=d_model // n_heads, out_features=d_model, param_dtype=D_TYPE
        )(y, mask=mask)
        return x + attn

class Transformer(nn.Module):
    d_model: int
    n_heads: int
    n_layers: int
    d_inner_factor: int = 4

    @nn.compact
    def __call__(self, x):
        # Shape (batch_size, seq_len) -> (batch_size, seq_len, d_model)
        x = nn.Embed(num_embeddings=TOKENIZER_SIZE, features=self.d_model, param_dtype=D_TYPE)(x)
        pos_emb = nn.Embed(num_embeddings=MAX_LEN, features=self.d_model, param_dtype=D_TYPE)(jnp.arange(MAX_LEN))
        x = x + pos_emb
        for _ in range(self.n_layers):
            x = SelfAttentionBlock(self.d_model, self.n_heads)(x)
            x = MLPBlock(self.d_model * self.d_inner_factor)(x)
        # Shape (batch_size, seq_len, d_model) -> (batch_size, seq_len, vocab_size)
        x = nn.Dense(features=TOKENIZER_SIZE, use_bias=False, param_dtype=D_TYPE)(x)
        return x
    
def param_count(params):
    return sum(x.size for x in jax.tree_util.tree_leaves(params)) / 1e6

class Config(TypedDict):
    d_model: int
    n_heads: int
    n_layers: int
    d_inner_factor: int
    learning_rate: float
    max_steps: int
    batch_size: int
    weight_decay: float

def print_param_dtypes(params):
    for name, value in params.items():
        if isinstance(value, dict):
            print(f"Layer: {name}")
            print_param_dtypes(value)  # Recursive call for nested layers
        else:
            print(f"Parameter {name}: param_dtype={value.dtype}")


def masked_cross_entropy(logits: jnp.ndarray, targets: jnp.ndarray, mask: jnp.ndarray):
    # logits shape: (batch_size, seq_len, vocab_size)
    # targets shape: (batch_size, seq_len)
    # mask shape: (batch_size, seq_len)
    # shift everything by 1
    logits = logits[:, :-1, :]
    targets = targets[:, 1:]
    mask = mask[:, 1:]
    vocab_size = logits.shape[-1]
    one_hot_targets = jax.nn.one_hot(targets, vocab_size)
    # one_hot_targets shape: (batch_size, seq_len, vocab_size)
    log_probs = jax.nn.log_softmax(logits)
    loss = -jnp.sum(log_probs * one_hot_targets, axis=-1)
    # loss shape: (batch_size, seq_len)
    loss = loss * mask
    # Flatten everything divide by the sum of the mask
    total_tokens = jnp.sum(mask.flatten())
    return jnp.sum(loss.flatten()) / total_tokens
    


def create_train_state(rng, config):
    model = Transformer(
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_inner_factor=config['d_inner_factor']
    )
    params = model.init(rng, jnp.ones((config['batch_size'], MAX_LEN), dtype=jnp.int32))
    optimizer = optax.adamw(learning_rate=config['learning_rate'], weight_decay=config['weight_decay'])
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer
    )

@jax.jit
def train_step(state: train_state.TrainState, batch: jnp.ndarray, mask: jnp.ndarray):
    def loss_fn(params):
        logits = state.apply_fn(params, batch)
        return masked_cross_entropy(logits, batch, mask)

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

rng = jax.random.PRNGKey(0)

config = Config(
    d_model=128,
    n_heads=8,
    n_layers=16,
    d_inner_factor=4,
    learning_rate=1e-5,
    batch_size=16,
    weight_decay=0.001,
)

# 15.123456M params for normal transformer with 2.249471999999999M non-embedding (daily-shape-24)

#wandb.init(project="kan-transformer", config=config)

print("Creating model...")
state = create_train_state(rng, config)
print("Number of parameters: ", param_count(state.params))
print("Number of non-embedding parameters:", param_count(state.params) - (config["d_model"] * TOKENIZER_SIZE * 2 + config["d_model"] * MAX_LEN) / 1e6)

for step, (batch, mask) in enumerate(TinyStoriesDataset(max_len=MAX_LEN).create_batches(config['batch_size'])):
    state, loss = train_step(state, batch, mask)
    if step % 50 == 0:
        print(f"Step {step}, Loss: {loss}")
        #wandb.log({"loss": loss})
    # save every 1000 steps
    if step % 1000 == 0:
        print("Saving params...")
        np.save("checkpoints/params.npy", state.params)
        print("Params saved.")