import numpy as np
from flax import linen as nn
import jax
import jax.numpy as jnp
from typing import Literal, TypedDict
import numpy as np
from tiny_stories import TinyStoriesDataset
from tiny_stories import TOKENIZER_SIZE
from flax.training import train_state
import wandb
import optax
from chebykan_layer import ChebyKAN
from time import perf_counter

D_TYPE = jnp.float32

MAX_LEN = 64

class MLP(nn.Module):

    @nn.compact
    def __call__(self, x):
        d_outer = x.shape[-1]
        # 84 is choosen so that the number of parameters matches then KAN layer
        x = nn.Dense(features=768, param_dtype=D_TYPE)(x)
        x = nn.gelu(x)
        x = nn.Dense(features=d_outer, param_dtype=D_TYPE)(x)
        return x
    
class MLPBlock(nn.Module):

    @nn.compact
    def __call__(self, x):
        y = nn.LayerNorm(param_dtype=D_TYPE)(x)
        y = MLP()(y)
        return x + y

class KANLayer(nn.Module):

    @nn.compact
    def __call__(self, x):
        # y has shape (batch_size, seq_len, d_model) -> (batch_size * seq_len, d_model)
        y = x.reshape((-1, x.shape[-1]))
        y = ChebyKAN(in_features=x.shape[-1], out_features=x.shape[-1], degree=8)(y)
        y = y.reshape(x.shape)
        return y

class KANBlock(nn.Module):

    @nn.compact
    def __call__(self, x):
        x_inner = nn.LayerNorm()(x)
        x_inner = KANLayer()(x_inner)
        return x + x_inner


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

    @nn.compact
    def __call__(self, x):
        # Shape (batch_size, seq_len) -> (batch_size, seq_len, d_model)
        x = nn.Embed(num_embeddings=TOKENIZER_SIZE, features=self.d_model, param_dtype=D_TYPE)(x)
        pos_emb = nn.Embed(num_embeddings=MAX_LEN, features=self.d_model, param_dtype=D_TYPE)(jnp.arange(MAX_LEN))
        x = x + pos_emb
        for _ in range(self.n_layers):
            x = SelfAttentionBlock(self.d_model, self.n_heads)(x)
            x = MLPBlock()(x)
        # Shape (batch_size, seq_len, d_model) -> (batch_size, seq_len, vocab_size)
        x = nn.Dense(features=TOKENIZER_SIZE, use_bias=False, param_dtype=D_TYPE)(x)
        return x
    
class KANTransformer(nn.Module):
    d_model: int
    n_heads: int
    n_layers: int

    @nn.compact
    def __call__(self, x):
        # Shape (batch_size, seq_len) -> (batch_size, seq_len, d_model)
        x = nn.Embed(num_embeddings=TOKENIZER_SIZE, features=self.d_model, param_dtype=D_TYPE)(x)
        pos_emb = nn.Embed(num_embeddings=MAX_LEN, features=self.d_model, param_dtype=D_TYPE)(jnp.arange(MAX_LEN))
        x = x + pos_emb
        for _ in range(self.n_layers):
            x = SelfAttentionBlock(self.d_model, self.n_heads)(x)
            x = KANBlock()(x)
        # Shape (batch_size, seq_len, d_model) -> (batch_size, seq_len, vocab_size)
        x = nn.Dense(features=TOKENIZER_SIZE, use_bias=False, param_dtype=D_TYPE)(x)
        return x

class KANHybridTransformer(nn.Module):
    d_model: int
    n_heads: int
    n_layers: int

    @nn.compact
    def __call__(self, x):
        assert self.n_layers % 2 == 0, "n_layers must be even"
        # Shape (batch_size, seq_len) -> (batch_size, seq_len, d_model)
        x = nn.Embed(num_embeddings=TOKENIZER_SIZE, features=self.d_model, param_dtype=D_TYPE)(x)
        pos_emb = nn.Embed(num_embeddings=MAX_LEN, features=self.d_model, param_dtype=D_TYPE)(jnp.arange(MAX_LEN))
        x = x + pos_emb
        for _ in range(self.n_layers // 2):
            x = SelfAttentionBlock(self.d_model, self.n_heads)(x)
            x = MLPBlock()(x)
            x = SelfAttentionBlock(self.d_model, self.n_heads)(x)
            x = KANBlock()(x)
        # Shape (batch_size, seq_len, d_model) -> (batch_size, seq_len, vocab_size)
        x = nn.Dense(features=TOKENIZER_SIZE, use_bias=False, param_dtype=D_TYPE)(x)
        return x

def param_count(params):
    return sum(x.size for x in jax.tree_util.tree_leaves(params)) / 1e6

class Config(TypedDict):
    d_model: int
    n_heads: int
    n_layers: int
    learning_rate: float
    max_steps: int
    batch_size: int
    weight_decay: float
    block_type: Literal["MLP", "KAN", "Hybrid"]


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
    if config["block_type"] == "MLP":
        model = Transformer(
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            n_layers=config['n_layers'],
        )
    elif config["block_type"] == "KAN":
        model = KANTransformer(
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            n_layers=config['n_layers'],
        )
    else:
        model = KANHybridTransformer(
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            n_layers=config['n_layers'],
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
    learning_rate=1e-5,
    batch_size=16,
    weight_decay=0.001,
    block_type="MLP", 
)

# ChebyKAN
# Number of parameters:  15.37536
# Number of non-embedding parameters: 2.5013760000000005

# MLP (hidden size 768)
# Number of parameters:  16.176128
# Number of non-embedding parameters: 3.3021439999999984

# Training loop

if __name__ == "__main__":

    wandb.init(project="kan-transformer", config=config)

    print("Creating model...")
    state = create_train_state(rng, config)
    print("Number of parameters: ", param_count(state.params))
    print("Number of non-embedding parameters:", param_count(state.params) - (config["d_model"] * TOKENIZER_SIZE * 2 + config["d_model"] * MAX_LEN) / 1e6)

    for step, (batch, mask) in enumerate(TinyStoriesDataset(max_len=MAX_LEN).create_batches(config['batch_size'])):
        step_start_time = perf_counter()
        state, loss = train_step(state, batch, mask)
        step_end_time = perf_counter()
        if step % 50 == 0:
            print(f"Step {step}, Loss: {loss}")
            wandb.log({"loss": loss})
            print(f"Time taken for step: {step_end_time - step_start_time}")
            wandb.log({"time": step_end_time - step_start_time})
        # save every 1000 steps
        if step % 1000 == 0:
            print("Saving params...")
            model_type = config["block_type"]
            np.save(f"checkpoints/params_{model_type}.npy", state.params)
            print("Params saved.")