import json
from transformer import Config
from transformer import Transformer
from transformer import KANTransformer
from transformer import MAX_LEN
import numpy as np
import jax
import jax.numpy as jnp
from tiny_stories import EOS_TOKEN
import tiktoken
from tiktoken import Encoding

EOS_TOKEN_INDEX = 50256

KAN_PATH = "checkpoints/params_KAN.npy"
MLP_PATH = "checkpoints/params_MLP.npy"

def make_model(config: Config):
    if config["block_type"] == "MLP":
        return Transformer(
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            n_layers=config['n_layers'],
        )
    elif config["block_type"] == "KAN":
        return KANTransformer(
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            n_layers=config['n_layers'],
        )
    raise ValueError(f"Unknown block type: {config['block_type']}")

def fetch_weights(config: Config):
    if config["block_type"] == "MLP":
        return np.load(MLP_PATH, allow_pickle=True)
    elif config["block_type"] == "KAN":
        return np.load(KAN_PATH, allow_pickle=True)
    raise ValueError(f"Unknown block type: {config['block_type']}")

def generate(prompt: str, model: Transformer | KANTransformer, params: dict, enc: Encoding):
    fast_apply = jax.jit(model.apply)
    prompt_tokens = enc.encode(prompt)
    while True:
        prompt_length = len(prompt_tokens)
        prompt_tokens_array = jnp.array([prompt_tokens + [0] * (MAX_LEN - len(prompt_tokens))], dtype=jnp.int32)
        if len(prompt_tokens) > MAX_LEN or prompt_tokens[-1] == EOS_TOKEN_INDEX:
            break
        all_logits = fast_apply(params, prompt_tokens_array)
        logit = all_logits[0, prompt_length - 1]
        # Greedy sampling
        next_token = jnp.argmax(logit)
        prompt_tokens.append(next_token.item())
    return enc.decode(prompt_tokens)


config = Config(
    d_model=128,
    n_heads=8,
    n_layers=16,
    learning_rate=1e-5,
    batch_size=16,
    weight_decay=0.001,
    block_type="KAN", 
)

model = make_model(config)

params = fetch_weights(config).item()

enc = tiktoken.encoding_for_model("gpt2")

prompt = "Once upon a time, a dog was"

print("===\n\n")

print(generate(prompt, model, params, enc))


