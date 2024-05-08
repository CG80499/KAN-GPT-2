# Training small GPT-2 style models using KANs instead of MLPs in JAX

This repository compares transformers using multilayer perceptron (MLP) and Kolmogorov-Arnold networks (KAN) layers.

**Key points**:
- Uses [Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756) but with Chebyshev polynomials as the basis (inspired by [this](https://github.com/SynodicMonth/ChebyKAN) repo).
- The tanh function is used to keep the activation values within [-1, 1] rather than dynamic grids.
- Both models are trained on 134M tokens of [TinyStories](https://arxiv.org/abs/2305.07759).
- They both use standard GPT-2 architecture (other than the KAN part).
- The MLP version has 3.3M non-embedding weights and the KAN model has 2.5M non-embedding weights (~25% fewer).

**Results**:

They both achieve a final loss of ~2.46 (despite the KAN model having fewer parameters).
![image](https://github.com/CG80499/KAN-GPT-2/blob/master/images/loss_graph.jpeg)

**Hyperparameters**:
- `d_model`: 128
- `d_mlp`: 768
- `n_heads`: 8
- `n_layers`: 16
- `learning_rate`: 1e-5
- `batch_size`: 16
- `weight_decay`: 0.001
- `optimizer`: adamw
- `seq_len`: 64

**Hardware**: Single 1080ti GPU

**Wandb** [link](https://wandb.ai/cg123/kan-transformer?nw=nwusercg123).
