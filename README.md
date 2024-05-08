# Training small GPT-2 style using KANs instead of MLPs in JAX

This repository compares transformers using multilayer perceptron (MLP) and Kolmogorov-Arnold networks (KAN) layers.

Key points:
- Uses [Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756) but with Chebyshev polynomials as the basis (inspired by [this](https://github.com/SynodicMonth/ChebyKAN) repo).
- The tanh activation function is used to keep the activation values within [-1, 1] rather than dynamic grids.


Hyperparameters:
- `d_model`: 128
- `n_heads`: 8
- `n_layers`: 16
- `learning_rate`: 1e-5
- `batch_size`: 16
- `weight_decay`: 0.001
