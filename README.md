# dictionnary_learning
[Presentation link](https://docs.google.com/presentation/d/1ocrZnrfmXJ5VwVtgdtoziGQrNzY5xx9TEyR_2ZuxOQE/edit#slide=id.p)

# Dictionary Learning with OMP and k-SVD

## Project Overview

This project implements two core algorithms for dictionary learning in sparse signal representation: **Orthogonal Matching Pursuit (OMP)** and **k-SVD (k-Singular Value Decomposition)**.

The main objective is to build and evaluate these algorithms on both synthetically generated noisy signals and image data from the **MNIST** dataset.

---

## Algorithms

### Orthogonal Matching Pursuit (OMP)

OMP is a greedy algorithm used to find sparse approximations of signals. Given a dictionary of basis elements, OMP iteratively selects the atoms (dictionary elements) that best match the residual part of the signal. At each step:

1. The dictionary atom most correlated with the current residual is selected.
2. The signal is projected onto the space spanned by the selected atoms.
3. The residual is updated.
4. The process repeats until a stopping criterion (e.g., sparsity level or error threshold) is met.

OMP is particularly useful due to its simplicity and efficiency for sparse coding tasks.

### k-SVD

k-SVD is an iterative algorithm used to learn an optimal dictionary for sparse representations. It alternates between two steps:

1. **Sparse Coding**: Given a fixed dictionary, find the sparse representations of the signals (commonly using OMP).
2. **Dictionary Update**: Update each dictionary atom to better fit the data by applying Singular Value Decomposition (SVD) to a residual matrix corresponding to that atom.

This alternating minimization continues until convergence. k-SVD is widely used in signal processing and image denoising tasks due to its effectiveness at learning compact and meaningful dictionaries.

---

## Testing and Evaluation

The implemented algorithms were tested on:

- **Synthetic Noisy Data**: Randomly generated signals with added Gaussian noise to assess robustness and reconstruction capability.
- **MNIST Dataset**: A standard dataset of handwritten digits used to evaluate performance on real-world high-dimensional data.


