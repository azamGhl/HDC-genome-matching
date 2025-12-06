# HDCgenome – Hyperdimensional Genome Sequence Matching (NumPy Implementation)

This project implements a hyperdimensional computing (HDC)–based framework for
**genome sequence matching**, using high-dimensional binary hypervectors and
similarity search methods.

Unlike deep-learning approaches, this implementation uses **lightweight NumPy
operations** and does not rely on PyTorch or GPU resources.  
The project supports both:

- **Exact matching**  
- **Approximate matching** (with controllable mutation levels)

Making it suitable for experimentation, teaching, and rapid prototyping.

---

## 🚀 Features

- Binary hypervector encoding for DNA bases (A, C, G, T)
- Position-dependent binding using circular permutation + XOR
- Partition-level encoding (subword encoding)
- Hypervector bundling using majority rule
- Exact matching and approximate matching with mutations
- Search using:
  - **Hamming distance**
  - **Cosine similarity**
- Accuracy evaluation + confusion matrix
- Fully reusable `BioHD` class

---

## 📁 Project Structure

