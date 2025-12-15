# TIAKT: Tri-level Interactive Attention Knowledge Tracing

##⚠️ Note
This repository contains pseudocode that illustrates the core mechanisms of our TIAKT model. The pseudocode is designed to help readers understand the key algorithmic ideas and model architecture presented in our paper. It is not intended to be directly executable but rather serves as a reference for understanding the methodology. If you have any questions or require further clarification, please feel free to contact us.
w18724284923@outlook.com
## 📋 Overview

TIAKT is a novel knowledge tracing model that incorporates three levels of memory interaction inspired by educational psychology:

1. **Short-term Memory Encoder**: Captures recent learning patterns using monotonic attention
2. **Advance Organizer Module**: Activates relevant prior knowledge based on current task
3. **Neural Memory Module**: Dynamically updates knowledge state with gating mechanism
4. **Persistent Memory Module**: Consolidates long-term stable knowledge
5. **Memory Fusion Module**: Integrates multi-level memories for prediction

## 🏗️ Architecture

```
Input: (questions, answers, problem_ids)
         ↓
┌─────────────────────────────────────┐
│     Multi-level Embedding Layer     │
│   e_t^q = e_base + μ_t * e_diff     │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│    Short-term Memory Encoder        │
│  (Dual-path Transformer with        │
│   Monotonic Attention)              │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│     Advance Organizer Module        │
│  α_t = softmax(h·W_a·e_t^q)         │
│  a_t = Σ α_t,i · h_i                │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│     Neural Memory Module            │
│  β_t = σ(w_β·s_t + b_β)             │
│  M^t = β·M^{t-1} + (1-β)·M^curr     │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│    Persistent Memory Module         │
│  P^t = W_1·P^{t-1} + W_2·M̄ + b     │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│     Memory Fusion (Transformer)     │
│  C = Fuse([S; M; P])                │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│     Prediction Layer                │
│  p_t = σ(FC([C_t; e_t^q]))          │
└─────────────────────────────────────┘
         ↓
Output: Probability of correct answer
```

## 📁 Project Structure

```
TIAKT/
├── pseudocode/
│   ├── tiakt_pseudocode.py      # Core algorithm pseudocode
│   └── README.md                # This file
├── data/
│   ├── assist2009_pid/          # ASSISTments 2009 dataset
│   ├── assist2017_pid/          # ASSISTments 2017 dataset
│   ├── assist2015/              # ASSISTments 2015 dataset
│   └── statics/                 # Statics 2011 dataset
└── results/                     # Experimental results
```

## 🔬 Key Components

### 1. Monotonic Attention (from AKT)
```python
# Position-aware exponential decay
position_distance = |i - j|
decay = exp(γ * position_distance)
attention = softmax(QK^T / √d * decay)
```

### 2. Advance Organizer
```python
# Task-oriented biased attention
α_t = softmax(h_i^T · W_a · e_t^q / √d)
prior_knowledge = Σ α_t,i · h_i
s_t^short = σ(W_h · [e_t^q; prior_knowledge])
```

### 3. Gated Memory Update
```python
# Neural memory with gating
gate = σ(W_gate · s_t + b_gate)
M^t = gate ⊙ M^{t-1} + (1 - gate) ⊙ candidate
```

### 4. Multi-level Fusion
```python
# Transformer-based fusion
Z_in = concat([S^short, M^t, P^t])
Z_out = TransformerEncoder(Z_in)
C = Z_out[:seq_len]  # Cognitive state
```

### Cross-Task Transfer

- Zero-shot transfer between datasets
- Fine-tuning with frozen Transformer layers
- Analysis of transferable components

### Supplementary Experiments

- Cold start performance (low-frequency skills)
- Sequence length analysis
- Position effect analysis
- Difficulty-based analysis
- 
### Hardware Environment
GPU: NVIDIA GPU with CUDA support
Framework: PyTorch
Python: 3.8+

## 📝 License

This code is released for academic research purposes only.

## 🙏 Acknowledgments

This work builds upon:
- AKT (Ghosh et al., 2020) for monotonic attention mechanism
- Educational psychology theories (Ausubel's Advance Organizer)
