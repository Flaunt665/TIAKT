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


## 🔬 Key Components

### 1. Monotonic Attention 
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

# TIAKT: Tri-level Interactive Attention Knowledge Tracing

## Experimental Settings

### 1. Datasets


ASSISTments 2009: Collected from the ASSISTments online learning platform (math) in 2009, this dataset contains 325,637 interactions from 4,151 students, covering 110 knowledge components and 16,891 items. The average interaction sequence length is 78.4 (with a maximum of 1,261), and the overall correctness rate is 65.8\%. Response-time information is available, facilitating analysis of temporal factors in knowledge-state modeling.
	
ASSISTments 2017: Also derived from ASSISTments math learning logs, this dataset is larger in interaction volume, with 942,816 interactions from 1,709 students, covering 102 knowledge components and 3,162 items. It features substantially longer sequences (average 551.7, maximum 3,057) but a lower correctness rate (37.3\%), suggesting higher task difficulty. Response-time information is also included.
	
ASSISTments 2015: Focusing on fundamental mathematics, this dataset has the largest number of students (19,840) and contains 683,801 interactions, covering 100 knowledge components. Sequences are relatively short (average 34.5, maximum 618), and the correctness rate is 73.2\%. Response-time records are not provided, making it suitable for evaluating performance under short-sequence, high-accuracy settings.
	
STATICS: Collected from interactions in a university-level physics course, this dataset includes 189,297 interactions from 282 students and covers a wide range of knowledge components (1,223). The average sequence length is 568.5 (maximum 1,181) with a high correctness rate of 76.5\%. No response-time information is available, and the dataset is useful for assessing performance in discipline-level KT with long sequences and dense knowledge-component spaces.


### 2. Data Preprocessing

```
Data Format (4 lines per student):
Line 1: Sequence length
Line 2: Problem ID sequence (problem_id)
Line 3: Skill ID sequence (skill_id)  
Line 4: Answer sequence (0: incorrect, 1: correct)

Split Ratio:
- Training set: 80%
- Validation set: 10%
- Test set: 10%

Sequence Processing:
- Maximum sequence length: 200
- Filter users with interactions 
- Split long sequences by seqlen
```

### 3. Model Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| n_heads | 8 | Number of attention heads |
| mem_slots | 8 | Number of neural memory slots |
| persistent_slots | 4 | Number of persistent memory slots |
| final_fc_dim | 512 | Output layer dimension |
| dropout | 0.05 | Dropout rate |

### 4. Training Settings

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning rate | 5e-4 |
| Batch size | 24 |
| Max epochs | 50 |
| Early stopping | 10 epochs |
| L2 regularization | 1e-5 |
| Random seed | 224 |

### 5. Evaluation Metrics

- **AUC** (Area Under ROC Curve): Primary evaluation metric
- **Accuracy**: Prediction accuracy
- **F1 Score**: F1 score

### 6. Hardware Environment

```
GPU: NVIDIA GPU with CUDA support
Framework: PyTorch
Python: 3.8+
```

### 7. Cross-Task Transfer Experiments

```
Transfer Settings:
- Source → Target combinations: 12 pairs (pairwise combinations of 4 datasets)
- Transfer modes:
  1. Zero-shot: Direct transfer without training on the target dataset
  2. Fine-tune: Full parameter fine-tuning
  3. Freeze: Freeze Transformer layers, only fine-tune embedding and output layers
```

### 8. Supplementary Experiments

#### 8.1 Cold Start Analysis
```
Grouping Criteria:

- Extremely low (< 50)
-  Low (50–200)
- Medium (200–500)
- High (> 500)
```

#### 8.2 Position Effect Analysis
```
Groups:
- Sequence beginning: Position 1-50
- Sequence middle: Position 51-150
- Sequence end: Position 151-200
```

#### 8.3 Difficulty Analysis
```
Grouping based on historical accuracy:
- Hard problems: Accuracy < 0.3
- Medium problems: 0.3 ≤ accuracy < 0.7
```

### 9. Reproducibility

```bash
# Set random seeds
np.random.seed(224)
torch.manual_seed(224)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

---

## License

This code is released for academic research purposes only.

## 📝 License

This code is released for academic research purposes only.

## 🙏 Acknowledgments

This work builds upon:
- Tians
- Educational psychology theories (Ausubel's Advance Organizer)
