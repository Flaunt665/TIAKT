"""
================================================================================
TIAKT: Tri-level Interactive Attention Knowledge Tracing
Pseudocode Version

This file demonstrates the core algorithmic mechanisms of the TIAKT model,
designed for understanding the methodology presented in the paper.
================================================================================
"""

import numpy as np

# ==============================================================================
# Hyperparameter Definitions
# ==============================================================================
"""
d_model: Model hidden dimension (default: 128)
n_heads: Number of attention heads (default: 8)
n_blocks: Number of Transformer blocks (default: 1)
mem_slots: Number of neural memory slots (default: 8)
persistent_slots: Number of persistent memory slots (default: 4)
dropout: Dropout rate (default: 0.05)
"""


# ==============================================================================
# Core Module 1: Multi-level Embedding Layer
# ==============================================================================
class EmbeddingLayer:
    """
    Question Embedding with Difficulty Modulation
    
    Formula:
        e_t^q = e_t^{q,base} + mu_t * e_t^{q,diff}
        e_t^{qa} = e_t^{qa,base} + mu_t * e_t^{qa,diff}
    
    Where:
        e_t^{q,base}: Base question embedding
        e_t^{q,diff}: Difficulty difference embedding  
        mu_t: Question difficulty parameter
    """
    
    def __init__(self, n_question, n_pid, d_model):
        # Base embeddings
        self.q_embed = Embedding(n_question + 1, d_model)      # Question embedding
        self.qa_embed = Embedding(2, d_model)                  # Answer embedding (correct/incorrect)
        
        # Difficulty-related embeddings (when problem IDs are available)
        if n_pid > 0:
            self.difficult_param = Embedding(n_pid + 1, 1)      # Difficulty parameter mu_t
            self.q_embed_diff = Embedding(n_question + 1, d_model)   # Question difficulty difference
            self.qa_embed_diff = Embedding(2 * n_question + 1, d_model)  # Answer difficulty difference
    
    def forward(self, q_data, qa_data, pid_data=None):
        """
        Input:
            q_data: Question ID sequence [batch, seq_len]
            qa_data: Answer sequence [batch, seq_len]  
            pid_data: Problem difficulty ID [batch, seq_len] (optional)
        
        Output:
            q_embed_data: Question embedding [batch, seq_len, d_model]
            qa_embed_data: Answer embedding [batch, seq_len, d_model]
        """
        # Base question embedding
        q_embed_data = self.q_embed(q_data)
        
        # Answer embedding (combine answer with question)
        qa_indicator = (qa_data - q_data) // n_question  # 0: incorrect, 1: correct
        qa_embed_data = self.qa_embed(qa_indicator) + q_embed_data
        
        # Difficulty modulation
        if pid_data is not None and self.n_pid > 0:
            q_embed_diff_data = self.q_embed_diff(q_data)
            mu_t = self.difficult_param(pid_data)  # Difficulty parameter
            
            # Difficulty-modulated embedding
            q_embed_data = q_embed_data + mu_t * q_embed_diff_data
            
            qa_embed_diff_data = self.qa_embed_diff(qa_data)
            qa_embed_data = qa_embed_data + mu_t * (qa_embed_diff_data + q_embed_diff_data)
            
            # L2 regularization loss
            reg_loss = (mu_t ** 2).sum() * l2_weight
        
        return q_embed_data, qa_embed_data, reg_loss


# ==============================================================================
# Core Module 2: Short-term Memory Encoder
# Based on AKT's dual-path Transformer structure
# ==============================================================================
class ShortTermEncoder:
    """
    Short-term Memory Encoder - Using Monotonic Attention Mechanism
    
    Structure:
        1. Answer sequence self-attention (blocks_1): Capture historical answer patterns
        2. Question-answer cross-attention (blocks_2): Establish question-answer associations
    
    Key: Using exponentially decaying monotonic attention
    """
    
    def __init__(self, d_model, n_heads, n_blocks, d_ff, dropout):
        # Answer sequence self-attention layers
        self.blocks_1 = [TransformerLayer(...) for _ in range(n_blocks)]
        # Question-answer cross-attention layers
        self.blocks_2 = [TransformerLayer(...) for _ in range(n_blocks * 2)]
    
    def forward(self, q_embed_data, qa_embed_data):
        """
        Input:
            q_embed_data: Question embedding [batch, seq, d_model]
            qa_embed_data: Answer embedding [batch, seq, d_model]
        
        Output:
            h_t: Short-term knowledge state [batch, seq, d_model]
        """
        y = qa_embed_data  # Answer sequence
        x = q_embed_data   # Question sequence
        
        # Step 1: Answer sequence self-attention
        for block in self.blocks_1:
            y = block(query=y, key=y, values=y, mask=CAUSAL_MASK)
        
        # Step 2: Question-answer cross-attention (alternating)
        is_self_attn = True
        for block in self.blocks_2:
            if is_self_attn:
                x = block(query=x, key=x, values=x, mask=CAUSAL_MASK, apply_pos=False)
            else:
                x = block(query=x, key=x, values=y, mask=PEEK_MASK, apply_pos=True)
            is_self_attn = not is_self_attn
        
        return x  # Short-term knowledge representation h_t


class MonotonicMultiHeadAttention:
    """
    Monotonic Attention Mechanism (from AKT)
    
    Core Formula:
        scores = Q·K^T / sqrt(d_k)
        
        # Compute position decay effect
        position_effect = |i - j|  # Position distance
        decay = exp(gamma * position_effect)  # Exponential decay
        
        # Apply decay
        scores = scores * decay
        attention = softmax(scores)
        output = attention · V
    """
    
    def forward(self, q, k, v, mask, zero_pad=False):
        # Linear projection
        Q = self.W_q(q)  # [batch, seq, d_model]
        K = self.W_k(k)
        V = self.W_v(v)
        
        # Split into heads
        Q = reshape_to_heads(Q)  # [batch, n_heads, seq, d_k]
        K = reshape_to_heads(K)
        V = reshape_to_heads(V)
        
        # Attention scores
        scores = matmul(Q, K.T) / sqrt(d_k)
        
        # === Key: Monotonic position decay ===
        seq_len = scores.shape[-1]
        pos_i = arange(seq_len).expand(seq_len, -1)
        pos_j = pos_i.T
        position_distance = abs(pos_i - pos_j)
        
        # Compute decay effect (learnable parameter gamma)
        gamma = -softplus(self.gamma_param)
        decay_effect = exp(gamma * position_distance)
        decay_effect = clamp(decay_effect, min=1e-5, max=1e5)
        
        # Apply decay
        scores = scores * decay_effect
        
        # Masking and normalization
        scores = masked_fill(scores, mask == 0, -inf)
        attention = softmax(scores, dim=-1)
        
        # Zero padding for first position (prevent leakage)
        if zero_pad:
            attention[:, :, 0, :] = 0
        
        # Output
        output = matmul(attention, V)
        output = reshape_to_original(output)
        
        return self.W_out(output)


# ==============================================================================
# Core Module 3: Advance Organizer Module
# ==============================================================================
class AdvanceOrganizer:
    """
    Advance Organizer Module - Based on Ausubel's Advance Organizer Theory
    
    Formula:
        alpha_{t,i} = softmax(h_i^T · W_a · e_t^q / sqrt(d))  # Task-oriented biased attention
        a_t = sum(alpha_{t,i} · h_i)                          # Prior knowledge aggregation
        s_t^short = sigma(W_h · [e_t^q; a_t] + b_h)           # Short-term cognitive readiness state
    
    Function:
        1. Activate relevant historical knowledge for current question
        2. Generate short-term cognitive readiness state
    """
    
    def __init__(self, d_model, dropout):
        self.W_a = Linear(d_model, d_model, bias=False)  # Attention projection
        self.W_h = Linear(2 * d_model, d_model)          # Fusion layer
        self.layer_norm = LayerNorm(d_model)
    
    def forward(self, history_states, current_task_embed):
        """
        Input:
            history_states: Historical knowledge states h_{1:t-1} [batch, seq, d_model]
            current_task_embed: Current question embedding e_t^q [batch, seq, d_model]
        
        Output:
            short_term_state: Short-term cognitive readiness state [batch, seq, d_model]
            attention_weights: Attention weights (for visualization)
        """
        batch, seq, d_model = history_states.shape
        
        # Build causal mask (can only see the past)
        causal_mask = upper_triangular(seq, seq, diagonal=1)
        
        # Compute task-oriented biased attention
        # alpha_{t,i} = softmax(h_i^T · W_a · e_t^q)
        projected_history = self.W_a(history_states)
        
        # Attention scores: [batch, seq, seq]
        attn_scores = bmm(projected_history, current_task_embed.T)
        attn_scores = attn_scores.T  # Transpose so each row corresponds to a time step
        
        # Apply causal mask
        attn_scores = masked_fill(attn_scores, causal_mask, -inf)
        
        # Softmax normalization
        attn_weights = softmax(attn_scores / sqrt(d_model), dim=-1)
        attn_weights = dropout(attn_weights)
        
        # Handle first position (no history)
        attn_weights = masked_fill(attn_weights, is_nan(attn_weights), 0)
        
        # Prior knowledge aggregation: a_t = sum(alpha_{t,i} · h_i)
        prior_knowledge = bmm(attn_weights, history_states)
        
        # Short-term cognitive readiness state: s_t^short = sigma(W_h · [e_t^q; a_t] + b_h)
        concat_input = concat([current_task_embed, prior_knowledge], dim=-1)
        short_term_state = sigmoid(self.W_h(concat_input))
        
        short_term_state = self.layer_norm(short_term_state)
        
        return short_term_state, attn_weights


# ==============================================================================
# Core Module 4: Neural Memory Module
# ==============================================================================
class NeuralMemory:
    """
    Neural Memory Module - Gated Update Mechanism
    
    Formula:
        beta_t = sigma(w_beta^T · s_t^short + b_beta)           # Gating coefficient
        M^curr = W_proj · s_t^short                              # Candidate memory
        M^(t) = beta_t * M^(t-1) + (1-beta_t) * M^curr           # Gated update
    
    Function:
        1. Dynamically adjust the ratio of knowledge retention and update
        2. Implement selective memory mechanism
    """
    
    def __init__(self, d_model, mem_slots):
        self.mem_slots = mem_slots
        
        # Initial memory state M^(0)
        self.memory_init = Parameter(randn(mem_slots, d_model))
        
        # Gating parameters
        self.gate_linear = Linear(d_model, mem_slots)
        
        # Input projection
        self.input_projection = Linear(d_model, mem_slots * d_model)
    
    def forward(self, short_term_state):
        """
        Input:
            short_term_state: Short-term cognitive state [batch, seq, d_model]
        
        Output:
            memory: Updated neural memory [batch, mem_slots, d_model]
        """
        batch, seq, d_model = short_term_state.shape
        
        # Initialize memory
        memory = self.memory_init.expand(batch, -1, -1)  # [batch, mem_slots, d_model]
        
        # Update step by step
        for t in range(seq):
            current_input = short_term_state[:, t, :]  # [batch, d_model]
            
            # Compute gating coefficient: beta_t = sigma(w_beta^T · s_t^short + b_beta)
            gate = sigmoid(self.gate_linear(current_input))  # [batch, mem_slots]
            gate = gate.unsqueeze(-1)  # [batch, mem_slots, 1]
            
            # Candidate memory: M^curr = W_proj · s_t^short
            candidate = self.input_projection(current_input)
            candidate = reshape(candidate, [batch, self.mem_slots, d_model])
            
            # Gated update: M^(t) = beta_t * M^(t-1) + (1-beta_t) * M^curr
            memory = gate * memory + (1 - gate) * candidate
        
        return memory


# ==============================================================================
# Core Module 5: Persistent Memory Module
# ==============================================================================
class PersistentMemory:
    """
    Persistent Memory Module - Long-term Knowledge Consolidation
    
    Formula:
        M_bar^(t) = W_agg · M^(t)                           # Neural memory aggregation
        P^(t) = W_1 · P^(t-1) + W_2 · M_bar^(t) + b_p       # Persistent memory update
    
    Function:
        1. Aggregate neural memory information
        2. Maintain long-term stable knowledge state
    """
    
    def __init__(self, d_model, persistent_slots, neural_mem_slots):
        # Initial persistent memory P^(0)
        self.persistent_init = Parameter(randn(persistent_slots, d_model))
        
        # Update parameters
        self.W1 = Linear(d_model, d_model, bias=False)  # Transform for P^(t-1)
        self.W2 = Linear(d_model, d_model, bias=False)  # Transform for M_bar
        self.bias = Parameter(zeros(persistent_slots, d_model))
        
        # Memory aggregation layer
        self.memory_aggregation = Linear(neural_mem_slots, persistent_slots)
    
    def forward(self, neural_memory):
        """
        Input:
            neural_memory: Neural memory [batch, mem_slots, d_model]
        
        Output:
            persistent: Persistent memory [batch, persistent_slots, d_model]
        """
        batch = neural_memory.shape[0]
        
        # Initialize persistent memory
        persistent = self.persistent_init.expand(batch, -1, -1)
        
        # Aggregate neural memory: M_bar = W_agg · M
        # [batch, d_model, mem_slots] -> [batch, d_model, persistent_slots]
        neural_memory_t = neural_memory.transpose(1, 2)
        aggregated = self.memory_aggregation(neural_memory_t)
        aggregated = aggregated.transpose(1, 2)  # [batch, persistent_slots, d_model]
        
        # Persistent memory update: P^(t) = W_1 · P^(t-1) + W_2 · M_bar^(t) + b_p
        persistent = self.W1(persistent) + self.W2(aggregated) + self.bias
        
        return persistent


# ==============================================================================
# Core Module 6: Memory Fusion Module
# ==============================================================================
class MemoryFusion:
    """
    Multi-level Memory Fusion Module
    
    Formula:
        Z^in = [S^short; M^(t); P^(t)]        # Memory concatenation
        Z^out = TransformerEncoder(Z^in)       # Transformer fusion
        C = Z^out[:T]                          # Cognitive state extraction
    
    Function:
        1. Fuse short-term, neural, and persistent memories
        2. Generate final cognitive state for prediction
    """
    
    def __init__(self, d_model, n_heads, dropout, fusion_layers=1):
        self.fusion_transformer = TransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            num_layers=fusion_layers
        )
    
    def forward(self, short_term_memory, neural_memory, persistent_memory):
        """
        Input:
            short_term_memory: Short-term memory [batch, seq, d_model]
            neural_memory: Neural memory [batch, mem_slots, d_model]
            persistent_memory: Persistent memory [batch, persistent_slots, d_model]
        
        Output:
            cognitive_state: Fused cognitive state [batch, seq, d_model]
        """
        batch, seq, d_model = short_term_memory.shape
        
        # Concatenate three types of memory: Z^in = [S^short; M^(t); P^(t)]
        fusion_input = concat([short_term_memory, neural_memory, persistent_memory], dim=1)
        # shape: [batch, seq + mem_slots + persistent_slots, d_model]
        
        # Transformer fusion
        fusion_output = self.fusion_transformer(fusion_input)
        
        # Extract cognitive state: C = Z^out[:T]
        cognitive_state = fusion_output[:, :seq, :]
        
        return cognitive_state


# ==============================================================================
# Main Model: TIAKT
# ==============================================================================
class TIAKT:
    """
    TIAKT: Tri-level Interactive Attention Knowledge Tracing
    
    Overall Flow:
        1. Embedding Layer: Generate question and answer embeddings
        2. Short-term Memory Encoding: Capture recent learning patterns
        3. Advance Organizer: Activate relevant prior knowledge
        4. Neural Memory Update: Dynamic knowledge state update
        5. Persistent Memory Update: Long-term knowledge consolidation
        6. Memory Fusion: Multi-level memory integration
        7. Knowledge Prediction: Predict probability of correct answer
    """
    
    def __init__(self, n_question, n_pid, d_model, n_blocks, dropout,
                 n_heads=8, d_ff=1024, mem_slots=8, persistent_slots=4):
        
        # 1. Embedding Layer
        self.embedding = EmbeddingLayer(n_question, n_pid, d_model)
        
        # 2. Short-term Memory Encoder
        self.short_term_encoder = ShortTermEncoder(d_model, n_heads, n_blocks, d_ff, dropout)
        
        # 3. Advance Organizer
        self.advance_organizer = AdvanceOrganizer(d_model, dropout)
        
        # 4. Neural Memory
        self.neural_memory = NeuralMemory(d_model, mem_slots)
        
        # 5. Persistent Memory
        self.persistent_memory = PersistentMemory(d_model, persistent_slots, mem_slots)
        
        # 6. Memory Fusion
        self.memory_fusion = MemoryFusion(d_model, n_heads, dropout)
        
        # 7. Prediction Output Layer
        self.output_layer = Sequential(
            Linear(d_model + d_model, 512),
            ReLU(),
            Dropout(dropout),
            Linear(512, 256),
            ReLU(),
            Dropout(dropout),
            Linear(256, 1)
        )
    
    def forward(self, q_data, qa_data, target, pid_data=None):
        """
        Forward Pass
        
        Input:
            q_data: Question ID sequence [batch, seq]
            qa_data: Answer sequence [batch, seq]
            target: Target labels [batch, seq]
            pid_data: Problem difficulty ID [batch, seq] (optional)
        
        Output:
            loss: Loss value
            predictions: Predicted probabilities
            num_valid: Number of valid predictions
        """
        # ========== Step 1: Embedding Layer ==========
        q_embed, qa_embed, reg_loss = self.embedding(q_data, qa_data, pid_data)
        
        # ========== Step 2: Short-term Memory Encoding ==========
        short_term_output = self.short_term_encoder(q_embed, qa_embed)
        
        # ========== Step 3: Advance Organizer ==========
        short_term_state, _ = self.advance_organizer(short_term_output, q_embed)
        
        # ========== Step 4: Neural Memory Update ==========
        neural_mem = self.neural_memory(short_term_state)
        
        # ========== Step 5: Persistent Memory Update ==========
        persistent_mem = self.persistent_memory(neural_mem)
        
        # ========== Step 6: Memory Fusion ==========
        cognitive_state = self.memory_fusion(short_term_state, neural_mem, persistent_mem)
        
        # ========== Step 7: Knowledge Prediction ==========
        # Concatenate cognitive state and question embedding: [C_t; e_t^q]
        concat_input = concat([cognitive_state, q_embed], dim=-1)
        logits = self.output_layer(concat_input)
        
        # Compute loss
        predictions = logits.reshape(-1)
        labels = target.reshape(-1)
        
        # Filter valid predictions (label > -0.9)
        mask = labels > -0.9
        masked_labels = labels[mask]
        masked_preds = predictions[mask]
        
        # BCE loss
        loss = BCEWithLogitsLoss(masked_preds, masked_labels)
        
        return loss.sum() + reg_loss, sigmoid(predictions), mask.sum()


# ==============================================================================
# Training Procedure Pseudocode
# ==============================================================================
def train_tiakt(model, train_data, valid_data, params):
    """
    Training Procedure
    
    Parameters:
        model: TIAKT model
        train_data: Training data (q_data, qa_data, pid_data, ms_data)
        valid_data: Validation data
        params: Training parameters
    """
    optimizer = Adam(model.parameters(), lr=params.lr)
    best_valid_auc = 0
    
    for epoch in range(params.max_epochs):
        model.train()
        
        # Batch training
        for batch in DataLoader(train_data, batch_size=params.batch_size, shuffle=True):
            q_batch, qa_batch, pid_batch, target = batch
            
            # Forward pass
            loss, predictions, num_valid = model(q_batch, qa_batch, target, pid_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (optional)
            if params.maxgradnorm > 0:
                clip_grad_norm_(model.parameters(), params.maxgradnorm)
            
            optimizer.step()
        
        # Validation
        model.eval()
        valid_auc = evaluate(model, valid_data)
        
        # Save best model
        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            save_model(model, "best_model.pt")
        
        # Early stopping
        if epoch - best_epoch > 10:
            break
    
    return best_valid_auc


# ==============================================================================
# Evaluation Metrics
# ==============================================================================
def evaluate(model, test_data):
    """
    Evaluate Model Performance
    
    Metrics:
        - AUC: Area Under ROC Curve
        - Accuracy: Prediction accuracy
        - F1: F1 score
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with no_grad():
        for batch in DataLoader(test_data):
            q_batch, qa_batch, pid_batch, target = batch
            
            _, predictions, _ = model(q_batch, qa_batch, target, pid_batch)
            
            # Collect predictions and labels
            mask = target.reshape(-1) > -0.9
            all_preds.extend(predictions[mask].cpu().numpy())
            all_labels.extend(target.reshape(-1)[mask].cpu().numpy())
    
    # Compute metrics
    auc = roc_auc_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
    f1 = f1_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
    
    return auc, acc, f1
