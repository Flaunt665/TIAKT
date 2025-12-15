"""
================================================================================
TIAKT: Tri-level Interactive Attention Knowledge Tracing
三层交互注意力知识追踪 - 伪代码版本

本文件展示TIAKT模型的核心算法机制，为理解论文方法而设计
================================================================================
"""

import numpy as np

# ==============================================================================
# 超参数定义
# ==============================================================================
"""
d_model: 模型隐藏维度 (默认: 128)
n_heads: 注意力头数 (默认: 8)
n_blocks: Transformer块数 (默认: 1)
mem_slots: 神经记忆槽位数 (默认: 8)
persistent_slots: 持久记忆槽位数 (默认: 4)
dropout: Dropout率 (默认: 0.05)
"""


# ==============================================================================
# 核心模块1: 多层次嵌入层 (Multi-level Embedding Layer)
# ==============================================================================
class EmbeddingLayer:
    """
    题目嵌入与难度调制
    
    公式:
        e_t^q = e_t^{q,base} + μ_t * e_t^{q,diff}
        e_t^{qa} = e_t^{qa,base} + μ_t * e_t^{qa,diff}
    
    其中:
        e_t^{q,base}: 基础题目嵌入
        e_t^{q,diff}: 难度差异嵌入  
        μ_t: 题目难度参数
    """
    
    def __init__(self, n_question, n_pid, d_model):
        # 基础嵌入
        self.q_embed = Embedding(n_question + 1, d_model)      # 题目嵌入
        self.qa_embed = Embedding(2, d_model)                  # 答题嵌入 (正确/错误)
        
        # 难度相关嵌入 (当有题目ID时)
        if n_pid > 0:
            self.difficult_param = Embedding(n_pid + 1, 1)      # 难度参数 μ_t
            self.q_embed_diff = Embedding(n_question + 1, d_model)   # 题目难度差异
            self.qa_embed_diff = Embedding(2 * n_question + 1, d_model)  # 答题难度差异
    
    def forward(self, q_data, qa_data, pid_data=None):
        """
        输入:
            q_data: 题目ID序列 [batch, seq_len]
            qa_data: 答题序列 [batch, seq_len]  
            pid_data: 题目难度ID [batch, seq_len] (可选)
        
        输出:
            q_embed_data: 题目嵌入 [batch, seq_len, d_model]
            qa_embed_data: 答题嵌入 [batch, seq_len, d_model]
        """
        # 基础题目嵌入
        q_embed_data = self.q_embed(q_data)
        
        # 答题嵌入 (将答案与题目结合)
        qa_indicator = (qa_data - q_data) // n_question  # 0: 错误, 1: 正确
        qa_embed_data = self.qa_embed(qa_indicator) + q_embed_data
        
        # 难度调制
        if pid_data is not None and self.n_pid > 0:
            q_embed_diff_data = self.q_embed_diff(q_data)
            μ_t = self.difficult_param(pid_data)  # 难度参数
            
            # 难度调制后的嵌入
            q_embed_data = q_embed_data + μ_t * q_embed_diff_data
            
            qa_embed_diff_data = self.qa_embed_diff(qa_data)
            qa_embed_data = qa_embed_data + μ_t * (qa_embed_diff_data + q_embed_diff_data)
            
            # L2正则化损失
            reg_loss = (μ_t ** 2).sum() * l2_weight
        
        return q_embed_data, qa_embed_data, reg_loss


# ==============================================================================
# 核心模块2: 短期记忆编码器 (Short-term Memory Encoder)
# 基于AKT的双路Transformer结构
# ==============================================================================
class ShortTermEncoder:
    """
    短期记忆编码器 - 使用单调注意力机制
    
    结构:
        1. 答题序列自注意力 (blocks_1): 捕获历史答题模式
        2. 题目-答题交叉注意力 (blocks_2): 建立题目与答题关联
    
    关键: 使用指数衰减的单调注意力
    """
    
    def __init__(self, d_model, n_heads, n_blocks, d_ff, dropout):
        # 答题序列自注意力层
        self.blocks_1 = [TransformerLayer(...) for _ in range(n_blocks)]
        # 题目-答题交叉注意力层
        self.blocks_2 = [TransformerLayer(...) for _ in range(n_blocks * 2)]
    
    def forward(self, q_embed_data, qa_embed_data):
        """
        输入:
            q_embed_data: 题目嵌入 [batch, seq, d_model]
            qa_embed_data: 答题嵌入 [batch, seq, d_model]
        
        输出:
            h_t: 短期知识状态 [batch, seq, d_model]
        """
        y = qa_embed_data  # 答题序列
        x = q_embed_data   # 题目序列
        
        # Step 1: 答题序列自注意力
        for block in self.blocks_1:
            y = block(query=y, key=y, values=y, mask=CAUSAL_MASK)
        
        # Step 2: 题目-答题交叉注意力 (交替进行)
        is_self_attn = True
        for block in self.blocks_2:
            if is_self_attn:
                x = block(query=x, key=x, values=x, mask=CAUSAL_MASK, apply_pos=False)
            else:
                x = block(query=x, key=x, values=y, mask=PEEK_MASK, apply_pos=True)
            is_self_attn = not is_self_attn
        
        return x  # 短期知识表示 h_t


class MonotonicMultiHeadAttention:
    """
    单调注意力机制 (来自AKT)
    
    核心公式:
        scores = Q·K^T / √d_k
        
        # 计算位置衰减效应
        position_effect = |i - j|  # 位置距离
        decay = exp(γ * position_effect)  # 指数衰减
        
        # 应用衰减
        scores = scores * decay
        attention = softmax(scores)
        output = attention · V
    """
    
    def forward(self, q, k, v, mask, zero_pad=False):
        # 线性投影
        Q = self.W_q(q)  # [batch, seq, d_model]
        K = self.W_k(k)
        V = self.W_v(v)
        
        # 分头
        Q = reshape_to_heads(Q)  # [batch, n_heads, seq, d_k]
        K = reshape_to_heads(K)
        V = reshape_to_heads(V)
        
        # 注意力分数
        scores = matmul(Q, K.T) / sqrt(d_k)
        
        # === 关键: 单调位置衰减 ===
        seq_len = scores.shape[-1]
        pos_i = arange(seq_len).expand(seq_len, -1)
        pos_j = pos_i.T
        position_distance = abs(pos_i - pos_j)
        
        # 计算衰减效应 (可学习参数γ)
        gamma = -softplus(self.gamma_param)
        decay_effect = exp(gamma * position_distance)
        decay_effect = clamp(decay_effect, min=1e-5, max=1e5)
        
        # 应用衰减
        scores = scores * decay_effect
        
        # 掩码与归一化
        scores = masked_fill(scores, mask == 0, -inf)
        attention = softmax(scores, dim=-1)
        
        # 首位零填充 (防止泄露)
        if zero_pad:
            attention[:, :, 0, :] = 0
        
        # 输出
        output = matmul(attention, V)
        output = reshape_to_original(output)
        
        return self.W_out(output)


# ==============================================================================
# 核心模块3: 先行组织者模块 (Advance Organizer Module)
# ==============================================================================
class AdvanceOrganizer:
    """
    先行组织者模块 - 基于教育心理学的先行组织者理论
    
    公式:
        α_{t,i} = softmax(h_i^T · W_a · e_t^q / √d)  # 任务导向偏置注意力
        a_t = Σ α_{t,i} · h_i                         # 先验知识聚合
        s_t^short = σ(W_h · [e_t^q; a_t] + b_h)       # 短期认知准备状态
    
    作用:
        1. 为当前题目激活相关的历史知识
        2. 生成短期认知准备状态
    """
    
    def __init__(self, d_model, dropout):
        self.W_a = Linear(d_model, d_model, bias=False)  # 注意力投影
        self.W_h = Linear(2 * d_model, d_model)          # 融合层
        self.layer_norm = LayerNorm(d_model)
    
    def forward(self, history_states, current_task_embed):
        """
        输入:
            history_states: 历史知识状态 h_{1:t-1} [batch, seq, d_model]
            current_task_embed: 当前题目嵌入 e_t^q [batch, seq, d_model]
        
        输出:
            short_term_state: 短期认知准备状态 [batch, seq, d_model]
            attention_weights: 注意力权重 (用于可视化)
        """
        batch, seq, d_model = history_states.shape
        
        # 构建因果掩码 (只能看到过去)
        causal_mask = upper_triangular(seq, seq, diagonal=1)
        
        # 计算任务导向的偏置注意力
        # α_{t,i} = softmax(h_i^T · W_a · e_t^q)
        projected_history = self.W_a(history_states)
        
        # 注意力分数: [batch, seq, seq]
        attn_scores = bmm(projected_history, current_task_embed.T)
        attn_scores = attn_scores.T  # 转置使每行对应一个时间步
        
        # 应用因果掩码
        attn_scores = masked_fill(attn_scores, causal_mask, -inf)
        
        # Softmax归一化
        attn_weights = softmax(attn_scores / sqrt(d_model), dim=-1)
        attn_weights = dropout(attn_weights)
        
        # 处理第一个位置 (没有历史)
        attn_weights = masked_fill(attn_weights, is_nan(attn_weights), 0)
        
        # 先验知识聚合: a_t = Σ α_{t,i} · h_i
        prior_knowledge = bmm(attn_weights, history_states)
        
        # 短期认知准备状态: s_t^short = σ(W_h · [e_t^q; a_t] + b_h)
        concat_input = concat([current_task_embed, prior_knowledge], dim=-1)
        short_term_state = sigmoid(self.W_h(concat_input))
        
        short_term_state = self.layer_norm(short_term_state)
        
        return short_term_state, attn_weights


# ==============================================================================
# 核心模块4: 神经记忆模块 (Neural Memory Module)
# ==============================================================================
class NeuralMemory:
    """
    神经记忆模块 - 门控更新机制
    
    公式:
        β_t = σ(w_β^T · s_t^short + b_β)           # 门控系数
        M^curr = W_proj · s_t^short                 # 候选记忆
        M^(t) = β_t ⊙ M^(t-1) + (1-β_t) ⊙ M^curr  # 门控更新
    
    作用:
        1. 动态调整知识保留与更新的比例
        2. 实现选择性记忆机制
    """
    
    def __init__(self, d_model, mem_slots):
        self.mem_slots = mem_slots
        
        # 初始记忆状态 M^(0)
        self.memory_init = Parameter(randn(mem_slots, d_model))
        
        # 门控参数
        self.gate_linear = Linear(d_model, mem_slots)
        
        # 输入投影
        self.input_projection = Linear(d_model, mem_slots * d_model)
    
    def forward(self, short_term_state):
        """
        输入:
            short_term_state: 短期认知状态 [batch, seq, d_model]
        
        输出:
            memory: 更新后的神经记忆 [batch, mem_slots, d_model]
        """
        batch, seq, d_model = short_term_state.shape
        
        # 初始化记忆
        memory = self.memory_init.expand(batch, -1, -1)  # [batch, mem_slots, d_model]
        
        # 逐时间步更新
        for t in range(seq):
            current_input = short_term_state[:, t, :]  # [batch, d_model]
            
            # 计算门控系数: β_t = σ(w_β^T · s_t^short + b_β)
            gate = sigmoid(self.gate_linear(current_input))  # [batch, mem_slots]
            gate = gate.unsqueeze(-1)  # [batch, mem_slots, 1]
            
            # 候选记忆: M^curr = W_proj · s_t^short
            candidate = self.input_projection(current_input)
            candidate = reshape(candidate, [batch, self.mem_slots, d_model])
            
            # 门控更新: M^(t) = β_t ⊙ M^(t-1) + (1-β_t) ⊙ M^curr
            memory = gate * memory + (1 - gate) * candidate
        
        return memory


# ==============================================================================
# 核心模块5: 持久记忆模块 (Persistent Memory Module)
# ==============================================================================
class PersistentMemory:
    """
    持久记忆模块 - 长期知识巩固
    
    公式:
        M̄^(t) = W_agg · M^(t)                           # 神经记忆聚合
        P^(t) = W_1 · P^(t-1) + W_2 · M̄^(t) + b_p      # 持久记忆更新
    
    作用:
        1. 聚合神经记忆信息
        2. 维护长期稳定的知识状态
    """
    
    def __init__(self, d_model, persistent_slots, neural_mem_slots):
        # 初始持久记忆 P^(0)
        self.persistent_init = Parameter(randn(persistent_slots, d_model))
        
        # 更新参数
        self.W1 = Linear(d_model, d_model, bias=False)  # 对P^(t-1)的变换
        self.W2 = Linear(d_model, d_model, bias=False)  # 对M̄的变换
        self.bias = Parameter(zeros(persistent_slots, d_model))
        
        # 记忆聚合层
        self.memory_aggregation = Linear(neural_mem_slots, persistent_slots)
    
    def forward(self, neural_memory):
        """
        输入:
            neural_memory: 神经记忆 [batch, mem_slots, d_model]
        
        输出:
            persistent: 持久记忆 [batch, persistent_slots, d_model]
        """
        batch = neural_memory.shape[0]
        
        # 初始化持久记忆
        persistent = self.persistent_init.expand(batch, -1, -1)
        
        # 聚合神经记忆: M̄ = W_agg · M
        # [batch, d_model, mem_slots] -> [batch, d_model, persistent_slots]
        neural_memory_t = neural_memory.transpose(1, 2)
        aggregated = self.memory_aggregation(neural_memory_t)
        aggregated = aggregated.transpose(1, 2)  # [batch, persistent_slots, d_model]
        
        # 持久记忆更新: P^(t) = W_1 · P^(t-1) + W_2 · M̄^(t) + b_p
        persistent = self.W1(persistent) + self.W2(aggregated) + self.bias
        
        return persistent


# ==============================================================================
# 核心模块6: 记忆融合模块 (Memory Fusion Module)
# ==============================================================================
class MemoryFusion:
    """
    多层次记忆融合模块
    
    公式:
        Z^in = [S^short; M^(t); P^(t)]        # 记忆拼接
        Z^out = TransformerEncoder(Z^in)       # Transformer融合
        C = Z^out[:T]                          # 认知状态提取
    
    作用:
        1. 融合短期、神经、持久三种记忆
        2. 生成最终认知状态用于预测
    """
    
    def __init__(self, d_model, n_heads, dropout, fusion_layers=1):
        self.fusion_transformer = TransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            num_layers=fusion_layers
        )
    
    def forward(self, short_term_memory, neural_memory, persistent_memory):
        """
        输入:
            short_term_memory: 短期记忆 [batch, seq, d_model]
            neural_memory: 神经记忆 [batch, mem_slots, d_model]
            persistent_memory: 持久记忆 [batch, persistent_slots, d_model]
        
        输出:
            cognitive_state: 融合认知状态 [batch, seq, d_model]
        """
        batch, seq, d_model = short_term_memory.shape
        
        # 拼接三种记忆: Z^in = [S^short; M^(t); P^(t)]
        fusion_input = concat([short_term_memory, neural_memory, persistent_memory], dim=1)
        # shape: [batch, seq + mem_slots + persistent_slots, d_model]
        
        # Transformer融合
        fusion_output = self.fusion_transformer(fusion_input)
        
        # 提取认知状态: C = Z^out[:T]
        cognitive_state = fusion_output[:, :seq, :]
        
        return cognitive_state


# ==============================================================================
# 主模型: TIAKT
# ==============================================================================
class TIAKT:
    """
    TIAKT: Tri-level Interactive Attention Knowledge Tracing
    三层交互注意力知识追踪模型
    
    整体流程:
        1. 嵌入层: 生成题目和答题嵌入
        2. 短期记忆编码: 捕获近期学习模式
        3. 先行组织者: 激活相关先验知识
        4. 神经记忆更新: 动态知识状态更新
        5. 持久记忆更新: 长期知识巩固
        6. 记忆融合: 多层次记忆整合
        7. 知识预测: 预测下一题正确率
    """
    
    def __init__(self, n_question, n_pid, d_model, n_blocks, dropout,
                 n_heads=8, d_ff=1024, mem_slots=8, persistent_slots=4):
        
        # 1. 嵌入层
        self.embedding = EmbeddingLayer(n_question, n_pid, d_model)
        
        # 2. 短期记忆编码器
        self.short_term_encoder = ShortTermEncoder(d_model, n_heads, n_blocks, d_ff, dropout)
        
        # 3. 先行组织者
        self.advance_organizer = AdvanceOrganizer(d_model, dropout)
        
        # 4. 神经记忆
        self.neural_memory = NeuralMemory(d_model, mem_slots)
        
        # 5. 持久记忆
        self.persistent_memory = PersistentMemory(d_model, persistent_slots, mem_slots)
        
        # 6. 记忆融合
        self.memory_fusion = MemoryFusion(d_model, n_heads, dropout)
        
        # 7. 预测输出层
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
        前向传播
        
        输入:
            q_data: 题目ID序列 [batch, seq]
            qa_data: 答题序列 [batch, seq]
            target: 目标标签 [batch, seq]
            pid_data: 题目难度ID [batch, seq] (可选)
        
        输出:
            loss: 损失值
            predictions: 预测概率
            num_valid: 有效预测数量
        """
        # ========== Step 1: 嵌入层 ==========
        q_embed, qa_embed, reg_loss = self.embedding(q_data, qa_data, pid_data)
        
        # ========== Step 2: 短期记忆编码 ==========
        short_term_output = self.short_term_encoder(q_embed, qa_embed)
        
        # ========== Step 3: 先行组织者 ==========
        short_term_state, _ = self.advance_organizer(short_term_output, q_embed)
        
        # ========== Step 4: 神经记忆更新 ==========
        neural_mem = self.neural_memory(short_term_state)
        
        # ========== Step 5: 持久记忆更新 ==========
        persistent_mem = self.persistent_memory(neural_mem)
        
        # ========== Step 6: 记忆融合 ==========
        cognitive_state = self.memory_fusion(short_term_state, neural_mem, persistent_mem)
        
        # ========== Step 7: 知识预测 ==========
        # 拼接认知状态和题目嵌入: [C_t; e_t^q]
        concat_input = concat([cognitive_state, q_embed], dim=-1)
        logits = self.output_layer(concat_input)
        
        # 计算损失
        predictions = logits.reshape(-1)
        labels = target.reshape(-1)
        
        # 过滤有效预测 (label > -0.9)
        mask = labels > -0.9
        masked_labels = labels[mask]
        masked_preds = predictions[mask]
        
        # BCE损失
        loss = BCEWithLogitsLoss(masked_preds, masked_labels)
        
        return loss.sum() + reg_loss, sigmoid(predictions), mask.sum()


# ==============================================================================
# 训练流程伪代码
# ==============================================================================
def train_tiakt(model, train_data, valid_data, params):
    """
    训练流程
    
    参数:
        model: TIAKT模型
        train_data: 训练数据 (q_data, qa_data, pid_data, ms_data)
        valid_data: 验证数据
        params: 训练参数
    """
    optimizer = Adam(model.parameters(), lr=params.lr)
    best_valid_auc = 0
    
    for epoch in range(params.max_epochs):
        model.train()
        
        # 批次训练
        for batch in DataLoader(train_data, batch_size=params.batch_size, shuffle=True):
            q_batch, qa_batch, pid_batch, target = batch
            
            # 前向传播
            loss, predictions, num_valid = model(q_batch, qa_batch, target, pid_batch)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪 (可选)
            if params.maxgradnorm > 0:
                clip_grad_norm_(model.parameters(), params.maxgradnorm)
            
            optimizer.step()
        
        # 验证
        model.eval()
        valid_auc = evaluate(model, valid_data)
        
        # 保存最佳模型
        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            save_model(model, "best_model.pt")
        
        # 早停
        if epoch - best_epoch > 10:
            break
    
    return best_valid_auc


# ==============================================================================
# 评估指标
# ==============================================================================
def evaluate(model, test_data):
    """
    评估模型性能
    
    指标:
        - AUC: ROC曲线下面积
        - Accuracy: 准确率
        - F1: F1分数
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with no_grad():
        for batch in DataLoader(test_data):
            q_batch, qa_batch, pid_batch, target = batch
            
            _, predictions, _ = model(q_batch, qa_batch, target, pid_batch)
            
            # 收集预测和标签
            mask = target.reshape(-1) > -0.9
            all_preds.extend(predictions[mask].cpu().numpy())
            all_labels.extend(target.reshape(-1)[mask].cpu().numpy())
    
    # 计算指标
    auc = roc_auc_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
    f1 = f1_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
    
    return auc, acc, f1
