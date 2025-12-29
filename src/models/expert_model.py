"""
Expert Model Implementation

Implements decoder-only transformer architecture for domain-specialized
expert models. Each expert handles behavioral modeling for a specific domain.
"""

import math
from typing import Optional, Tuple, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    LayerNorm，LLM
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        """
           Args:
           dim:
           eps:
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: Tensor) -> Tensor:
        """计算RMS归一化"""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        """前向传播"""
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE)
    用于在自注意力中编码位置信息，支持更长的序列外推
    """

    def __init__(self, dim: int, max_seq_length: int = 2048, base: int = 10000):
        """
        Args:
            dim: 嵌入维度
            max_seq_length: 最大序列长度
            base: 频率基数
        """
        super().__init__()
        self.dim = dim
        self.max_seq_length = max_seq_length
        self.base = base

        # 预计算频率
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # 预计算位置编码
        self._build_cache(max_seq_length)

    def _build_cache(self, seq_length: int):
        """构建位置编码缓存"""
        t = torch.arange(seq_length, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, x: Tensor, seq_length: int) -> Tuple[Tensor, Tensor]:
        """
        前向传播

        Args:
            x: 输入张量
            seq_length: 序列长度

        Returns:
            cos, sin: 位置编码的余弦和正弦部分
        """
        if seq_length > self.max_seq_length:
            self._build_cache(seq_length)
        return self.cos_cached[:seq_length], self.sin_cached[:seq_length]


def rotate_half(x: Tensor) -> Tensor:
    """将输入张量的后半部分旋转到前半部分"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> Tuple[Tensor, Tensor]:
    """
    应用旋转位置编码

    Args:
        q: Query张量 [batch, heads, seq, dim]
        k: Key张量 [batch, heads, seq, dim]
        cos: 余弦位置编码
        sin: 正弦位置编码

    Returns:
        应用位置编码后的q, k
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, dim]
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    """
    多头自注意力机制

    支持:
        - Rotary Positional Embedding
        - KV Cache (用于推理加速)
        - Flash Attention (可选)
    """

    def __init__(
            self,
            hidden_dim: int,
            num_heads: int,
            dropout: float = 0.1,
            max_seq_length: int = 2048,
    ):
        """
        Args:
            hidden_dim: 隐藏层维度
            num_heads: 注意力头数
            dropout: Dropout率
            max_seq_length: 最大序列长度
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim必须能被num_heads整除"

        # QKV投影
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # 位置编码
        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_length)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
            self,
            x: Tensor,
            attention_mask: Optional[Tensor] = None,
            past_kv: Optional[Tuple[Tensor, Tensor]] = None,
            use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """
        前向传播

        Args:
            x: 输入张量 [batch, seq, hidden]
            attention_mask: 注意力掩码
            past_kv: 缓存的KV对 (用于推理)
            use_cache: 是否返回KV缓存

        Returns:
            output: 输出张量
            present_kv: 当前KV缓存 (如果use_cache=True)
        """
        batch_size, seq_length, _ = x.shape

        # 计算QKV
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 重塑为多头格式 [batch, heads, seq, head_dim]
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # 应用RoPE
        cos, sin = self.rope(x, seq_length)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # 处理KV缓存
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present_kv = (k, v) if use_cache else None

        # 计算注意力分数
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # 应用因果掩码
        kv_length = k.shape[2]
        causal_mask = torch.triu(
            torch.ones(seq_length, kv_length, device=x.device, dtype=torch.bool),
            diagonal=kv_length - seq_length + 1
        )
        attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))

        # 应用额外的注意力掩码
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax和Dropout
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = self.dropout(attn_weights)

        # 计算输出
        attn_output = torch.matmul(attn_weights, v)

        # 重塑回原始形状
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_length, self.hidden_dim)

        # 输出投影
        output = self.o_proj(attn_output)

        return output, present_kv


class FeedForward(nn.Module):
    """
    前馈网络 (SwiGLU变体)

    SwiGLU激活函数在现代LLM中表现更好
    """

    def __init__(
            self,
            hidden_dim: int,
            ffn_dim: int,
            dropout: float = 0.1,
    ):
        """
        Args:
            hidden_dim: 隐藏层维度
            ffn_dim: 前馈网络维度
            dropout: Dropout率
        """
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """前向传播 - SwiGLU激活"""
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.dropout(self.down_proj(gate * up))


class TransformerBlock(nn.Module):
    """
    Transformer解码器块

    结构: Pre-LayerNorm + Attention + FFN (残差连接)
    """

    def __init__(
            self,
            hidden_dim: int,
            num_heads: int,
            ffn_dim: int,
            dropout: float = 0.1,
            max_seq_length: int = 2048,
    ):
        """
        Args:
            hidden_dim: 隐藏层维度
            num_heads: 注意力头数
            ffn_dim: 前馈网络维度
            dropout: Dropout率
            max_seq_length: 最大序列长度
        """
        super().__init__()

        self.attention = MultiHeadAttention(
            hidden_dim, num_heads, dropout, max_seq_length
        )
        self.feed_forward = FeedForward(hidden_dim, ffn_dim, dropout)

        self.attention_norm = RMSNorm(hidden_dim)
        self.ffn_norm = RMSNorm(hidden_dim)

    def forward(
            self,
            x: Tensor,
            attention_mask: Optional[Tensor] = None,
            past_kv: Optional[Tuple[Tensor, Tensor]] = None,
            use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """
        前向传播

        Args:
            x: 输入张量
            attention_mask: 注意力掩码
            past_kv: KV缓存
            use_cache: 是否使用缓存

        Returns:
            output: 输出张量
            present_kv: KV缓存
        """
        # Self-Attention with residual
        residual = x
        x = self.attention_norm(x)
        attn_out, present_kv = self.attention(x, attention_mask, past_kv, use_cache)
        x = residual + attn_out

        # Feed Forward with residual
        residual = x
        x = self.ffn_norm(x)
        x = residual + self.feed_forward(x)

        return x, present_kv


class ExpertModel(nn.Module):
    """
    专家模型 - Decoder-only Transformer

    论文架构:
        - 40层深度
        - 隐藏维度5120
        - 注意力头数32
        - 前馈网络维度20480
        - 词汇表大小50000

    每个专家专注于特定领域的语言建模任务
    """

    def __init__(
            self,
            expert_id: int,
            vocab_size: int = 50000,
            hidden_dim: int = 5120,
            num_layers: int = 40,
            num_heads: int = 32,
            ffn_dim: int = 20480,
            max_seq_length: int = 2048,
            dropout: float = 0.1,
            tie_word_embeddings: bool = True,
    ):
        """
        Args:
            expert_id: 专家ID (0-7)
            vocab_size: 词汇表大小
            hidden_dim: 隐藏层维度
            num_layers: Transformer层数
            num_heads: 注意力头数
            ffn_dim: 前馈网络维度
            max_seq_length: 最大序列长度
            dropout: Dropout率
            tie_word_embeddings: 是否共享输入输出嵌入
        """
        super().__init__()

        self.expert_id = expert_id
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_seq_length = max_seq_length

        # 词嵌入层
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Transformer层
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                max_seq_length=max_seq_length,
            )
            for _ in range(num_layers)
        ])

        # 最终层归一化
        self.final_norm = RMSNorm(hidden_dim)

        # 输出投影 (语言模型头)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

        # 权重绑定
        if tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self) -> int:
        """获取模型参数数量"""
        return sum(p.numel() for p in self.parameters())

    def forward(
            self,
            input_ids: Tensor,
            attention_mask: Optional[Tensor] = None,
            past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None,
            use_cache: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = True,
    ) -> Dict[str, Any]:
        """
        前向传播

        Args:
            input_ids: 输入token ID [batch, seq]
            attention_mask: 注意力掩码
            past_key_values: KV缓存列表
            use_cache: 是否使用缓存
            output_hidden_states: 是否输出所有隐藏状态
            return_dict: 是否返回字典格式

        Returns:
            包含以下键的字典:
                - logits: 输出logits [batch, seq, vocab]
                - hidden_states: 隐藏状态 (如果output_hidden_states=True)
                - past_key_values: KV缓存 (如果use_cache=True)
        """
        batch_size, seq_length = input_ids.shape

        # Token嵌入
        hidden_states = self.token_embedding(input_ids)
        hidden_states = self.dropout(hidden_states)

        # 处理attention_mask的形状
        # 输入形状: [batch, seq] -> 需要转换为 [batch, 1, 1, seq] 或 [batch, 1, seq, seq]
        if attention_mask is not None:
            # 确保是浮点类型
            if attention_mask.dtype == torch.bool:
                attention_mask = attention_mask.float()
            elif attention_mask.dtype == torch.long:
                attention_mask = attention_mask.float()

            # 将0/1掩码转换为attention掩码 (0 -> -inf, 1 -> 0)
            # attention_mask: [batch, seq] -> [batch, 1, 1, seq]
            if attention_mask.dim() == 2:
                # [batch, seq] -> [batch, 1, 1, seq]
                attention_mask = attention_mask[:, None, None, :]
                # 转换: 1 -> 0 (attend), 0 -> -inf (ignore)
                attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min
            elif attention_mask.dim() == 3:
                # [batch, 1, seq] -> [batch, 1, 1, seq]
                attention_mask = attention_mask[:, None, :, :]
                attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min
            # 如果已经是4D，假设已经处理过了

        # 存储所有隐藏状态
        all_hidden_states = [hidden_states] if output_hidden_states else None

        # 处理KV缓存
        if past_key_values is None:
            past_key_values = [None] * self.num_layers

        present_key_values = [] if use_cache else None

        # 通过所有Transformer层
        for i, layer in enumerate(self.layers):
            hidden_states, present_kv = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_kv=past_key_values[i],
                use_cache=use_cache,
            )

            if use_cache:
                present_key_values.append(present_kv)

            if output_hidden_states:
                all_hidden_states.append(hidden_states)

        # 最终层归一化
        hidden_states = self.final_norm(hidden_states)

        # 计算logits
        logits = self.lm_head(hidden_states)

        if return_dict:
            return {
                "logits": logits,
                "hidden_states": all_hidden_states,
                "past_key_values": present_key_values,
                "last_hidden_state": hidden_states,
            }
        else:
            return logits

    def compute_loss(
            self,
            input_ids: Tensor,
            labels: Optional[Tensor] = None,
            attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        计算语言模型损失

        Args:
            input_ids: 输入token ID
            labels: 目标标签 (默认为shifted input_ids)
            attention_mask: 注意力掩码

        Returns:
            loss: 交叉熵损失
            logits: 模型输出logits
        """
        outputs = self.forward(input_ids, attention_mask)
        logits = outputs["logits"]

        # 如果没有提供标签，使用shifted input_ids
        if labels is None:
            labels = input_ids

        # Shift for causal LM loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # 计算交叉熵损失
        loss_fct = nn.CrossEntropyLoss(reduction='mean')
        loss = loss_fct(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1)
        )

        return loss, logits

    def get_probabilities(
            self,
            input_ids: Tensor,
            attention_mask: Optional[Tensor] = None,
            temperature: float = 1.0,
    ) -> Tensor:
        """
        获取输出概率分布

        Args:
            input_ids: 输入token ID
            attention_mask: 注意力掩码
            temperature: 温度参数

        Returns:
            概率分布 [batch, seq, vocab]
        """
        outputs = self.forward(input_ids, attention_mask)
        logits = outputs["logits"]

        # 应用温度
        if temperature != 1.0:
            logits = logits / temperature

        return F.softmax(logits, dim=-1)

    def get_perplexity(
            self,
            input_ids: Tensor,
            attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        计算困惑度

        Args:
            input_ids: 输入token ID
            attention_mask: 注意力掩码

        Returns:
            困惑度值
        """
        loss, _ = self.compute_loss(input_ids, attention_mask=attention_mask)
        return torch.exp(loss)


class ExpertModelSmall(ExpertModel):
    """
    小型专家模型 (用于实验和测试)

    缩小版架构:
        - 6层深度
        - 隐藏维度768
        - 注意力头数12
        - 前馈网络维度3072
    """

    def __init__(
            self,
            expert_id: int,
            vocab_size: int = 50000,
            hidden_dim: int = 768,
            num_layers: int = 6,
            num_heads: int = 12,
            ffn_dim: int = 3072,
            max_seq_length: int = 512,
            dropout: float = 0.1,
    ):
        super().__init__(
            expert_id=expert_id,
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            max_seq_length=max_seq_length,
            dropout=dropout,
        )


class MultiExpertSystem(nn.Module):
    """
    多专家系统 - 管理所有K个专家模型

    论文中K* = 8个专家，每个专家负责特定领域
    """

    def __init__(
            self,
            num_experts: int = 8,
            vocab_size: int = 50000,
            hidden_dim: int = 768,
            num_layers: int = 6,
            num_heads: int = 12,
            ffn_dim: int = 3072,
            max_seq_length: int = 512,
            dropout: float = 0.1,
            use_small_model: bool = True,
    ):
        """
        Args:
            num_experts: 专家数量 (论文中K* = 8)
            其他参数: 单个专家模型的配置
            use_small_model: 是否使用小型模型
        """
        super().__init__()

        self.num_experts = num_experts
        self.hidden_dim = hidden_dim

        # 创建专家模型
        ExpertClass = ExpertModelSmall if use_small_model else ExpertModel

        self.experts = nn.ModuleList([
            ExpertClass(
                expert_id=i,
                vocab_size=vocab_size,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                max_seq_length=max_seq_length,
                dropout=dropout,
            )
            for i in range(num_experts)
        ])

        # 专家领域映射
        self.domain_mapping = {}

    def set_domain_mapping(self, mapping: Dict[int, str]):
        """设置专家-领域映射"""
        self.domain_mapping = mapping

    def get_expert(self, expert_id: int) -> ExpertModel:
        """获取指定ID的专家模型"""
        return self.experts[expert_id]

    def forward_single_expert(
            self,
            expert_id: int,
            input_ids: Tensor,
            attention_mask: Optional[Tensor] = None,
            **kwargs,
    ) -> Dict[str, Any]:
        """使用单个专家进行前向传播"""
        return self.experts[expert_id](input_ids, attention_mask, **kwargs)

    def forward_all_experts(
            self,
            input_ids: Tensor,
            attention_mask: Optional[Tensor] = None,
            **kwargs,
    ) -> List[Dict[str, Any]]:
        """使用所有专家进行前向传播"""
        outputs = []
        for expert in self.experts:
            output = expert(input_ids, attention_mask, **kwargs)
            outputs.append(output)
        return outputs

    def get_all_probabilities(
            self,
            input_ids: Tensor,
            attention_mask: Optional[Tensor] = None,
            temperature: float = 1.0,
    ) -> List[Tensor]:
        """获取所有专家的概率分布"""
        probs = []
        for expert in self.experts:
            prob = expert.get_probabilities(input_ids, attention_mask, temperature)
            probs.append(prob)
        return probs

    def get_total_params(self) -> int:
        """获取总参数数量"""
        return sum(expert.get_num_params() for expert in self.experts)


if __name__ == "__main__":
    # 测试代码
    print("Testing Expert Model...")

    # 创建小型专家模型进行测试
    expert = ExpertModelSmall(expert_id=0)
    print(f"Expert parameters: {expert.get_num_params():,}")

    # 测试前向传播
    batch_size, seq_length = 2, 128
    input_ids = torch.randint(0, 50000, (batch_size, seq_length))

    outputs = expert(input_ids, output_hidden_states=True)
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Hidden states count: {len(outputs['hidden_states'])}")

    # 测试损失计算
    loss, logits = expert.compute_loss(input_ids)
    print(f"Loss: {loss.item():.4f}")

    # 测试多专家系统
    print("\nTesting Multi-Expert System...")
    multi_expert = MultiExpertSystem(num_experts=8)
    print(f"Total parameters: {multi_expert.get_total_params():,}")

    # 测试所有专家前向传播
    all_outputs = multi_expert.forward_all_experts(input_ids)
    print(f"Number of expert outputs: {len(all_outputs)}")

    print("\nAll tests passed!")