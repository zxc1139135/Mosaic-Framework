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
 """RMS"""
 return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

 def forward(self, x: Tensor) -> Tensor:
 """"""
 output = self._norm(x.float()).type_as(x)
 return output * self.weight

class RotaryPositionalEmbedding(nn.Module):
 """
 Rotary Positional Embedding (RoPE)
 ，
 """

 def __init__(self, dim: int, max_seq_length: int = 2048, base: int = 10000):
 """
 Args:
 dim: 
 max_seq_length: 
 base: 
 """
 super().__init__()
 self.dim = dim
 self.max_seq_length = max_seq_length
 self.base = base

 # 
 inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
 self.register_buffer("inv_freq", inv_freq)

 # 
 self._build_cache(max_seq_length)

 def _build_cache(self, seq_length: int):
 """"""
 t = torch.arange(seq_length, device=self.inv_freq.device)
 freqs = torch.outer(t, self.inv_freq)
 emb = torch.cat([freqs, freqs], dim=-1)
 self.register_buffer("cos_cached", emb.cos())
 self.register_buffer("sin_cached", emb.sin())

 def forward(self, x: Tensor, seq_length: int) -> Tuple[Tensor, Tensor]:
 """
 

 Args:
 x: 
 seq_length: 

 Returns:
 cos, sin: 
 """
 if seq_length > self.max_seq_length:
 self._build_cache(seq_length)
 return self.cos_cached[:seq_length], self.sin_cached[:seq_length]

def rotate_half(x: Tensor) -> Tensor:
 """"""
 x1 = x[..., : x.shape[-1] // 2]
 x2 = x[..., x.shape[-1] // 2:]
 return torch.cat([-x2, x1], dim=-1)

def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> Tuple[Tensor, Tensor]:
 """
 

 Args:
 q: Query [batch, heads, seq, dim]
 k: Key [batch, heads, seq, dim]
 cos: 
 sin: 

 Returns:
 q, k
 """
 cos = cos.unsqueeze(0).unsqueeze(0) # [1, 1, seq, dim]
 sin = sin.unsqueeze(0).unsqueeze(0)
 q_embed = (q * cos) + (rotate_half(q) * sin)
 k_embed = (k * cos) + (rotate_half(k) * sin)
 return q_embed, k_embed

class MultiHeadAttention(nn.Module):
 """
 

 :
 - Rotary Positional Embedding
 - KV Cache ()
 - Flash Attention ()
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
 hidden_dim: 
 num_heads: 
 dropout: Dropout
 max_seq_length: 
 """
 super().__init__()
 self.hidden_dim = hidden_dim
 self.num_heads = num_heads
 self.head_dim = hidden_dim // num_heads

 assert hidden_dim % num_heads == 0, "hidden_dimnum_heads"

 # QKV
 self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
 self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
 self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
 self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

 # 
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
 

 Args:
 x: [batch, seq, hidden]
 attention_mask: 
 past_kv: KV ()
 use_cache: KV

 Returns:
 output: 
 present_kv: KV (use_cache=True)
 """
 batch_size, seq_length, _ = x.shape

 # QKV
 q = self.q_proj(x)
 k = self.k_proj(x)
 v = self.v_proj(x)

 # [batch, heads, seq, head_dim]
 q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
 k = k.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
 v = v.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

 # RoPE
 cos, sin = self.rope(x, seq_length)
 q, k = apply_rotary_pos_emb(q, k, cos, sin)

 # KV
 if past_kv is not None:
 past_k, past_v = past_kv
 k = torch.cat([past_k, k], dim=2)
 v = torch.cat([past_v, v], dim=2)

 present_kv = (k, v) if use_cache else None

 # 
 attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

 # 
 kv_length = k.shape[2]
 causal_mask = torch.triu(
 torch.ones(seq_length, kv_length, device=x.device, dtype=torch.bool),
 diagonal=kv_length - seq_length + 1
 )
 attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))

 # 
 if attention_mask is not None:
 attn_weights = attn_weights + attention_mask

 # SoftmaxDropout
 attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
 attn_weights = self.dropout(attn_weights)

 # 
 attn_output = torch.matmul(attn_weights, v)

 # 
 attn_output = attn_output.transpose(1, 2).contiguous()
 attn_output = attn_output.view(batch_size, seq_length, self.hidden_dim)

 # 
 output = self.o_proj(attn_output)

 return output, present_kv

class FeedForward(nn.Module):
 """
 (SwiGLU)

 SwiGLULLM
 """

 def __init__(
 self,
 hidden_dim: int,
 ffn_dim: int,
 dropout: float = 0.1,
 ):
 """
 Args:
 hidden_dim: 
 ffn_dim: 
 dropout: Dropout
 """
 super().__init__()
 self.gate_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
 self.up_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
 self.down_proj = nn.Linear(ffn_dim, hidden_dim, bias=False)
 self.dropout = nn.Dropout(dropout)

 def forward(self, x: Tensor) -> Tensor:
 """ - SwiGLU"""
 gate = F.silu(self.gate_proj(x))
 up = self.up_proj(x)
 return self.dropout(self.down_proj(gate * up))

class TransformerBlock(nn.Module):
 """
 Transformer

 : Pre-LayerNorm + Attention + FFN ()
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
 hidden_dim: 
 num_heads: 
 ffn_dim: 
 dropout: Dropout
 max_seq_length: 
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
 

 Args:
 x: 
 attention_mask: 
 past_kv: KV
 use_cache: 

 Returns:
 output: 
 present_kv: KV
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
 - Decoder-only Transformer

 :
 - 40
 - 5120
 - 32
 - 20480
 - 50000

 
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
 expert_id: ID (0-7)
 vocab_size: 
 hidden_dim: 
 num_layers: Transformer
 num_heads: 
 ffn_dim: 
 max_seq_length: 
 dropout: Dropout
 tie_word_embeddings: 
 """
 super().__init__()

 self.expert_id = expert_id
 self.vocab_size = vocab_size
 self.hidden_dim = hidden_dim
 self.num_layers = num_layers
 self.max_seq_length = max_seq_length

 # 
 self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
 self.dropout = nn.Dropout(dropout)

 # Transformer
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

 # 
 self.final_norm = RMSNorm(hidden_dim)

 # ()
 self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

 # 
 if tie_word_embeddings:
 self.lm_head.weight = self.token_embedding.weight

 # 
 self._init_weights()

 def _init_weights(self):
 """"""
 for module in self.modules():
 if isinstance(module, nn.Linear):
 torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
 if module.bias is not None:
 torch.nn.init.zeros_(module.bias)
 elif isinstance(module, nn.Embedding):
 torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

 def get_num_params(self) -> int:
 """"""
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
 

 Args:
 input_ids: token ID [batch, seq]
 attention_mask: 
 past_key_values: KV
 use_cache: 
 output_hidden_states: 
 return_dict: 

 Returns:
 :
 - logits: logits [batch, seq, vocab]
 - hidden_states: (output_hidden_states=True)
 - past_key_values: KV (use_cache=True)
 """
 batch_size, seq_length = input_ids.shape

 # Token
 hidden_states = self.token_embedding(input_ids)
 hidden_states = self.dropout(hidden_states)

 # attention_mask
 # : [batch, seq] -> [batch, 1, 1, seq] [batch, 1, seq, seq]
 if attention_mask is not None:
 # 
 if attention_mask.dtype == torch.bool:
 attention_mask = attention_mask.float()
 elif attention_mask.dtype == torch.long:
 attention_mask = attention_mask.float()

 # 0/1attention (0 -> -inf, 1 -> 0)
 # attention_mask: [batch, seq] -> [batch, 1, 1, seq]
 if attention_mask.dim() == 2:
 # [batch, seq] -> [batch, 1, 1, seq]
 attention_mask = attention_mask[:, None, None, :]
 # : 1 -> 0 (attend), 0 -> -inf (ignore)
 attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min
 elif attention_mask.dim() == 3:
 # [batch, 1, seq] -> [batch, 1, 1, seq]
 attention_mask = attention_mask[:, None, :, :]
 attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min
 # 4D，

 # 
 all_hidden_states = [hidden_states] if output_hidden_states else None

 # KV
 if past_key_values is None:
 past_key_values = [None] * self.num_layers

 present_key_values = [] if use_cache else None

 # Transformer
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

 # 
 hidden_states = self.final_norm(hidden_states)

 # logits
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
 

 Args:
 input_ids: token ID
 labels: (shifted input_ids)
 attention_mask: 

 Returns:
 loss: 
 logits: logits
 """
 outputs = self.forward(input_ids, attention_mask)
 logits = outputs["logits"]

 # ，shifted input_ids
 if labels is None:
 labels = input_ids

 # Shift for causal LM loss
 shift_logits = logits[..., :-1, :].contiguous()
 shift_labels = labels[..., 1:].contiguous()

 # 
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
 

 Args:
 input_ids: token ID
 attention_mask: 
 temperature: 

 Returns:
 [batch, seq, vocab]
 """
 outputs = self.forward(input_ids, attention_mask)
 logits = outputs["logits"]

 # 
 if temperature != 1.0:
 logits = logits / temperature

 return F.softmax(logits, dim=-1)

 def get_perplexity(
 self,
 input_ids: Tensor,
 attention_mask: Optional[Tensor] = None,
 ) -> Tensor:
 """
 

 Args:
 input_ids: token ID
 attention_mask: 

 Returns:
 
 """
 loss, _ = self.compute_loss(input_ids, attention_mask=attention_mask)
 return torch.exp(loss)

class ExpertModelSmall(ExpertModel):
 """
 ()

 :
 - 6
 - 768
 - 12
 - 3072
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
 - K

 K* = 8，
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
 num_experts: (K* = 8)
 : 
 use_small_model: 
 """
 super().__init__()

 self.num_experts = num_experts
 self.hidden_dim = hidden_dim

 # 
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

 # 
 self.domain_mapping = {}

 def set_domain_mapping(self, mapping: Dict[int, str]):
 """-"""
 self.domain_mapping = mapping

 def get_expert(self, expert_id: int) -> ExpertModel:
 """ID"""
 return self.experts[expert_id]

 def forward_single_expert(
 self,
 expert_id: int,
 input_ids: Tensor,
 attention_mask: Optional[Tensor] = None,
 **kwargs,
 ) -> Dict[str, Any]:
 """"""
 return self.experts[expert_id](input_ids, attention_mask, **kwargs)

 def forward_all_experts(
 self,
 input_ids: Tensor,
 attention_mask: Optional[Tensor] = None,
 **kwargs,
 ) -> List[Dict[str, Any]]:
 """"""
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
 """"""
 probs = []
 for expert in self.experts:
 prob = expert.get_probabilities(input_ids, attention_mask, temperature)
 probs.append(prob)
 return probs

 def get_total_params(self) -> int:
 """"""
 return sum(expert.get_num_params() for expert in self.experts)

if __name__ == "__main__":
 # 
 print("Testing Expert Model...")

 # 
 expert = ExpertModelSmall(expert_id=0)
 print(f"Expert parameters: {expert.get_num_params():,}")

 # 
 batch_size, seq_length = 2, 128
 input_ids = torch.randint(0, 50000, (batch_size, seq_length))

 outputs = expert(input_ids, output_hidden_states=True)
 print(f"Logits shape: {outputs['logits'].shape}")
 print(f"Hidden states count: {len(outputs['hidden_states'])}")

 # 
 loss, logits = expert.compute_loss(input_ids)
 print(f"Loss: {loss.item():.4f}")

 # 
 print("\nTesting Multi-Expert System...")
 multi_expert = MultiExpertSystem(num_experts=8)
 print(f"Total parameters: {multi_expert.get_total_params():,}")

 # 
 all_outputs = multi_expert.forward_all_experts(input_ids)
 print(f"Number of expert outputs: {len(all_outputs)}")

 print("\nAll tests passed!")