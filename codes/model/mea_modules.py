import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import apply_chunking_to_forward


class MultiModalTransformerFusion(nn.Module):
    def __init__(self, hidden_size, num_hidden_layers, num_attention_heads,
                 modal_num, use_intermediate, intermediate_size):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.modal_num = modal_num

        self.layers = nn.ModuleList([
            CustomBertLayer(hidden_size, num_attention_heads, use_intermediate, intermediate_size)
            for _ in range(num_hidden_layers)
        ])

    def forward(self, embs):
        hidden_states = embs

        all_attentions = []
        for layer_module in self.layers:
            layer_outputs = layer_module(hidden_states, output_attentions=True)
            hidden_states = layer_outputs[0]
            all_attentions.append(layer_outputs[1])

        last_layer_attn = all_attentions[-1]

        attention_pro = torch.sum(last_layer_attn, dim=-3)
        attention_pro_comb = torch.sum(attention_pro, dim=-2) / math.sqrt(self.modal_num * self.num_attention_heads)

        weight_norm = F.softmax(attention_pro_comb, dim=-1)
        weight_norm_expanded = weight_norm.unsqueeze(-1)  # [B, N, 1]

        embs_weighted = embs * weight_norm_expanded
        joint_emb = embs_weighted.sum(dim=1)  # [B, D]

        return joint_emb, hidden_states


class FeedForwardExpert(nn.Module):
    def __init__(self, input_dim, hidden_dim, act_type='identity', dropout_prob=0.1):
        super().__init__()
        act_fn = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'identity': nn.Identity()
        }
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout_prob),
            nn.LayerNorm(hidden_dim),
            act_fn.get(act_type, nn.Identity())
        )

    def forward(self, x):
        return self.net(x)


class TokenRouter(nn.Module):
    def __init__(self, input_dim, num_out=2):
        super().__init__()
        self.net = nn.Linear(input_dim, num_out)

    def forward(self, x):
        return self.net(x)


class CustomBertLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, use_intermediate, intermediate_size):
        super().__init__()
        self.use_intermediate = use_intermediate
        self.chunk_size_feed_forward = 0
        self.seq_len_dim = 1
        self.attention = CustomBertAttention(hidden_size, num_attention_heads)

        if self.use_intermediate:
            self.intermediate = CustomBertIntermediate(hidden_size, intermediate_size)
            self.output = CustomBertOutput(intermediate_size, hidden_size)

    def forward(self, hidden_states, output_attentions=False):
        self_attention_outputs = self.attention(hidden_states, output_attentions=output_attentions)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # Keep attention weights if present

        if not self.use_intermediate:
            return (attention_output,) + outputs

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        return (layer_output,) + outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class CustomBertAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads):
        super().__init__()
        self.self = CustomBertSelfAttention(hidden_size, num_attention_heads)

    def forward(self, hidden_states, output_attentions=False):
        return self.self(hidden_states, output_attentions)


class CustomBertSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({hidden_size}) is not a multiple of the number of attention heads ({num_attention_heads})")

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(0.1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, output_attentions=False):
        mixed_query_layer = self.query(hidden_states)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        return (context_layer, attention_probs) if output_attentions else (context_layer,)


class CustomBertIntermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class CustomBertOutput(nn.Module):
    def __init__(self, intermediate_size, hidden_size):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states