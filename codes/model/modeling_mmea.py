import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel
from mea_modules import MultiModalTransformerFusion, FeedForwardExpert, TokenRouter


class CLIPBackboneWrapper(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.clip = CLIPModel.from_pretrained(self.args.pretrained_model)
        # Projection for image patch tokens to align dimensions
        self.image_tokens_fc = nn.Linear(
            self.args.model.input_image_hidden_dim,
            self.args.model.hidden_dim
        )

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, pixel_values=None):
        clip_output = self.clip(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values
        )

        text_global = clip_output.text_embeds
        image_global = clip_output.image_embeds

        text_seq_tokens = clip_output.text_model_output[0]
        image_patch_tokens = clip_output.vision_model_output[0]

        image_patch_tokens = self.image_tokens_fc(image_patch_tokens)
        return text_global, image_global, text_seq_tokens, image_patch_tokens


class ExpertFusionGate(nn.Module):
    """
    Stage 1 Fusion: Combines Modality, Grain, and Shared expert outputs.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate_attn_modal = nn.Linear(hidden_dim, 1, bias=True)
        self.gate_attn_grain = nn.Linear(hidden_dim, 1, bias=True)

    def forward(self, modality_out, grain_out, shared_out):
        # Calculate gating weights
        w_m = torch.sigmoid(self.gate_attn_modal(modality_out))
        w_g = torch.sigmoid(self.gate_attn_grain(grain_out))

        # Weighted sum residual connection
        joint_emb = w_m * modality_out + w_g * grain_out + shared_out
        return joint_emb


class CoarseFineFusionGate(nn.Module):
    """
    Stage 2 Fusion: Combines Coarse (Global) and Fine (Local) representations.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate_scorer = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, coarse_emb, fine_emb):
        # Stack inputs: [B, 2, D]
        e_joint = torch.stack([coarse_emb, fine_emb], dim=1)

        scores = self.gate_scorer(torch.tanh(e_joint)).squeeze(-1)
        scores_norm = F.softmax(scores, dim=-1)

        weighted_features = scores_norm.unsqueeze(-1) * e_joint
        joint_emb = weighted_features.view(weighted_features.size(0), -1)
        return joint_emb


class MultiViewExpert(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        hidden_dim = args.model.hidden_dim
        hidden_size = args.model.hidden_size

        # Experts
        self.modal_experts = nn.ModuleList([
            FeedForwardExpert(hidden_dim, hidden_size) for _ in range(args.model.modal_num)
        ])
        self.grain_experts = nn.ModuleList([
            FeedForwardExpert(hidden_dim, hidden_size) for _ in range(2)  # 0: Coarse, 1: Fine
        ])
        self.shared_experts = nn.ModuleList([
            FeedForwardExpert(hidden_dim, hidden_size) for _ in range(1)
        ])

        # Routers
        self.router_modality = TokenRouter(hidden_dim)
        self.router_grain = TokenRouter(hidden_dim)

        # Learnable Temperatures
        self.target_scale = self.args.model.target_scale
        self.router_temp_modal = nn.Parameter(torch.tensor(args.model.router_temperature_modal))
        self.router_temp_grain = nn.Parameter(torch.tensor(args.model.router_temperature_grain))

        # Stage 1 Fusion (inside the block in original logic)
        self.fusion_gate = ExpertFusionGate(hidden_dim)

    def _compute_router_loss(self, logits, targets, temp):
        """Helper to compute KL divergence loss for routing."""
        T = torch.clamp(temp, min=0.01, max=10.0)
        target_probs = F.softmax(targets / T, dim=-1)
        log_pred_probs = F.log_softmax(logits / T, dim=-1)
        return F.kl_div(log_pred_probs, target_probs, reduction='batchmean')

    def _process_experts(self, hidden_states, experts, weights):
        """Applies experts and weighted summation."""
        # Calculate expert outputs: List of [B, Seq, D]
        expert_outputs = [exp(hidden_states).unsqueeze(2) for exp in experts]
        # Stack: [B, Seq, Num_Experts, D]
        expert_stack = torch.cat(expert_outputs, dim=2)
        # Apply Weights: weights is [B, Seq, Num_Experts] -> unsqueeze to [B, Seq, Num_Experts, 1]
        weighted_out = (expert_stack * weights.unsqueeze(-1)).sum(dim=2)
        return weighted_out

    def forward(self, hidden_states, label_modal=None, label_grain=None, training=True):
        # 1. Routing Logits
        logits_modal = self.router_modality(hidden_states)
        logits_grain = self.router_grain(hidden_states)

        # 2. Temperature Clamping
        Tm = torch.clamp(self.router_temp_modal, min=0.01, max=10.0)
        Tg = torch.clamp(self.router_temp_grain, min=0.01, max=10.0)

        # 3. Softmax Weights
        weights_modal = torch.softmax(logits_modal / Tm, dim=-1)
        weights_grain = torch.softmax(logits_grain / Tg, dim=-1)

        # 4. Expert Execution
        out_modal = self._process_experts(hidden_states, self.modal_experts, weights_modal)
        out_grain = self._process_experts(hidden_states, self.grain_experts, weights_grain)
        out_shared = self.shared_experts[0](hidden_states)

        # 5. Fusion
        fused_output = self.fusion_gate(out_modal, out_grain, out_shared)

        # 6. Loss Calculation (Only during training)
        router_loss = 0.0
        if training and label_modal is not None:
            # Prepare one-hot targets
            target_modal_logits = F.one_hot(label_modal, num_classes=2).float() * self.target_scale
            target_grain_logits = F.one_hot(label_grain, num_classes=2).float() * self.target_scale

            loss_m = self._compute_router_loss(logits_modal, target_modal_logits, Tm)
            loss_g = self._compute_router_loss(logits_grain, target_grain_logits, Tg)
            router_loss = (loss_m + loss_g) / 2.0

        return fused_output, router_loss


class RetrievalModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.fusion_transformer = MultiModalTransformerFusion(
            args.model.hidden_size, args.model.num_hidden_layers,
            args.model.num_attention_heads, args.model.modal_num,
            args.model.use_intermediate, args.model.intermediate_size
        )

        self.moe_block = MultiViewExpert(args)
        self.final_fusion = CoarseFineFusionGate(args.model.hidden_dim)

        self.layernorms = nn.ModuleDict({
            'text_cls': nn.LayerNorm(args.model.hidden_dim),
            'text_tok': nn.LayerNorm(args.model.hidden_dim),
            'img_cls': nn.LayerNorm(args.model.hidden_dim),
            'img_tok': nn.LayerNorm(args.model.hidden_dim),
        })

        self.fine_counts = []
        if hasattr(args.data, 'text_max_length'): self.fine_counts.append(args.data.text_max_length)
        if hasattr(args.data, 'visual_patch_length'): self.fine_counts.append(args.data.visual_patch_length)

    def _prepare_templates(self, device):
        """Generates label templates for routing supervision."""
        L_text, L_image = self.fine_counts
        # 0: Text/Coarse, 1: Image/Fine (depending on usage context)
        # Template Modal: Text(0), TextTokens(0), Image(1), ImageTokens(1)
        modal_template = torch.tensor([0] * 1 + [0] * L_text + [1] * 1 + [1] * L_image, device=device)
        # Template Grain: TextCls(0), TextTokens(1), ImageCls(0), ImageTokens(1)
        grain_template = torch.tensor([0] * 1 + [1] * L_text + [0] * 1 + [1] * L_image, device=device)
        return modal_template, grain_template

    def _normalize_and_concat(self, text_cls, text_tok, img_cls, img_tok):
        t_c = self.layernorms['text_cls'](text_cls).unsqueeze(1)
        t_t = self.layernorms['text_tok'](text_tok)
        i_c = self.layernorms['img_cls'](img_cls).unsqueeze(1)
        i_t = self.layernorms['img_tok'](img_tok)

        # Concatenate along sequence dimension
        return torch.cat([t_c, t_t, i_c, i_t], dim=1)

    def _compute_matching_scores(self, mention_emb, entity_emb):
        logit_coarse = []
        logit_fine = []
        current_pos = 0

        for fine_len in self.fine_counts:
            # 1. Coarse Matching (CLS token)
            # Slice [B, 1, D]
            cls_m = mention_emb[:, current_pos:current_pos + 1, :]
            cls_e = entity_emb[:, current_pos:current_pos + 1, :]

            # Dot product
            score_c = torch.matmul(cls_m.squeeze(1), cls_e.squeeze(1).transpose(-1, -2))
            logit_coarse.append(score_c)

            # 2. Fine Matching (Cross-Attention on Tokens)
            local_start = current_pos + 1
            local_end = local_start + fine_len

            local_m = mention_emb[:, local_start:local_end, :]
            local_e = entity_emb[:, local_start:local_end, :]

            score_f = self.cross_attention_match(local_m, local_e)
            logit_fine.append(score_f)

            current_pos = local_end

        return logit_coarse, logit_fine

    @staticmethod
    def cross_attention_match(mention_tokens, entity_tokens):
        """Calculates token-to-token alignment score via attention."""
        batch_size_m, _, hidden_size = mention_tokens.size()
        batch_size_e, _, _ = entity_tokens.size()

        mention_exp = mention_tokens.unsqueeze(1)  # [Bm, 1, Seq, D]
        entity_exp = entity_tokens.unsqueeze(0)  # [1, Be, Seq, D]

        # Scaled Dot-Product Attention
        attn_scores = torch.matmul(mention_exp, entity_exp.transpose(-2, -1)) / math.sqrt(hidden_size)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Expand for broadcasting
        mention_for_attn = mention_exp.expand(-1, batch_size_e, -1, -1)
        entity_for_attn = entity_exp.expand(batch_size_m, -1, -1, -1)

        attended_features = torch.matmul(attn_weights, entity_for_attn)

        # Element-wise matching -> Sum -> Mean
        matching_scores = torch.sum(mention_for_attn * attended_features, dim=-1)
        final_scores = torch.mean(matching_scores, dim=-1)
        return final_scores

    def split_coarse_fine(self, inputs):
        coarse_list = []
        fine_list = []
        pos = 0
        for count in self.fine_counts:
            coarse_list.append(inputs[:, pos:pos + 1, :])
            fine_list.append(inputs[:, pos + 1:pos + 1 + count, :])
            pos += 1 + count

        coarse_stacked = torch.cat(coarse_list, dim=1)  # [B, Num_Modal, D]
        fine_stacked = torch.cat(fine_list, dim=1)  # [B, Total_Fine_Len, D]
        return coarse_stacked, fine_stacked

    def forward(self,
                entity_text_cls, entity_text_tokens,
                mention_text_cls, mention_text_tokens,
                entity_image_cls, entity_image_tokens,
                mention_image_cls, mention_image_tokens):

        entity_input = self._normalize_and_concat(
            entity_text_cls, entity_text_tokens, entity_image_cls, entity_image_tokens
        )
        mention_input = self._normalize_and_concat(
            mention_text_cls, mention_text_tokens, mention_image_cls, mention_image_tokens
        )

        # Generate routing targets (for loss calculation)
        m_modal, m_grain = self._prepare_templates(entity_input.device)

        entity_output, loss_router_ent = self.moe_block(
            entity_input,
            m_modal.expand(entity_input.size(0), -1),
            m_grain.expand(entity_input.size(0), -1),
            training=self.training
        )
        mention_output, loss_router_men = self.moe_block(
            mention_input,
            m_modal.expand(mention_input.size(0), -1),
            m_grain.expand(mention_input.size(0), -1),
            training=self.training
        )

        cors_logits, fins_logits = self._compute_matching_scores(mention_output, entity_output)
        # Average of all component matching scores
        expert_matching_score = (sum(cors_logits) + sum(fins_logits)) / (len(cors_logits) + len(fins_logits))

        # Split back to structural components
        ent_coarse, ent_fine = self.split_coarse_fine(entity_output)
        men_coarse, men_fine = self.split_coarse_fine(mention_output)

        # Apply Transformer Fusion within granularities
        joint_ent_coarse, _ = self.fusion_transformer(ent_coarse)
        joint_men_coarse, _ = self.fusion_transformer(men_coarse)
        joint_ent_fine, _ = self.fusion_transformer(ent_fine)
        joint_men_fine, _ = self.fusion_transformer(men_fine)

        # Final Gating Fusion (Coarse vs Fine)
        entity_final = self.final_fusion(joint_ent_coarse, joint_ent_fine)
        mention_final = self.final_fusion(joint_men_coarse, joint_men_fine)

        # Dot product of fused representations
        joint_matching_score = torch.matmul(mention_final, entity_final.transpose(-1, -2))

        # Combine Expert Score and Joint Score
        total_score = (expert_matching_score + joint_matching_score) / 2.0

        temp = self.args.model.temperature
        total_score = total_score / temp
        expert_matching_score = expert_matching_score / temp
        joint_matching_score_scaled = joint_matching_score / temp

        total_router_loss = (loss_router_men + loss_router_ent) / 2.0

        return total_score, (expert_matching_score, joint_matching_score_scaled), total_router_loss