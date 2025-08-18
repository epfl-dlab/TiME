# train/minilmv2_ltg_fast.py
import logging
import math
from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn.functional as F
from torch import nn

def _kl_relation(rel_T: torch.Tensor, rel_S: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """KL‑divergence between teacher & student relation logits."""
    B, A_r, S, _ = rel_T.shape
    if B == 0:  # Handle empty batch
        return torch.tensor(0.0, device=rel_T.device, dtype=rel_T.dtype)

    mask_b = mask.to(dtype=torch.bool, device=rel_T.device)  # Ensure on same device
    q_mask = mask_b[:, None, :, None]
    k_mask = mask_b[:, None, None, :]
    m = q_mask & k_mask

    neg_inf = torch.finfo(rel_T.dtype).min

    # Use torch.where for out-of-place masked_fill to avoid in-place modification issues
    rel_T_m = torch.where(~m, neg_inf, rel_T)
    rel_S_m = torch.where(~m, neg_inf, rel_S)

    p = F.softmax(rel_T_m, dim=-1)
    q_log = F.log_softmax(rel_S_m, dim=-1)

    # Ensure target p does not have NaNs that could propagate from 0*log(0) in softmax
    # if a whole row was -inf (though masked_fill with -inf should handle this for softmax)
    # More robust: F.kl_div expects log-probs as input and probs as target if log_target=False
    kl_el = F.kl_div(q_log, p, reduction="none", log_target=False)

    # Zero out losses for padding tokens AFTER kl_div
    # kl_el might be nan where p_i=0 (0 log 0), kl_el * 0 will be nan.
    # If p_i is 0, its contribution to KL divergence is 0.
    # So, where p is 0 (due to masking and softmax), kl_el should effectively be 0.
    # A common way to handle this:
    kl_el = torch.where(p == 0, torch.zeros_like(kl_el), kl_el)  # Handles 0 log 0 -> 0
    kl_el = torch.where(torch.isnan(kl_el), torch.zeros_like(kl_el), kl_el)  # Catch any other NaNs and zero them

    kl_sum_per_token_pair = (kl_el * m)  # Apply mask to zero out padded positions in loss
    kl_sum_per_head_per_example = kl_sum_per_token_pair.sum(dim=(-1, -2))  # Sum over S, S

    # L_seq is the number of actual tokens per example
    L_seq = mask_b.sum(dim=-1).clamp(min=1).to(dtype=rel_T.dtype)  # Ensure float for division

    # Sum KL over heads for each example, then normalize
    loss_b_per_example = kl_sum_per_head_per_example.sum(dim=-1) / (A_r * L_seq)

    # Average over valid examples in the batch
    num_valid_examples = (mask_b.sum(dim=-1) > 0).sum()
    if num_valid_examples > 0:
        return loss_b_per_example.sum() / num_valid_examples
    else:
        return torch.tensor(0.0, device=rel_T.device, dtype=rel_T.dtype)


class MiniLMLtgAdapted(nn.Module):
    def __init__(
            self,
            teacher: nn.Module,  # LtgbertModel or LtgbertForMaskedLM
            student: nn.Module,  # LtgbertModel or standard BertModel
            L: int,  # 1-based teacher layer index
            M: int,  # 1-based student layer index
            relations: Dict[Tuple[int, int], float],
            A_r: int,  # Number of relation heads
    ):
        super().__init__()
        self.teacher = teacher.eval()
        self.student = student.train()

        for p in self.teacher.parameters():
            p.requires_grad = False

        self.L = L
        self.M = M
        self.relations = relations
        self.A_r = A_r

        self._teacher_qkv_cache: Dict[str, torch.Tensor] = {}
        self._student_qkv_cache: Dict[str, torch.Tensor] = {}
        self._hook_handles: list = []

        self.is_teacher_ltg = hasattr(self.teacher, 'transformer') and \
                              hasattr(self.teacher.transformer, 'layers')
        self.is_student_ltg = hasattr(self.student, 'transformer') and \
                              hasattr(self.student.transformer, 'layers')

        if not self.is_teacher_ltg:
            raise ValueError("Teacher must be an LTG-BERT model for MiniLMLtgAdapted.")

        teacher_attention_module = self.teacher.transformer.layers[self.L - 1].attention
        self._register_ltg_hooks(teacher_attention_module, self._teacher_qkv_cache, self.teacher.config.hidden_size)

        if self.is_student_ltg:
            student_attention_module = self.student.transformer.layers[self.M - 1].attention
            self._register_ltg_hooks(student_attention_module, self._student_qkv_cache, self.student.config.hidden_size)
        else:
            if not (hasattr(self.student, 'encoder') and hasattr(self.student.encoder, 'layer')):
                raise ValueError(
                    "Student model does not have the expected 'encoder.layer' structure for BERT-like models.")
            try:
                student_self_attention_module = self.student.encoder.layer[self.M - 1].attention.self
                self._register_bert_hooks(student_self_attention_module, self._student_qkv_cache,
                                          self.student.config.hidden_size)
            except AttributeError as e:
                raise ValueError(
                    f"Could not find 'attention.self' for student layer {self.M - 1}. Student structure: {type(self.student)}. Original error: {e}")

        logging.info(
            f"MiniLMLtgAdapted: Teacher (LTG Layer {L}), Student (Layer {M}, type: {'LTG' if self.is_student_ltg else 'BERT'}). Relations: {relations}, A_r: {A_r}"
        )

    def _reshape_to_relation_heads(self, x_bsh: torch.Tensor, model_hidden_size: int) -> torch.Tensor:
        """
        Reshapes x_bsh (expected B, S, H) into (B, A_r, S, d_r).
        Validates that x_bsh is indeed 3D.
        """
        if not isinstance(x_bsh, torch.Tensor) or len(x_bsh.shape) != 3:
            # This indicates it's likely from relative_embedding projection or other unexpected source.
            raise ValueError(
                f"Input to _reshape_to_relation_heads must be a 3D tensor (B,S,H), got shape {x_bsh.shape if isinstance(x_bsh, torch.Tensor) else type(x_bsh)}")

        _B, _S, H = x_bsh.size()  # Use _B, _S to avoid conflict if B, S are class members or globals
        if H != model_hidden_size:
            raise ValueError(f"Tensor hidden size {H} does not match model_hidden_size {model_hidden_size}")
        if model_hidden_size % self.A_r != 0:
            raise ValueError(f"Model hidden size {model_hidden_size} is not divisible by A_r {self.A_r}")
        d_r = model_hidden_size // self.A_r
        return x_bsh.view(_B, _S, self.A_r, d_r).permute(0, 2, 1, 3).contiguous()

    def _ltg_qk_hook(self, module: nn.Linear, inp: Tuple[torch.Tensor], out: torch.Tensor,
                     cache: Dict[str, torch.Tensor], hidden_size: int):
        input_tensor_to_linear = inp[0]
        if len(input_tensor_to_linear.shape) != 3:  # Expects (S, B, H) from LTG attention input
            # logging.debug(f"LTG QK Hook: Skipping input with shape {input_tensor_to_linear.shape}")
            return

        q_proj, k_proj = out.chunk(2, dim=-1)
        q_bsh = q_proj.transpose(0, 1)  # Transpose S,B,H -> B,S,H
        k_bsh = k_proj.transpose(0, 1)  # Transpose S,B,H -> B,S,H

        try:
            cache['q'] = self._reshape_to_relation_heads(q_bsh, hidden_size)
            cache['k'] = self._reshape_to_relation_heads(k_bsh, hidden_size)
        except ValueError as e:
            logging.warning(
                f"LTG QK Hook: Failed to reshape Q/K. Error: {e}. Q_proj shape: {q_proj.shape}, K_proj shape: {k_proj.shape}")
            pass  # Or re-raise if this shouldn't happen

    def _ltg_v_hook(self, module: nn.Linear, inp: Tuple[torch.Tensor], out: torch.Tensor,
                    cache: Dict[str, torch.Tensor], hidden_size: int):
        input_tensor_to_linear = inp[0]
        if len(input_tensor_to_linear.shape) != 3:  # Expects (S, B, H)
            # logging.debug(f"LTG V Hook: Skipping input with shape {input_tensor_to_linear.shape}")
            return

        v_bsh = out.transpose(0, 1)  # Transpose S,B,H -> B,S,H
        try:
            cache['v'] = self._reshape_to_relation_heads(v_bsh, hidden_size)
        except ValueError as e:
            logging.warning(f"LTG V Hook: Failed to reshape V. Error: {e}. V_proj shape: {out.shape}")
            pass

    def _register_ltg_hooks(self, attention_module: nn.Module, cache: Dict[str, torch.Tensor], hidden_size: int):
        self._hook_handles.append(
            attention_module.in_proj_qk.register_forward_hook(
                lambda m, i, o: self._ltg_qk_hook(m, i, o, cache, hidden_size)
            )
        )
        self._hook_handles.append(
            attention_module.in_proj_v.register_forward_hook(
                lambda m, i, o: self._ltg_v_hook(m, i, o, cache, hidden_size)
            )
        )

    def _bert_qkv_hook(self, module: nn.Linear, inp: Tuple[torch.Tensor], out: torch.Tensor, name: str,
                       cache: Dict[str, torch.Tensor], hidden_size: int):
        input_tensor_to_linear = inp[0]  # Expects (B,S,H) for BertSelfAttention
        if len(input_tensor_to_linear.shape) != 3:
            # logging.debug(f"BERT {name} Hook: Skipping input with shape {input_tensor_to_linear.shape}")
            return
        # `out` from BertSelfAttention query/key/value linear layers is (B, S, H)
        try:
            cache[name] = self._reshape_to_relation_heads(out, hidden_size)
        except ValueError as e:
            logging.warning(f"BERT {name} Hook: Failed to reshape. Error: {e}. Out shape: {out.shape}")
            pass

    def _register_bert_hooks(self, self_attention_module: nn.Module, cache: Dict[str, torch.Tensor], hidden_size: int):
        for name, layer_attr in [('q', 'query'), ('k', 'key'), ('v', 'value')]:
            try:
                linear_layer = getattr(self_attention_module, layer_attr)
                self._hook_handles.append(
                    linear_layer.register_forward_hook(
                        # Bind name, cache, hidden_size at definition time
                        lambda m, i, o, n=name, c=cache, h=hidden_size: self._bert_qkv_hook(m, i, o, n, c, h)
                    )
                )
            except AttributeError:
                raise AttributeError(
                    f"Student BertSelfAttention module for layer {self.M - 1} does not have attribute '{layer_attr}'")

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: Optional[torch.Tensor] = None,  # Added Optional
    ) -> Tuple[torch.Tensor]:

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        self._teacher_qkv_cache.clear()
        self._student_qkv_cache.clear()

        # Ensure inputs are on the same device as the model parameters
        # This is typically handled by the Trainer, but good for standalone use
        expected_device = next(self.teacher.parameters()).device
        input_ids = input_ids.to(expected_device)
        attention_mask = attention_mask.to(expected_device)
        token_type_ids = token_type_ids.to(expected_device)

        common_inputs = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "output_hidden_states": False,
            "output_attentions": False,
        }

        with torch.inference_mode():
            # try:
            _ = self.teacher(**common_inputs)
            # except Exception as e:
            #     logging.error(f"Error during teacher forward pass: {e}")
            #     # Potentially log common_inputs shapes here for debugging
            #     raise

        # try:
        _ = self.student(**common_inputs)
        # except Exception as e:
        #     logging.error(f"Error during student forward pass: {e}")
        #     raise

        if not self._teacher_qkv_cache or 'q' not in self._teacher_qkv_cache or 'k' not in self._teacher_qkv_cache or 'v' not in self._teacher_qkv_cache:
            # Add more debug info if cache is empty
            logging.error(f"Teacher QKV cache is incomplete or empty. Cache contents: {self._teacher_qkv_cache.keys()}")
            # You might want to add debug prints inside the hooks to see if they are even called
            # or if the conditions (like input shape) are met.
            raise RuntimeError(
                "Teacher QKV cache is incomplete or empty after forward pass. Hooks might not have run correctly for all Q,K,V components.")

        qT, kT, vT = (self._teacher_qkv_cache['q'], self._teacher_qkv_cache['k'], self._teacher_qkv_cache['v'])

        if not self._student_qkv_cache or 'q' not in self._student_qkv_cache or 'k' not in self._student_qkv_cache or 'v' not in self._student_qkv_cache:
            logging.error(f"Student QKV cache is incomplete or empty. Cache contents: {self._student_qkv_cache.keys()}")
            raise RuntimeError(
                "Student QKV cache is incomplete or empty after forward pass. Hooks might not have run correctly for all Q,K,V components.")

        qS, kS, vS = (self._student_qkv_cache['q'], self._student_qkv_cache['k'], self._student_qkv_cache['v'])

        loss = torch.tensor(0.0, device=input_ids.device, dtype=torch.float32)

        dr_T = qT.size(-1)
        dr_S = qS.size(-1)

        if dr_T <= 0 or dr_S <= 0:  # Should be caught by _reshape_to_relation_heads earlier
            raise ValueError(f"Relation head dimensions must be positive. Got Teacher: {dr_T}, Student: {dr_S}")

        sqrt_dr_T = math.sqrt(dr_T)
        sqrt_dr_S = math.sqrt(dr_S)

        for (m_idx, n_idx), weight in self.relations.items():
            vec_T1, vec_T2 = (qT, kT, vT)[m_idx - 1], (qT, kT, vT)[n_idx - 1]
            vec_S1, vec_S2 = (qS, kS, vS)[m_idx - 1], (qS, kS, vS)[n_idx - 1]

            logits_T = (vec_T1 @ vec_T2.transpose(-1, -2)) / sqrt_dr_T
            logits_S = (vec_S1 @ vec_S2.transpose(-1, -2)) / sqrt_dr_S

            loss += weight * _kl_relation(logits_T.detach(), logits_S, attention_mask)

        return (loss,)

    def __del__(self):
        for handle in self._hook_handles:
            try:
                handle.remove()
            except Exception:  # Catch any exception during handle removal
                pass