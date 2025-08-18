# train/minilmv2_fast.py
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F

def _kl_relation(rel_T: torch.Tensor, rel_S: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """KL‑divergence between teacher & student relation logits.

    Arguments
    ---------
    rel_T, rel_S : (B, A_r, S, S) – un‑normalised logits.
    mask         : (B, S)         – 1 = real token, 0 = padding.

    Returns
    -------
    torch.Tensor scalar – mean KL loss over the batch (same scaling as paper).
    """
    B, A_r, S, _ = rel_T.shape

    mask_b = mask.to(torch.bool)                     # (B, S)
    q_mask = mask_b[:, None, :, None]                # (B,1,S,1)
    k_mask = mask_b[:, None, None, :]                # (B,1,1,S)
    m = q_mask & k_mask                              # (B,1,S,S)

    # exclude padded keys before softmax/log_softmax to match reference slice
    neg_inf = torch.finfo(rel_T.dtype).min
    rel_T_m = rel_T.masked_fill(~m, neg_inf)
    rel_S_m = rel_S.masked_fill(~m, neg_inf)

    p = F.softmax(rel_T_m, dim=-1)                   # teacher probs
    q_log = F.log_softmax(rel_S_m, dim=-1)           # student log‑probs

    kl_el = F.kl_div(q_log, p, reduction="none")    # (B,A_r,S,S)
    kl_sum = (kl_el * m).sum(dim=(-1, -2))           # (B,A_r) – sum over keys & queries

    L = mask_b.sum(dim=-1).clamp(min=1)              # (B,) real sequence lengths
    loss_b = kl_sum.sum(dim=-1) / (A_r * L)          # per‑batch loss (paper eq. 6)

    return loss_b.mean()                             # mean over batch


# ---------------------------------------------------------------------------
# Main class – drop‑in replacement
# ---------------------------------------------------------------------------

class MiniLMv2Fast(nn.Module):
    """Faster MiniLMv2 loss module with correct QKV hooks, FlashAttention‑2, AMP."""

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        L: int,
        M: int,
        relations: Dict[Tuple[int, int], float],
        A_r: int,
        *,
        compile_student: bool = False  # default False to avoid Accelerate issues
    ) -> None:
        super().__init__()
        self.teacher = teacher.eval()
        self.student = student
        self.L = L
        self.M = M
        self.relations = relations
        self.A_r = A_r
        # caches for hooks
        self._teacher_qkv: Dict[str, torch.Tensor] = {}
        self._student_qkv: Dict[str, torch.Tensor] = {}
        self._hook_handles = []

        # freeze teacher
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        # register forward hooks on query, key, value for teacher and student
        t_self = self._teacher_layer.attention.self
        s_self = self._student_layer.attention.self
        for name, module, cache in [
            ('q', t_self.query, self._teacher_qkv),
            ('k', t_self.key,   self._teacher_qkv),
            ('v', t_self.value, self._teacher_qkv),
            ('q', s_self.query, self._student_qkv),
            ('k', s_self.key,   self._student_qkv),
            ('v', s_self.value, self._student_qkv),
        ]:
            handle = module.register_forward_hook(
                lambda mod, inp, out, nm=name, c=cache: c.update({nm: self._reshape(out)})
            )
            self._hook_handles.append(handle)

        # optional compile only student
        if compile_student and hasattr(torch, 'compile'):
            self.student = torch.compile(self.student, mode='max-autotune')

    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape (B,S,d_h) -> (B,A_r,S,d_r)."""
        B, S, d_h = x.size()
        d_r = d_h // self.A_r
        return x.view(B, S, self.A_r, d_r).permute(0, 2, 1, 3).contiguous()

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor]:
        if attention_mask is None:
            attention_mask = (input_ids != 0).to(torch.long, device=input_ids.device)
        inputs = dict(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        # 1) teacher pass
        with torch.inference_mode():
            _ = self.teacher(**inputs)
        # 2) student pass
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            _ = self.student(**inputs)

        # extract Q, K, V
        qT, kT, vT = (self._teacher_qkv['q'], self._teacher_qkv['k'], self._teacher_qkv['v'])
        qS, kS, vS = (self._student_qkv['q'], self._student_qkv['k'], self._student_qkv['v'])

        loss = input_ids.new_zeros(()).float()
        sqrt_T, sqrt_S = math.sqrt(qT.size(-1)), math.sqrt(qS.size(-1))
        for (m, n), w in self.relations.items():
            T1, T2 = (qT, kT, vT)[m-1], (qT, kT, vT)[n-1]
            S1, S2 = (qS, kS, vS)[m-1], (qS, kS, vS)[n-1]
            logits_T = T1 @ T2.transpose(-1,-2) / sqrt_T
            logits_S = S1 @ S2.transpose(-1,-2) / sqrt_S
            loss = loss + w * _kl_relation(logits_T.detach(), logits_S, attention_mask)
        return (loss,)

    @property
    def _teacher_layer(self):
        return self.teacher.encoder.layer[self.L - 1]
    @property
    def _student_layer(self):
        return self.student.encoder.layer[self.M - 1]

    def __del__(self):
        for h in getattr(self, '_hook_handles', []):
            try: h.remove()
            except: pass

