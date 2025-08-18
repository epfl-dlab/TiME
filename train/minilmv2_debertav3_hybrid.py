# train/minilmv2_debertav3_hybrid.py

from __future__ import annotations

import ast
import math
from typing import Dict, Tuple, List

import torch
from torch import nn
from torch.nn import functional as F


def _kl_relation(rel_T: torch.Tensor, rel_S: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Calculates the KL-divergence between teacher and student relation logits.
    This version is vectorized and numerically stable.
    """
    mask_bool = mask.to(torch.bool)
    q_mask = mask_bool[:, None, :, None]
    k_mask = mask_bool[:, None, None, :]
    attention_mask_4d = q_mask & k_mask

    # Ensure the 4D mask is broadcastable to the logits tensor shape
    if rel_T.shape[1] != attention_mask_4d.shape[1] and attention_mask_4d.shape[1] == 1:
        attention_mask_4d = attention_mask_4d.expand_as(rel_T)

    neg_inf = torch.finfo(rel_T.dtype).min
    rel_T_masked = rel_T.masked_fill(~attention_mask_4d, neg_inf)
    rel_S_masked = rel_S.masked_fill(~attention_mask_4d, neg_inf)

    p = F.softmax(rel_T_masked, dim=-1)
    q_log = F.log_softmax(rel_S_masked, dim=-1)

    kl_div_elements = F.kl_div(q_log, p, reduction="none")
    kl_div_masked = kl_div_elements.masked_fill(~attention_mask_4d, 0.0)

    kl_div_sum_per_batch_item = kl_div_masked.sum(dim=(1, 2, 3))

    num_tokens_per_item = mask_bool.sum(dim=-1).clamp(min=1)
    num_relation_heads = rel_T.shape[1]
    loss_per_batch_item = kl_div_sum_per_batch_item / (num_relation_heads * num_tokens_per_item)

    return loss_per_batch_item.mean()


class DeBERTaToBERTDistiller(nn.Module):
    """
    Distills knowledge from DeBERTa-v3 using a hybrid strategy:
    - Full Q-K attention logits from the teacher.
    - Content-only Q-Q, K-K, V-V relations from the teacher.
    """

    def __init__(
            self,
            teacher: nn.Module,
            student: nn.Module,
            L: int,
            M: int,
            relations: Dict | str,
            A_r: int,
            *,
            compile_student: bool = False
    ) -> None:
        super().__init__()
        self.teacher = teacher.eval()
        self.student = student
        self.teacher_layer_idx = L
        self.student_layer_idx = M
        self.num_relation_heads = A_r

        if isinstance(relations, str):
            relations = ast.literal_eval(relations)
        self._relations = self._process_relations_dict(relations)

        self._teacher_tensors: Dict[str, torch.Tensor] = {}
        self._student_projections: Dict[str, torch.Tensor] = {}
        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []

        for param in self.teacher.parameters():
            param.requires_grad_(False)
        self._register_hooks()

        if compile_student and hasattr(torch, 'compile'):
            self.student = torch.compile(self.student, mode='max-autotune')

    def _process_relations_dict(self, relations_input: Dict) -> Dict[Tuple[str, str], float]:
        # Same as before
        processed = {}
        int_to_str_map = {1: 'query', 2: 'key', 3: 'value'}
        for k_tuple, weight in relations_input.items():
            k1, k2 = k_tuple
            str_k1 = int_to_str_map.get(k1, k1);
            str_k2 = int_to_str_map.get(k2, k2)
            if str_k1 in int_to_str_map.values() and str_k2 in int_to_str_map.values():
                processed[(str_k1, str_k2)] = float(weight)
        print(f"[Distiller INFO] Using relations: {processed}")
        if not processed: raise ValueError("No valid relations were processed.")
        return processed

    def _get_attention_module(self, model, layer_idx):
        # Same as before
        if hasattr(model, 'encoder'):
            encoder = model.encoder
        elif hasattr(model, 'bert'):
            encoder = model.bert.encoder
        else:
            raise AttributeError(f"Could not find an 'encoder' on model {model.__class__.__name__}.")
        return encoder.layer[layer_idx - 1].attention.self

    def _teacher_hook_fn(self, module, inputs, outputs):
        """
        Hook on the DebertaV2Attention module (one level up).
        It captures the FINAL attention logits and the content-only Q/K/V projections.
        """
        # `outputs` of DebertaV2Attention is a tuple (attention_output, attention_matrix)
        # `attention_matrix` is the final attention LOGITS before softmax.
        # Shape: (B, NumHeads, S, S)
        if outputs[1] is not None:
            self._teacher_tensors['qk_logits'] = outputs[1].clone().detach()

        # We also need the content projections for Q-Q, K-K relations.
        # These are attributes of the `self` module inside DebertaV2Attention
        self_module = module.self
        self._teacher_tensors['query_content'] = self_module.query_proj.weight.data.clone().detach() @ inputs[
            0].transpose(-1, -2)
        self._teacher_tensors['query_content'] = self._teacher_tensors['query_content'].transpose(-1, -2)

        self._teacher_tensors['key_content'] = self_module.key_proj.weight.data.clone().detach() @ inputs[0].transpose(
            -1, -2)
        self._teacher_tensors['key_content'] = self._teacher_tensors['key_content'].transpose(-1, -2)

        self._teacher_tensors['value_content'] = self_module.value_proj.weight.data.clone().detach() @ inputs[
            0].transpose(-1, -2)
        self._teacher_tensors['value_content'] = self._teacher_tensors['value_content'].transpose(-1, -2)

    def _student_hook_fn(self, cache_key: str):
        def hook(module, inputs, output):
            self._student_projections[cache_key] = output

        return hook

    def _register_hooks(self):
        # Hook one level up: on DebertaV2Attention, not DisentangledSelfAttention
        teacher_attention_module = self.teacher.encoder.layer[self.teacher_layer_idx - 1].attention
        print(f"[Distiller INFO] Registering hook on teacher module: {teacher_attention_module.__class__.__name__}")
        handle = teacher_attention_module.register_forward_hook(self._teacher_hook_fn)
        self._hook_handles.append(handle)

        # Student hooks remain the same
        student_attention_module = self._get_attention_module(self.student, self.student_layer_idx)
        print(f"[Distiller INFO] Registering hooks on student module: {student_attention_module.__class__.__name__}")
        for name, layer in {'query': student_attention_module.query, 'key': student_attention_module.key,
                            'value': student_attention_module.value}.items():
            handle = layer.register_forward_hook(self._student_hook_fn(name))
            self._hook_handles.append(handle)

    def _reshape_qkv(self, x: torch.Tensor) -> torch.Tensor:
        """Reshapes Q/K/V vectors for relation calculation."""
        B, S, D = x.shape
        d_r = D // self.num_relation_heads
        return x.view(B, S, self.num_relation_heads, d_r).permute(0, 2, 1, 3).contiguous()

    def _reshape_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Reshapes attention logits to match the number of relation heads using interpolation."""
        B, NumHeads, S, S_ = x.shape

        if NumHeads == self.num_relation_heads:
            return x


        x_permuted = x.permute(0, 2, 3, 1)  # Shape is now (B, S, S, NumHeads)




        x_interpolated = F.interpolate(
            x_permuted,  # The input tensor

            size=(S_, self.num_relation_heads),  # Target shape for the last two dims: (32, 64)


            mode='bilinear',


            align_corners=False
        )

        x_final = x_interpolated.permute(0, 3, 1, 2)  # Shape is now (B, Relation_Heads, S, S)


        return x_final

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            token_type_ids: torch.Tensor | None = None,
            **kwargs,
    ) -> Tuple[torch.Tensor]:

        if attention_mask is None:
            attention_mask = (input_ids != self.student.config.pad_token_id).long()

        self._teacher_tensors.clear();
        self._student_projections.clear()

        # Teacher and Student forward passes
        with torch.inference_mode():
            self.teacher(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)

        student_inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "output_attentions": True}
        if self.student.config.type_vocab_size > 1 and token_type_ids is not None:
            student_inputs["token_type_ids"] = token_type_ids

        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available(), dtype=amp_dtype):
            self.student(**student_inputs)

        # Prepare Student tensors
        student_map = {name: self._reshape_qkv(proj) for name, proj in self._student_projections.items()}

        # Prepare Teacher tensors (content-only for now)
        teacher_map = {
            'query': self._reshape_qkv(self._teacher_tensors['query_content']),
            'key': self._reshape_qkv(self._teacher_tensors['key_content']),
            'value': self._reshape_qkv(self._teacher_tensors['value_content']),
        }

        total_loss = torch.tensor(0.0, device=input_ids.device, dtype=torch.float32)
        sqrt_dk_S = math.sqrt(student_map['query'].size(-1))
        sqrt_dk_T_content = math.sqrt(teacher_map['query'].size(-1))

        for (rel_type1, rel_type2), weight in self._relations.items():
            if weight == 0: continue

            S1, S2 = student_map[rel_type1], student_map[rel_type2]
            logits_S = torch.matmul(S1, S2.transpose(-1, -2)) / sqrt_dk_S

            # HYBRID LOGIC: Use full logits for Q-K, content-only for others
            if rel_type1 == 'query' and rel_type2 == 'key':
                logits_T = self._reshape_logits(self._teacher_tensors['qk_logits'])
            else:
                T1, T2 = teacher_map[rel_type1], teacher_map[rel_type2]
                logits_T = torch.matmul(T1, T2.transpose(-1, -2)) / sqrt_dk_T_content

            current_loss = weight * _kl_relation(logits_T.detach().float(), logits_S.float(), attention_mask.float())
            total_loss += current_loss

        return (total_loss,)

    def remove_hooks(self):
        for handle in self._hook_handles: handle.remove()

    def __del__(self):
        self.remove_hooks()


# ---------------------------------------------------------------------------
# 4. Test Cases
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    from transformers import AutoModel, AutoTokenizer, AutoConfig, BertConfig, BertModel

    teacher_model_name = "microsoft/mdeberta-v3-base"
    student_hidden_size = 384
    student_num_layers = 4
    student_num_attention_heads = 6
    num_relation_heads = 12

    print(f"Loading teacher: {teacher_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
    teacher_model = AutoModel.from_pretrained(teacher_model_name)
    print(f"\n--- Teacher Model Structure: {teacher_model.__class__.__name__} ---")

    student_config = BertConfig(
        vocab_size=tokenizer.vocab_size, hidden_size=student_hidden_size,
        num_hidden_layers=student_num_layers, num_attention_heads=student_num_attention_heads,
        intermediate_size=student_hidden_size * 4, max_position_embeddings=teacher_model.config.max_position_embeddings,
        pad_token_id=tokenizer.pad_token_id, type_vocab_size=2,
    )
    student_model = BertModel(config=student_config)

    # Ensure head counts are compatible for this strategy
    assert teacher_model.config.num_attention_heads % num_relation_heads == 0 or \
           num_relation_heads % teacher_model.config.num_attention_heads == 0, \
        "For this hybrid strategy, teacher heads and relation heads must be multiples of each other."

    distiller_args = {
        "teacher": teacher_model, "student": student_model,
        "L": teacher_model.config.num_hidden_layers, "M": student_model.config.num_hidden_layers,
        "A_r": num_relation_heads, "relations": "{(1, 1): 1.0, (2, 2): 1.0, (1, 2): 1.0}",  # Added Q-K relation
        "compile_student": False,
    }

    print("\nInitializing distiller...")
    distiller = DeBERTaToBERTDistiller(**distiller_args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nRunning on device: {device}")
    distiller.to(device)

    dummy_inputs = tokenizer(
        ["This is a test sentence.", "Here is another one for the batch."],
        padding=True, truncation=True, max_length=32, return_tensors="pt"
    ).to(device)

    try:
        print("\n--- Test Run Starting ---")

        # Test 1: Forward pass for loss calculation
        loss_tuple = distiller(**dummy_inputs)
        loss = loss_tuple[0]

        print("\n--- Hook Validation ---")
        assert 'qk_logits' in distiller._teacher_tensors, "Teacher hook did not catch 'qk_logits'!"
        print(f"Teacher 'qk_logits' caught, Shape: {distiller._teacher_tensors['qk_logits'].shape}")
        assert 'query' in distiller._student_projections, "Student hook did not catch 'query'!"
        print(f"Student 'query' caught, Shape: {distiller._student_projections['query'].shape}")

        print(f"\nCalculated distillation loss: {loss.item():.4f}")
        assert loss.item() >= 0, "Loss should be non-negative!"
        print(" Loss calculation: OK")

        # Test 2: Gradient flow
        print("\nPerforming backward pass...")
        optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-4)
        optimizer.zero_grad()
        loss.backward()

        student_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in student_model.parameters() if p.requires_grad)
        teacher_has_grad = any(p.grad is not None for p in teacher_model.parameters())

        assert student_has_grad, "Student model did not receive gradients!"
        assert not teacher_has_grad, "Teacher model should not have gradients!"
        print(" Gradient flow: OK (Only student is being trained)")

    except Exception as e:
        import traceback

        print(f"\nAn error occurred during the test run: {e}")
        traceback.print_exc()
    finally:
        distiller.remove_hooks()
        print("\n--- Test Run Finished ---")