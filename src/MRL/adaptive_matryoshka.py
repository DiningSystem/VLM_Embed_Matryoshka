from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class AdaptiveMatryoshkaStage1Loss(nn.Module):
    """
    Stage-1 loss for Adaptive Matryoshka representation learning.

    This implements:
      1) CLIP-style cross-modal alignment on a chosen student prefix.
      2) Curriculum training across nested dimensions with trainable projections.
      3) Orthogonality regularization on each projection matrix (P^T P -> I).
      4) Adjacent-dimension spectral consistency via SVD-spectrum KL.

    Supported prefix chain (default): [64, 128, 256, 512, 768, 1024].
    Curriculum stage pairs are built from:
      - explicit user projection graph (`stage1_projection_spec`), or
      - adjacent larger->smaller valid pairs from configured dims.
    Multiple larger dimensions can project into the same smaller dimension.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.temperature = getattr(args, "temperature", 0.02)
        nested_dims = getattr(args, "nested_dims", None) or [64, 128, 256, 512, 768, 1024]
        self.nested_dims = sorted(set(nested_dims))
        self.phase = str(getattr(args, "stage1_phase", "all")).upper()
        self.projection_spec = str(getattr(args, "stage1_projection_spec", "")).strip()
        self.align_l1_weight = float(getattr(args, "align_l1_weight", 0.0))
        self.full_dim_l1_weight = float(getattr(args, "full_dim_l1_weight", 0.0))
        self.orthogonal_weight = float(getattr(args, "orthogonal_weight", 0.01))
        self.orthogonal_pair_weights = self._parse_pair_weight_map(getattr(args, "orthogonal_pair_weights", ""))
        self.spectrum_kl_weight = float(getattr(args, "spectrum_kl_weight", 0.0))
        self.projection_weights = self._parse_pair_weight_map(getattr(args, "stage1_projection_weights", ""))
        self.dim_align_l1_weights = self._parse_dim_weight_map(getattr(args, "align_l1_weights", ""))

        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.process_rank = dist.get_rank()
        else:
            self.world_size = 1
            self.process_rank = 0

    def _dist_gather_tensor(self, t: Tensor) -> Tensor:
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        return torch.cat(all_tensors, dim=0)

    def _build_contrastive_target(self, q: Tensor, p: Tensor) -> Tensor:
        # Supports grouped positives (n_hardneg + 1 layout used in this repo).
        target = torch.arange(q.size(0), device=q.device, dtype=torch.long)
        target_per_qry = p.size(0) // q.size(0)
        return target * target_per_qry

    def _project_to_dim(self, model, x: Tensor, dim: int, src_dim: Optional[int] = None) -> Tensor:
        if src_dim is None:
            src_dim = x.size(-1)
        if src_dim == dim:
            return x[:, :dim]
        if src_dim < dim:
            raise ValueError(f"Cannot project {src_dim} -> {dim}: source dim is smaller.")
        proj_bank = self._get_projection_bank(model)
        if proj_bank is None:
            raise RuntimeError("Model missing `matryoshka_proj_bank`. Attach it before stage1 training.")
        return proj_bank.project(x[:, :src_dim], src_dim=src_dim, dst_dim=dim)

    @staticmethod
    def _unwrap_model(model):
        while hasattr(model, "module"):
            model = model.module
        return model

    def _get_projection_bank(self, model):
        base_model = self._unwrap_model(model)
        return getattr(base_model, "matryoshka_proj_bank", None)

    def _cross_alignment_l1(
        self,
        model,
        qry: Tensor,
        pos: Tensor,
        target: Tensor,
        dim: int,
        bigger_dim: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Cross alignment requested by review:
          - keep a single directional contrastive CE (like base_mrl.py)
          - add L1 between two cross-projected cosine similarity maps

        Branch A (contrastive + cosine map):
          qry_dim vs pos projected from bigger_dim -> dim.
        Branch B (cosine map only):
          qry projected from bigger_dim -> dim vs pos_dim.
        """
        if bigger_dim is None:
            bigger_dim = dim

        q_dim = F.normalize(self._project_to_dim(model, qry, dim, src_dim=bigger_dim), p=2, dim=-1)
        p_dim = F.normalize(self._project_to_dim(model, pos, dim, src_dim=bigger_dim), p=2, dim=-1)

        # One-direction contrastive CE, consistent with base_mrl style.
        logits = (q_dim @ p_dim.t()) / self.temperature
        contrastive = F.cross_entropy(logits, target)

        # Cross-projected cosine maps for L1 consistency.
        # Use native prefix slices for the "small" side so only adjacent projections are required.
        # Map-1: qry_prefix(dim) x proj(pos_big->dim)
        q_small = F.normalize(qry[:, :dim], p=2, dim=-1)
        p_from_big = F.normalize(self._project_to_dim(model, pos, dim, src_dim=bigger_dim), p=2, dim=-1)
        cosine_map_1 = q_small @ p_from_big.t()

        # Map-2: proj(qry_big->dim) x pos_prefix(dim)
        q_from_big = F.normalize(self._project_to_dim(model, qry, dim, src_dim=bigger_dim), p=2, dim=-1)
        p_small = F.normalize(pos[:, :dim], p=2, dim=-1)
        cosine_map_2 = q_from_big @ p_small.t()

        l1_consistency = F.l1_loss(cosine_map_1, cosine_map_2)
        return contrastive, l1_consistency, logits

    def _resolve_dims(self, full_dim: int) -> List[int]:
        valid_dims = [d for d in self.nested_dims if d <= full_dim]
        if full_dim not in valid_dims:
            valid_dims.append(full_dim)
        return sorted(set(valid_dims))

    def _adjacent_spectrum_kl(self, qry_full: Tensor, pos_full: Tensor, valid_dims: List[int]) -> Tuple[Tensor, Dict[str, Tensor]]:
        eps = 1e-8
        device = qry_full.device
        sorted_dims = sorted(set(valid_dims), reverse=True)
        adjacent_pairs = [(sorted_dims[i], sorted_dims[i + 1]) for i in range(len(sorted_dims) - 1)]
        if not adjacent_pairs:
            z = torch.zeros((), device=device, dtype=qry_full.dtype)
            return z, {}

        def _symmetric_kl_from_svals(x_src: Tensor, x_dst: Tensor) -> Tensor:
            s_src = torch.linalg.svdvals(x_src.float())
            s_dst = torch.linalg.svdvals(x_dst.float())
            k = min(s_src.numel(), s_dst.numel())
            p = (s_src[:k] + eps) / (s_src[:k].sum() + eps * k)
            q = (s_dst[:k] + eps) / (s_dst[:k].sum() + eps * k)
            kl_pq = (p * (torch.log(p) - torch.log(q))).sum()
            kl_qp = (q * (torch.log(q) - torch.log(p))).sum()
            return 0.5 * (kl_pq + kl_qp)

        losses: List[Tensor] = []
        aux: Dict[str, Tensor] = {}
        for src_dim, dst_dim in adjacent_pairs:
            q_src, q_dst = qry_full[:, :src_dim], qry_full[:, :dst_dim]
            p_src, p_dst = pos_full[:, :src_dim], pos_full[:, :dst_dim]

            q_loss = _symmetric_kl_from_svals(q_src, q_dst)
            p_loss = _symmetric_kl_from_svals(p_src, p_dst)
            loss_pair = 0.5 * (q_loss + p_loss)
            losses.append(loss_pair.to(dtype=qry_full.dtype))

            aux[f"spectrum_rank_qry_{src_dim}"] = torch.linalg.matrix_rank(q_src.float()).to(dtype=qry_full.dtype).detach()
            aux[f"spectrum_rank_qry_{dst_dim}"] = torch.linalg.matrix_rank(q_dst.float()).to(dtype=qry_full.dtype).detach()
            aux[f"spectrum_rank_pos_{src_dim}"] = torch.linalg.matrix_rank(p_src.float()).to(dtype=qry_full.dtype).detach()
            aux[f"spectrum_rank_pos_{dst_dim}"] = torch.linalg.matrix_rank(p_dst.float()).to(dtype=qry_full.dtype).detach()
            aux[f"spectrum_kl_qry_{src_dim}_to_{dst_dim}"] = q_loss.detach()
            aux[f"spectrum_kl_pos_{src_dim}_to_{dst_dim}"] = p_loss.detach()
            aux[f"spectrum_kl_{src_dim}_to_{dst_dim}"] = loss_pair.detach()
        return torch.stack(losses).mean(), aux

    def _parse_dim_weight_map(self, spec) -> Dict[int, float]:
        """
        Parse per-dimension weight spec.

        Accepted format: "64:0.5,256:1.0,512:1.2"
        """
        if not spec:
            return {}
        if isinstance(spec, dict):
            return {int(k): float(v) for k, v in spec.items()}

        out: Dict[int, float] = {}
        for item in str(spec).split(","):
            item = item.strip()
            if not item:
                continue
            if ":" not in item:
                raise ValueError(
                    f"Invalid dim weight entry '{item}'. Expected format like '64:0.5,256:1.0'."
                )
            dim_str, weight_str = item.split(":", 1)
            out[int(dim_str.strip())] = float(weight_str.strip())
        return out

    def _resolve_selected_stage_ids(self, stage_pairs: List[Tuple[int, int]]) -> List[int]:
        """
        Resolve user-selected curriculum stages.

        Backward compatibility:
          - "A/B/C/D" still maps to stage indices 0/1/2/3.
        Generalized behavior:
          - Any single alphabetic token maps to an index (A=0, B=1, ... Z=25).
          - Comma-separated lists are supported, e.g. "A,C" or "0,2,4".
          - "ALL" means include every available stage built from nested_dims.

        If selection is empty or invalid, defaults to all available stage pairs.
        """
        if not stage_pairs:
            return []
        max_idx = len(stage_pairs) - 1
        phase = self.phase.strip().upper()

        if phase == "ALL":
            return list(range(len(stage_pairs)))

        tokens = [tok.strip() for tok in phase.replace(";", ",").split(",") if tok.strip()]
        selected_ids: List[int] = []
        for token in tokens:
            if token.isdigit():
                selected_ids.append(int(token))
                continue

            if token.isalpha():
                # Support arbitrary alphabetic stage labels beyond D.
                if len(token) == 1:
                    selected_ids.append(ord(token) - ord("A"))
                else:
                    # Accept labels like "PHASE_E" by reading the trailing letter.
                    last_char = token[-1]
                    if "A" <= last_char <= "Z":
                        selected_ids.append(ord(last_char) - ord("A"))

        selected_ids = sorted({idx for idx in selected_ids if 0 <= idx <= max_idx})
        return selected_ids or list(range(len(stage_pairs)))

    def _parse_pair_weight_map(self, spec) -> Dict[Tuple[int, int], float]:
        """
        Parse per-pair orthogonal regularizer weights.

        Accepted format:
          - "1024->512:1.0,512->256:0.7"
          - "1024:512:1.0,512:256:0.7"
        """
        if not spec:
            return {}

        out: Dict[Tuple[int, int], float] = {}
        for item in str(spec).split(","):
            item = item.strip()
            if not item:
                continue

            if "->" in item and ":" in item:
                pair_spec, weight_spec = item.rsplit(":", 1)
                src_str, dst_str = pair_spec.split("->", 1)
            else:
                parts = item.split(":")
                if len(parts) != 3:
                    raise ValueError(
                        f"Invalid orthogonal pair weight entry '{item}'. "
                        f"Use '1024->512:1.0' (or '1024:512:1.0')."
                    )
                src_str, dst_str, weight_spec = parts

            src_dim = int(src_str.strip())
            dst_dim = int(dst_str.strip())
            out[(src_dim, dst_dim)] = float(weight_spec.strip())
        return out

    @staticmethod
    def _resolve_pair_weight(weight_map: Dict[Tuple[int, int], float], src_dim: int, dst_dim: int, default: float = 1.0) -> float:
        """
        Resolve a pair weight with explicit fallback.
        - If (src_dim, dst_dim) exists in `weight_map`, use it.
        - Otherwise use `default`.
        """
        return float(weight_map.get((src_dim, dst_dim), default))

    def _parse_projection_pairs(self, spec: str) -> List[Tuple[int, int]]:
        """
        Parse projection pair spec in formats:
          - "1024->768,1024->512,768->512"
          - "1024:768,1024:512"
        """
        out: List[Tuple[int, int]] = []
        if not spec:
            return out
        for item in str(spec).split(","):
            item = item.strip()
            if not item:
                continue
            if "->" in item:
                src_str, dst_str = item.split("->", 1)
            else:
                parts = item.split(":")
                if len(parts) != 2:
                    raise ValueError(
                        f"Invalid stage1 projection entry '{item}'. "
                        f"Use '1024->768' (or '1024:768')."
                    )
                src_str, dst_str = parts
            src_dim = int(src_str.strip())
            dst_dim = int(dst_str.strip())
            if src_dim <= dst_dim:
                raise ValueError(
                    f"Invalid projection pair {src_dim}->{dst_dim}. Source dim must be larger than destination dim."
                )
            out.append((src_dim, dst_dim))
        return out

    def forward(self, model_trainer, input_data: Dict[str, Dict[str, Tensor]]) -> Dict[str, Tensor]:
        model = model_trainer.model
        qry_input = input_data["qry"]
        pos_input = input_data["pos"]

        qry_output = model.encode_input(qry_input, output_hidden_states=False, output_attentions=False)
        pos_output = model.encode_input(pos_input, output_hidden_states=False, output_attentions=False)

        if isinstance(qry_output, tuple):
            qry_full = qry_output[0]
        else:
            qry_full = qry_output
        if isinstance(pos_output, tuple):
            pos_full = pos_output[0]
        else:
            pos_full = pos_output
        if self.world_size > 1:
            qry_full = self._dist_gather_tensor(qry_full)
            pos_full = self._dist_gather_tensor(pos_full)

        full_dim = qry_full.size(-1)
        valid_dims = self._resolve_dims(full_dim)
        target = self._build_contrastive_target(qry_full, pos_full)

        desc_dims = sorted(valid_dims, reverse=True)
        stage_pairs: List[Tuple[int, int]] = []
        if self.projection_spec:
            parsed_pairs = self._parse_projection_pairs(self.projection_spec)
            valid_dim_set = set(valid_dims)
            stage_pairs = [
                (src_dim, dst_dim)
                for src_dim, dst_dim in parsed_pairs
                if src_dim in valid_dim_set and dst_dim in valid_dim_set
            ]
        else:
            # Default: adjacent larger->smaller pairs only (stable baseline).
            # For dims [1024, 768, 512, 256], this yields:
            #   1024->768, 768->512, 512->256
            for i in range(len(desc_dims) - 1):
                src_dim = desc_dims[i]
                dst_dim = desc_dims[i + 1]
                if src_dim > dst_dim:
                    stage_pairs.append((src_dim, dst_dim))
        # remove duplicates while preserving order
        stage_pairs = list(dict.fromkeys(stage_pairs))

        selected_ids = self._resolve_selected_stage_ids(stage_pairs)

        losses = []
        align_losses = []
        orth_losses = []
        metrics: Dict[str, Tensor] = {}

        # Always keep a full-dimension anchor objective so projected stages do not
        # drift away from the base retrieval representation.
        full_dim = desc_dims[0]
        full_align_ce, full_align_l1, _ = self._cross_alignment_l1(
            model=model,
            qry=qry_full,
            pos=pos_full,
            target=target,
            dim=full_dim,
            bigger_dim=full_dim,
        )
        full_align_loss = full_align_ce + self.full_dim_l1_weight * full_align_l1
        losses.append(full_align_loss)
        align_losses.append(full_align_loss)
        orth_losses.append(torch.zeros_like(full_align_loss))
        metrics[f"align_ce_{full_dim}_to_{full_dim}"] = full_align_ce.detach()
        metrics[f"align_l1_{full_dim}_to_{full_dim}"] = full_align_l1.detach()
        metrics[f"align_l1_weight_{full_dim}_to_{full_dim}"] = torch.tensor(self.full_dim_l1_weight, device=full_align_ce.device)
        metrics[f"projection_weight_{full_dim}_to_{full_dim}"] = torch.tensor(1.0, device=full_align_ce.device)
        metrics[f"orthogonal_pair_weight_{full_dim}_to_{full_dim}"] = torch.tensor(0.0, device=full_align_ce.device)
        metrics[f"align_loss_{full_dim}_to_{full_dim}"] = full_align_loss.detach()
        metrics[f"orthogonal_loss_{full_dim}_to_{full_dim}"] = torch.zeros_like(full_align_loss).detach()

        for idx in selected_ids:
            teacher_dim, student_dim = stage_pairs[idx]
            align_ce, align_l1, _ = self._cross_alignment_l1(
                model=model,
                qry=qry_full,
                pos=pos_full,
                target=target,
                dim=student_dim,
                bigger_dim=teacher_dim,
            )

            l1_weight = self.dim_align_l1_weights.get(student_dim, self.align_l1_weight)
            weighted_align_loss = align_ce + l1_weight * align_l1
            projection_weight = self._resolve_pair_weight(
                self.projection_weights,
                teacher_dim,
                student_dim,
                default=1.0,
            )

            proj_bank = self._get_projection_bank(model)
            if proj_bank is not None:
                base_orth = proj_bank.orthogonality_loss(src_dim=teacher_dim, dst_dim=student_dim)
                orth_pair_weight = self._resolve_pair_weight(
                    self.orthogonal_pair_weights,
                    teacher_dim,
                    student_dim,
                    default=1.0,
                )
                orth_loss = orth_pair_weight * base_orth
            else:
                orth_pair_weight = 1.0
                orth_loss = torch.zeros_like(weighted_align_loss)

            total = (
                projection_weight * weighted_align_loss
                + self.orthogonal_weight * projection_weight * orth_loss
            )
            metrics[f"align_ce_{teacher_dim}_to_{student_dim}"] = align_ce.detach()
            metrics[f"align_l1_{teacher_dim}_to_{student_dim}"] = align_l1.detach()
            metrics[f"align_l1_weight_{teacher_dim}_to_{student_dim}"] = torch.tensor(l1_weight, device=align_ce.device)
            metrics[f"projection_weight_{teacher_dim}_to_{student_dim}"] = torch.tensor(projection_weight, device=align_ce.device)
            metrics[f"orthogonal_pair_weight_{teacher_dim}_to_{student_dim}"] = torch.tensor(
                orth_pair_weight, device=align_ce.device
            )
            metrics[f"align_loss_{teacher_dim}_to_{student_dim}"] = weighted_align_loss.detach()
            metrics[f"orthogonal_loss_{teacher_dim}_to_{student_dim}"] = orth_loss.detach()
            losses.append(total)
            align_losses.append(weighted_align_loss)
            orth_losses.append(orth_loss)

        spectrum_kl, spectrum_metrics = self._adjacent_spectrum_kl(qry_full=qry_full, pos_full=pos_full, valid_dims=valid_dims)
        metrics.update(spectrum_metrics)
        if self.spectrum_kl_weight > 0.0:
            final_loss = torch.stack(losses).mean() + self.spectrum_kl_weight * spectrum_kl
        else:
            final_loss = torch.stack(losses).mean()
        metrics["spectrum_kl_loss"] = spectrum_kl.detach()

        mean_align_loss = torch.stack(align_losses).mean()
        mean_orth_loss = torch.stack(orth_losses).mean()

        # Keep `contrastive_loss` for compatibility with existing trainer logging.
        metrics["loss"] = final_loss
        metrics["total_loss"] = final_loss.detach()
        metrics["contrastive_loss"] = mean_align_loss
        metrics["align_loss"] = mean_align_loss.detach()
        metrics["orthogonal_loss"] = mean_orth_loss.detach()
        return metrics


class PairwiseProjectionBank(nn.Module):
    """Trainable projection matrices P for mapping src_dim -> dst_dim with orthogonality regularization."""

    def __init__(self, dimension_pairs: List[Tuple[int, int]]):
        super().__init__()
        self.projections = nn.ParameterDict()
        self.residual_gates = nn.ParameterDict()
        for src_dim, dst_dim in dimension_pairs:
            key = self._key(src_dim, dst_dim)
            self.projections[key] = nn.Parameter(self._init_projection(src_dim, dst_dim))
            if src_dim > dst_dim:
                self.residual_gates[key] = nn.Parameter(torch.zeros(src_dim - dst_dim, dtype=torch.float32))

    @staticmethod
    def _key(src_dim: int, dst_dim: int) -> str:
        return f"{int(src_dim)}_to_{int(dst_dim)}"

    @staticmethod
    def _init_projection(src_dim: int, dst_dim: int) -> Tensor:
        if src_dim == dst_dim:
            return torch.eye(src_dim, dtype=torch.float32)
        mat = torch.randn(src_dim, dst_dim, dtype=torch.float32)
        q, _ = torch.linalg.qr(mat, mode="reduced")
        return q[:, :dst_dim]

    def project(self, x: Tensor, src_dim: int, dst_dim: int) -> Tensor:
        if src_dim == dst_dim:
            return x[:, :dst_dim]
        key = self._key(src_dim, dst_dim)
        if key not in self.projections:
            raise KeyError(f"Missing projection matrix for {src_dim}->{dst_dim}.")
        return x @ self.projections[key]

    def residual_gate(self, src_dim: int, dst_dim: int, device: torch.device, dtype: torch.dtype) -> Optional[Tensor]:
        key = self._key(src_dim, dst_dim)
        if key not in self.residual_gates:
            return None
        gate = torch.sigmoid(self.residual_gates[key])
        return gate.to(device=device, dtype=dtype).unsqueeze(0)

    def orthogonality_loss(self, src_dim: int, dst_dim: int) -> Tensor:
        if src_dim == dst_dim:
            return torch.zeros((), device=next(self.parameters()).device)
        key = self._key(src_dim, dst_dim)
        if key not in self.projections:
            raise KeyError(f"Missing projection matrix for {src_dim}->{dst_dim}.")
        p = self.projections[key]
        gram = p.transpose(0, 1) @ p
        eye = torch.eye(dst_dim, device=p.device, dtype=p.dtype)
        return ((gram - eye) ** 2).mean()
