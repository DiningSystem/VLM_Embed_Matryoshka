"""Conditional Multimodal Subspace Matryoshka Representation Learning.

The loss keeps groups contiguous so that an exported embedding remains a normal
tensor, while a learned router determines which groups are active at each
budget.  The group order is therefore sample dependent and its prefixes are
strictly nested.
"""
from typing import Dict, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class ConditionalGroupRouter(nn.Module):
    """Score a group from fixed-size activation statistics.

    A router that consumes the full embedding would require a backbone-specific
    input dimension.  Using mean, variance, and RMS activation per group keeps
    the router conditional on the input while making all of its parameters
    materialized before DistributedDataParallel is constructed.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, groups: torch.Tensor) -> torch.Tensor:
        groups = groups.float()
        features = torch.stack(
            (
                groups.mean(dim=-1),
                groups.var(dim=-1, unbiased=False),
                groups.square().mean(dim=-1).sqrt(),
            ),
            dim=-1,
        )
        return self.network(features).squeeze(-1)


class CMSMatryoshkaLoss(nn.Module):
    """Multi-budget InfoNCE with utility-guided and CMI routing objectives.

    ``cms_ug`` and ``cms_cmi`` are ablation-friendly variants selected through
    ``kd_loss_type``; their disabled regularizer is returned as zero.
    """

    def __init__(self, args):
        super().__init__()
        self.temperature = getattr(args, "temperature", 0.02)
        self.num_groups = getattr(args, "cms_num_groups", 8)
        self.utility_temperature = getattr(args, "cms_utility_temperature", 0.1)
        self.utility_weight = getattr(args, "cms_utility_weight", 1.0)
        self.cmi_weight = getattr(args, "cms_cmi_weight", 0.1)
        self.loss_type = getattr(args, "kd_loss_type", "cms_mrl")
        hidden = getattr(args, "cms_router_hidden_dim", 256)
        self.router = ConditionalGroupRouter(hidden)

    @staticmethod
    def _unpack(encoded):
        return encoded[0] if isinstance(encoded, tuple) else encoded

    @staticmethod
    def _gather(tensor):
        if not dist.is_initialized():
            return tensor
        tensors = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(tensors, tensor.contiguous())
        tensors[dist.get_rank()] = tensor
        return torch.cat(tensors, dim=0)

    def _group(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Pad then reshape embeddings into equal, disjoint routing groups."""
        width = (embeddings.size(-1) + self.num_groups - 1) // self.num_groups
        padded = F.pad(embeddings, (0, width * self.num_groups - embeddings.size(-1)))
        return padded.view(embeddings.size(0), self.num_groups, width), width

    def _infonce(self, query, target, mask, positives, reduction="mean"):
        # Every query has its own routed coordinates, so apply its prefix mask
        # to every candidate target before computing the [B, N] score matrix.
        q = F.normalize((query * mask).flatten(1), dim=-1)
        t = F.normalize((target.unsqueeze(0) * mask.unsqueeze(1)).flatten(2), dim=-1)
        logits = torch.einsum("bd,bnd->bn", q, t) / self.temperature
        return F.cross_entropy(logits, positives, reduction=reduction)

    def forward(self, model_trainer, input_data) -> Dict[str, torch.Tensor]:
        model = model_trainer.model
        query = self._unpack(model.encode_input(input_data["qry"]))
        target = self._unpack(model.encode_input(input_data["pos"]))
        # Global negatives are used for base loss; router supervision stays local.
        all_target = self._gather(target)
        offset = dist.get_rank() * query.size(0) if dist.is_initialized() else 0
        positives = torch.arange(query.size(0), device=query.device) + offset

        q_groups, width = self._group(query)
        p_groups, _ = self._group(target)
        all_p_groups, _ = self._group(all_target)
        router_logits = self.router(q_groups).to(query.dtype)
        order = router_logits.argsort(dim=-1, descending=True)
        selected = torch.zeros_like(router_logits, dtype=torch.bool)
        base_loss = query.new_zeros(())
        ug_loss = query.new_zeros(())
        cmi_loss = query.new_zeros(())

        # Interaction residuals use the paired retrieval views as the two
        # modalities. This works for image-text and text-image MMEB pairs
        # without requiring backbone-specific extra forward passes.
        interaction = q_groups - p_groups
        for step in range(self.num_groups):
            available = ~selected
            masked_logits = router_logits.masked_fill(~available, torch.finfo(router_logits.dtype).min)
            probabilities = F.softmax(masked_logits, dim=-1)

            # Utility targets are detached: they teach the router but do not
            # backpropagate through every candidate InfoNCE computation.
            with torch.no_grad():
                current_mask = selected.unsqueeze(-1).to(query.dtype)
                current = self._infonce(q_groups, all_p_groups, current_mask, positives, reduction="none")
                candidate_losses = []
                for group in range(self.num_groups):
                    candidate = selected.clone()
                    candidate[:, group] = candidate[:, group] | available[:, group]
                    candidate_losses.append(self._infonce(q_groups, all_p_groups, candidate.unsqueeze(-1).to(query.dtype), positives, reduction="none"))
                utilities = torch.stack([current - value for value in candidate_losses], dim=-1)
                utility_target = F.softmax(utilities / self.utility_temperature, dim=-1)
                utility_target = utility_target * available
                utility_target = utility_target / utility_target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            ug_loss = ug_loss + -(utility_target * F.log_softmax(masked_logits, dim=-1)).sum(dim=-1).mean()

            # Penalize router mass on redundant residual interaction groups.
            residual = interaction - (selected.unsqueeze(-1) * interaction).sum(1, keepdim=True) / selected.sum(1, keepdim=True).clamp_min(1).unsqueeze(-1)
            residual = F.normalize(residual.float(), dim=-1)
            redundancy = residual @ residual.transpose(1, 2)
            redundancy = redundancy.square() * (1 - torch.eye(self.num_groups, device=query.device)).unsqueeze(0)
            cmi_loss = cmi_loss + torch.einsum("bi,bij,bj->b", probabilities.float(), redundancy, probabilities.float()).mean()

            # Do not update ``selected`` in place. ``masked_logits`` retains
            # this boolean mask for its backward pass, and an in-place scatter
            # would trigger PyTorch's tensor-versioning autograd error.
            next_selected = selected.scatter(1, order[:, step : step + 1], True)
            # The selected prefix is nested by construction.
            base_loss = base_loss + self._infonce(q_groups, all_p_groups, next_selected.unsqueeze(-1).to(query.dtype), positives)
            selected = next_selected

        base_loss = base_loss / self.num_groups
        ug_loss = ug_loss / self.num_groups
        cmi_loss = cmi_loss / self.num_groups
        use_ug = self.loss_type in {"cms_mrl", "cms_ug"}
        use_cmi = self.loss_type in {"cms_mrl", "cms_cmi"}
        total = base_loss + (self.utility_weight * ug_loss if use_ug else 0) + (self.cmi_weight * cmi_loss if use_cmi else 0)
        return {"loss": total, "contrastive_loss": base_loss, "cms_base_loss": base_loss.detach(), "cms_ug_loss": ug_loss.detach(), "cms_cmi_loss": cmi_loss.detach(), "cms_group_width": torch.tensor(width, device=query.device)}
