# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from kvpress.presses.kvzap_press import KVzapPress


@dataclass
class KVzapPrunePress(KVzapPress):
    """
    Ratio-controlled real pruning with KVzap scores.

    KVzap's default DMS usage masks low-scoring head/token pairs. This press averages
    KVzap scores across KV heads to choose one shared token set, then physically shrinks
    the dense KV cache tensors.
    """

    compression_ratio: float = 0.0
    score_aggregation: Literal["mean"] = "mean"
    n_sink: int = 4
    recent_window: int = 128

    def __post_init__(self):
        super().__post_init__()
        assert self.score_aggregation == "mean", "Only mean score aggregation is currently supported"
        assert self.n_sink >= 0, "n_sink must be non-negative"
        assert self.recent_window >= 0, "recent_window must be non-negative"

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.compression_ratio == 0:
            return keys, values

        scores = self.score(module, hidden_states, keys, values, attentions, kwargs)
        token_scores = scores.mean(dim=1)

        bsz, _, k_len, _ = keys.shape
        n_pruned = int(k_len * self.compression_ratio)
        if n_pruned == 0:
            return keys, values

        protected = torch.zeros((bsz, k_len), dtype=torch.bool, device=keys.device)
        if self.n_sink > 0:
            protected[:, : min(self.n_sink, k_len)] = True
        if self.recent_window > 0:
            protected[:, max(k_len - self.recent_window, 0) :] = True

        candidate_count = int((~protected[0]).sum().item())
        n_pruned = min(n_pruned, candidate_count)
        n_kept_candidates = candidate_count - n_pruned

        keep_mask = protected.clone()
        if n_kept_candidates > 0:
            candidate_scores = token_scores.masked_fill(protected, torch.finfo(token_scores.dtype).min)
            kept_candidate_indices = candidate_scores.topk(n_kept_candidates, dim=-1).indices
            keep_mask.scatter_(1, kept_candidate_indices, True)

        indices = torch.stack([torch.where(keep_mask[batch_idx])[0] for batch_idx in range(bsz)])
        indices = indices.sort(dim=-1).values
        gather_indices = indices[:, None, :, None].expand(-1, keys.shape[1], -1, keys.shape[3])

        keys = keys.gather(2, gather_indices).contiguous()
        values = values.gather(2, gather_indices).contiguous()

        return keys, values
