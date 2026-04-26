# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from kvpress import KVzapPrunePress
from kvpress.presses.kvzap_press import KVzapConfig


class DeterministicKVzapPrunePress(KVzapPrunePress):
    def __init__(self, scores: torch.Tensor, **kwargs):
        super().__init__(**kwargs)
        self._scores = scores

    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        return self._scores.to(device=keys.device, dtype=keys.dtype)


def test_kvzap_prune_press_imported_from_top_level_package():
    assert KVzapPrunePress.__name__ == "KVzapPrunePress"


def test_kvzap_config_can_be_constructed_with_defaults_for_pretrained_loading():
    config = KVzapConfig()

    assert config.input_dim == 0
    assert config.output_dim == 0
    assert config.n_modules == 0


def test_kvzap_prune_press_physically_prunes_cache_by_ratio():
    scores = torch.tensor(
        [
            [
                [0.0, 10.0, 100.0, 9.0, 8.0, 7.0, 6.0, 1.0],
                [0.0, 10.0, -100.0, 9.0, 8.0, 7.0, 6.0, 1.0],
            ]
        ]
    )
    press = DeterministicKVzapPrunePress(scores, compression_ratio=0.5, n_sink=1, recent_window=1)
    module = SimpleNamespace(head_dim=3)
    hidden_states = torch.zeros(1, 8, 6)
    keys = torch.arange(1 * 2 * 8 * 3, dtype=torch.float32).reshape(1, 2, 8, 3)
    values = keys + 1000

    compressed_keys, compressed_values = press.compress(module, hidden_states, keys, values, None, {})

    expected_indices = torch.tensor([0, 1, 2, 7])
    assert compressed_keys.shape == (1, 2, 4, 3)
    assert compressed_values.shape == (1, 2, 4, 3)
    assert torch.equal(compressed_keys, keys[:, :, expected_indices])
    assert torch.equal(compressed_values, values[:, :, expected_indices])


def test_kvzap_prune_press_always_keeps_sink_and_recent_window_tokens():
    scores = torch.tensor(
        [
            [
                [0.0, 0.0, 10.0, 9.0, 8.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 10.0, 9.0, 8.0, 0.0, 0.0, 0.0],
            ]
        ]
    )
    press = DeterministicKVzapPrunePress(scores, compression_ratio=0.5, n_sink=2, recent_window=2)
    module = SimpleNamespace(head_dim=1)
    hidden_states = torch.zeros(1, 8, 4)
    keys = torch.arange(1 * 2 * 8 * 1, dtype=torch.float32).reshape(1, 2, 8, 1)
    values = keys + 1000

    compressed_keys, compressed_values = press.compress(module, hidden_states, keys, values, None, {})

    expected_indices = torch.tensor([0, 1, 6, 7])
    assert compressed_keys.shape == (1, 2, 4, 1)
    assert compressed_values.shape == (1, 2, 4, 1)
    assert torch.equal(compressed_keys, keys[:, :, expected_indices])
    assert torch.equal(compressed_values, values[:, :, expected_indices])
