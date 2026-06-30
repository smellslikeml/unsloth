# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Tests for the Riemannian-preconditioned LoRA optimizer and its wiring into
the ``unsloth-cli.py`` training script.

Paper: "Riemannian Preconditioned LoRA for Fine-Tuning Foundation Models"
(https://arxiv.org/abs/2402.02347).

The pure pair-detection / opt-in gating tests run without torch. The
optimizer-math test is gated on a torch install (skips on the no-torch
sandbox, like the rest of the GPU suite).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import lora_preconditioner as lp  # noqa: E402


class _FakeParam:
    """Stand-in for an nn.Parameter for the torch-free pair-detection tests."""

    def __init__(self, requires_grad=True, shape=(8, 16)):
        self.requires_grad = requires_grad
        self.shape = shape


def _peft_named_params():
    base = "base_model.model.model.layers.0.self_attn.q_proj"
    return [
        (f"{base}.lora_A.default.weight", _FakeParam(shape=(8, 16))),
        (f"{base}.lora_B.default.weight", _FakeParam(shape=(16, 8))),
        # A frozen base weight that must never be preconditioned.
        (f"{base}.base_layer.weight", _FakeParam(requires_grad=False)),
        # A trainable non-LoRA param (e.g. a bias) -> not a pair.
        ("base_model.model.lm_head.bias", _FakeParam(shape=(32,))),
    ]


def test_pair_key_collapses_a_and_b_to_same_key():
    a = "x.q_proj.lora_A.default.weight"
    b = "x.q_proj.lora_B.default.weight"
    assert lp._pair_key(a) == lp._pair_key(b)
    assert lp._pair_key("x.q_proj.base_layer.weight") is None


def test_find_lora_pairs_matches_only_trainable_factor_pairs():
    pairs = lp.find_lora_pairs(_peft_named_params())
    assert len(pairs) == 1
    pair = pairs[0]
    # The A factor is r×k (rank-first); B is d×r.
    assert pair.lora_A.shape == (8, 16)
    assert pair.lora_B.shape == (16, 8)


def test_build_lora_optimizers_is_opt_in_and_returns_none_when_disabled():
    # Disabled is the default path: returns None so the caller keeps the
    # Trainer's configured --optim untouched (no behavior change).
    assert lp.build_lora_optimizers(object(), enabled=False, learning_rate=2e-4) is None


def test_callsite_wires_the_preconditioner_into_unsloth_cli():
    """The integration proof: the training CLI must actually invoke the
    builder and forward its result to SFTTrainer. unsloth-cli.py imports torch
    at call time so it cannot be imported here; assert the wiring at source
    level instead."""
    source = (REPO_ROOT / "unsloth-cli.py").read_text()
    assert "from lora_preconditioner import build_lora_optimizers" in source
    assert "build_lora_optimizers(" in source
    assert "enabled = args.use_riemannian" in source
    # The built optimizer pair is forwarded to the trainer only when present.
    assert 'trainer_kwargs["optimizers"] = optimizers' in source
    # The opt-in flag is exposed on the CLI.
    assert '"--use_riemannian"' in source


@pytest.mark.server
def test_preconditioner_transforms_gradient_when_torch_available():
    torch = pytest.importorskip("torch")

    torch.manual_seed(0)
    rank, k, d = 4, 6, 5
    lora_A = torch.nn.Parameter(torch.randn(rank, k))
    lora_B = torch.nn.Parameter(torch.randn(d, rank))

    class _FakeModel:
        def named_parameters(self):
            base = "base_model.model.layers.0.q_proj"
            yield f"{base}.lora_A.default.weight", lora_A
            yield f"{base}.lora_B.default.weight", lora_B

    optimizers = lp.build_lora_optimizers(
        _FakeModel(), enabled=True, learning_rate=1e-2, weight_decay=0.0
    )
    assert optimizers is not None
    optimizer, scheduler = optimizers
    assert scheduler is None

    # Seed deterministic gradients and capture them before preconditioning.
    grad_A = torch.randn(rank, k)
    grad_B = torch.randn(d, rank)
    lora_A.grad = grad_A.clone()
    lora_B.grad = grad_B.clone()

    # Expected preconditioned gradients per the paper's closed form.
    reg = 1e-6
    eye = torch.eye(rank)
    expected_A = torch.linalg.solve(lora_B.t() @ lora_B + reg * eye, grad_A)
    expected_B = torch.linalg.solve(lora_A @ lora_A.t() + reg * eye, grad_B.t()).t()

    optimizer._apply_preconditioner()

    assert torch.allclose(lora_A.grad, expected_A, atol=1e-4)
    assert torch.allclose(lora_B.grad, expected_B, atol=1e-4)
    # Preconditioning genuinely changes the search direction.
    assert not torch.allclose(lora_A.grad, grad_A, atol=1e-4)
