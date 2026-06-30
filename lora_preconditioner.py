"""Riemannian-preconditioned optimizer for LoRA fine-tuning.

Adapted from "Riemannian Preconditioned LoRA for Fine-Tuning Foundation
Models" (Zhang & Pilanci, https://arxiv.org/abs/2402.02347).

LoRA reparameterises a weight update as ``ΔW = (alpha / r) · B @ A`` where
``A`` (``lora_A``) is ``r × k`` and ``B`` (``lora_B``) is ``d × r``. Plain
optimisers treat ``A`` and ``B`` as ordinary tensors, but the loss is
invariant to the transform ``A → G⁻¹A``, ``B → BG`` for any invertible
``r × r`` ``G`` — so the Euclidean gradient is a poor search direction. The
paper introduces an ``r × r`` preconditioner per LoRA pair that scales each
step by the inverse Gram matrix of the *other* factor:

    precond(grad_A) = (Bᵀ B + λI)⁻¹ @ grad_A
    precond(grad_B) = grad_B @ (A Aᵀ + λI)⁻¹

This is the closed-form Riemannian gradient on the low-rank manifold and is
shown to stabilise feature learning and speed up convergence.

This module provides that preconditioner as a drop-in ``AdamW`` variant plus
a small builder that the training CLI calls when ``--use_riemannian`` is set.
``torch`` is imported lazily so the module (and its pure pair-detection
helpers) can be imported and unit-tested without a torch install.
"""

from __future__ import annotations

from typing import Any, Iterable, NamedTuple, Optional

# Markers PEFT uses in the qualified parameter name of each LoRA factor.
_LORA_A_MARKER = ".lora_A."
_LORA_B_MARKER = ".lora_B."


class LoRAPair(NamedTuple):
    """A matched ``lora_A`` / ``lora_B`` factor pair for one adapted module."""

    key: str
    lora_A: Any
    lora_B: Any


def _pair_key(name: str) -> Optional[str]:
    """Collapse a LoRA factor's parameter name to a key shared by its partner.

    ``...q_proj.lora_A.default.weight`` and ``...q_proj.lora_B.default.weight``
    both map to ``...q_proj.<lora>.default.weight`` so the two factors of one
    adapted module group together. Returns ``None`` for non-LoRA parameters.
    """
    if _LORA_A_MARKER in name:
        return name.replace(_LORA_A_MARKER, ".<lora>.")
    if _LORA_B_MARKER in name:
        return name.replace(_LORA_B_MARKER, ".<lora>.")
    return None


def find_lora_pairs(named_parameters: Iterable[tuple[str, Any]]) -> list[LoRAPair]:
    """Group trainable parameters into matched ``lora_A`` / ``lora_B`` pairs.

    ``named_parameters`` is any iterable of ``(name, param)`` such as
    ``model.named_parameters()``. Parameters that are not LoRA factors, or
    LoRA factors whose partner is missing, are skipped. Pure string/attribute
    logic — no torch required, which keeps it unit-testable.
    """
    a_factors: dict[str, Any] = {}
    b_factors: dict[str, Any] = {}
    for name, param in named_parameters:
        if not getattr(param, "requires_grad", False):
            continue
        key = _pair_key(name)
        if key is None:
            continue
        if _LORA_A_MARKER in name:
            a_factors[key] = param
        else:
            b_factors[key] = param

    pairs = []
    for key in a_factors:
        if key in b_factors:
            pairs.append(LoRAPair(key=key, lora_A=a_factors[key], lora_B=b_factors[key]))
    return pairs


def _load_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without torch
        raise ImportError(
            "Riemannian-preconditioned LoRA training requires torch. "
            "Install the training extras to use --use_riemannian."
        ) from exc
    return torch


def build_lora_optimizers(
    model: Any,
    *,
    enabled: bool,
    learning_rate: float,
    weight_decay: float = 0.0,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    reg: float = 1e-6,
) -> Optional[tuple[Any, None]]:
    """Build the ``(optimizer, lr_scheduler)`` tuple a Trainer expects.

    Returns ``None`` when ``enabled`` is false (so the caller keeps the
    Trainer's default optimiser and nothing changes), and otherwise a
    ``(RiemannianPreconditionedAdamW, None)`` tuple. Passing ``None`` for the
    scheduler lets the Trainer build its configured schedule on top of our
    optimiser.
    """
    if not enabled:
        return None

    pairs = find_lora_pairs(model.named_parameters())
    if not pairs:
        raise ValueError(
            "--use_riemannian was set but no trainable lora_A/lora_B parameter "
            "pairs were found. Did you call get_peft_model() first?"
        )

    paired_ids = set()
    for pair in pairs:
        paired_ids.add(id(pair.lora_A))
        paired_ids.add(id(pair.lora_B))
    other_params = [
        param
        for _, param in model.named_parameters()
        if getattr(param, "requires_grad", False) and id(param) not in paired_ids
    ]

    optimizer_cls = _make_optimizer_base()
    optimizer = optimizer_cls(
        pairs,
        other_params,
        lr=learning_rate,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        reg=reg,
    )
    return optimizer, None


def _make_optimizer_base():
    """Return the ``RiemannianPreconditionedAdamW`` class, defined lazily so the
    module imports without torch."""
    torch = _load_torch()

    class _RiemannianPreconditionedAdamW(torch.optim.AdamW):
        """AdamW that left/right-multiplies each LoRA factor's gradient by the
        inverse Gram matrix of its partner before the standard AdamW step.

        ``lora_pairs`` are the matched factors that get preconditioned;
        ``other_params`` (biases, embeddings, ...) take an ordinary AdamW step.
        """

        def __init__(self, lora_pairs, other_params, *, reg=1e-6, **adamw_kwargs):
            self._lora_pairs = list(lora_pairs)
            self._reg = reg
            params = []
            for pair in self._lora_pairs:
                params.append(pair.lora_A)
                params.append(pair.lora_B)
            params.extend(other_params)
            super().__init__(params, **adamw_kwargs)

        @torch.no_grad()
        def step(self, closure=None):
            self._apply_preconditioner()
            return super().step(closure)

        def _apply_preconditioner(self):
            torch_mod = torch
            eye_cache = {}
            for pair in self._lora_pairs:
                a, b = pair.lora_A, pair.lora_B
                if a.grad is None or b.grad is None:
                    continue
                rank = a.shape[0]
                eye = eye_cache.get((rank, a.dtype, a.device))
                if eye is None:
                    eye = torch_mod.eye(rank, dtype=a.dtype, device=a.device)
                    eye_cache[(rank, a.dtype, a.device)] = eye

                # grad_A <- (BᵀB + λI)⁻¹ grad_A
                gram_b = b.transpose(0, 1) @ b + self._reg * eye
                a.grad.copy_(torch_mod.linalg.solve(gram_b, a.grad))

                # grad_B <- grad_B (AAᵀ + λI)⁻¹
                gram_a = a @ a.transpose(0, 1) + self._reg * eye
                solved = torch_mod.linalg.solve(gram_a, b.grad.transpose(0, 1))
                b.grad.copy_(solved.transpose(0, 1))

    return _RiemannianPreconditionedAdamW


def __getattr__(name):
    # Expose RiemannianPreconditionedAdamW only when torch is available, so the
    # module import itself stays torch-free.
    if name == "RiemannianPreconditionedAdamW":
        return _make_optimizer_base()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
