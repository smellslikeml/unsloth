# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Multi-query fusion retrieval (RAG-Fusion).

RAG-Fusion (Rackauckas, 2024 — arXiv:2402.03367) extends plain RAG by expanding
one user query into several related queries, retrieving for each, and fusing the
per-query rankings with Reciprocal Rank Fusion. The intuition: a single phrasing
under-covers the relevant chunks, so reformulations surface documents the
original misses, and RRF rewards chunks that recur across reformulations.

This module reuses the existing ``retrieval._rrf`` fuser (the algorithm the repo
already runs over the lexical/dense pair) and only adds the query-expansion +
fan-out step. The LLM-based query generator from the paper is intentionally left
pluggable via ``generate_variants`` so this stays a torch-free, deterministic
default; the bundled :func:`expand_query` is rule-based and needs no model call.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Callable

from . import config, retrieval

# Small, dependency-free stopword list for the keyword-only reformulation. Kept
# deliberately short — just the high-frequency function words that add no lexical
# signal to FTS5 / dense matching.
_STOPWORDS = frozenset(
    """
    a an the and or but of to in on at for with by from as is are was were be been
    being do does did how what why when where which who whom this that these those
    i you he she it we they me him her us them my your his its our their can could
    should would may might will shall about into over under than then so such
    """.split()
)

# Connectors that usually join independent information needs in one question.
_CLAUSE_SPLIT = re.compile(r"\?|;|,| and | or | versus | vs\.? ", flags = re.IGNORECASE)
_WORD = re.compile(r"[A-Za-z0-9]+")


def _norm(text: str) -> str:
    """Collapse whitespace; the comparison key for de-duplicating variants."""
    return " ".join(text.split()).strip()


def _keyword_variant(query: str) -> str | None:
    """Drop stopwords, keeping order. Returns ``None`` if nothing distinct is left
    (e.g. the query was already keyword-dense) so we don't emit a duplicate."""
    kept = [w for w in _WORD.findall(query) if w.lower() not in _STOPWORDS]
    if not kept:
        return None
    variant = " ".join(kept)
    return variant if _norm(variant).lower() != _norm(query).lower() else None


def _clause_variants(query: str) -> list[str]:
    """Split a compound question into its independent sub-queries."""
    parts = [_norm(p) for p in _CLAUSE_SPLIT.split(query)]
    # Keep only clauses with real content; a lone stopword fragment is noise.
    return [p for p in parts if len(_WORD.findall(p)) >= 2]


def expand_query(query: str, *, max_variants: int | None = None) -> list[str]:
    """Rule-based query expansion used when no LLM generator is supplied.

    Produces, in order: the original query, its standalone sub-clauses, then a
    stopword-stripped keyword form. Variants are de-duplicated case-insensitively
    and the original always leads, so single-phrasing queries simply fall back to
    one-query retrieval. Capped at ``max_variants`` (default ``config.FUSION_MAX_VARIANTS``).
    """
    max_variants = max_variants if max_variants is not None else config.FUSION_MAX_VARIANTS
    candidates = [_norm(query)]
    candidates.extend(_clause_variants(query))
    kw = _keyword_variant(query)
    if kw:
        candidates.append(kw)

    seen: set[str] = set()
    out: list[str] = []
    for c in candidates:
        key = c.lower()
        if c and key not in seen:
            seen.add(key)
            out.append(c)
        if len(out) >= max(1, max_variants):
            break
    return out


def retrieve_fusion(
    conn: sqlite3.Connection,
    scope: str | list[str],
    query: str,
    *,
    k: int | None = None,
    model_name: str | None = None,
    generate_variants: Callable[[str], list[str]] | None = None,
    max_variants: int | None = None,
) -> list[retrieval.Hit]:
    """RAG-Fusion retrieval: expand ``query`` into variants, run hybrid retrieval
    per variant, and RRF-fuse every resulting ranking into one list.

    ``generate_variants`` may be any callable mapping a query to a list of query
    strings (e.g. an LLM reformulator); it defaults to the rule-based
    :func:`expand_query`. Candidate pool sizes and the RRF constant come from
    ``config``, matching the single-query ``retrieve_hybrid`` path.
    """
    k = k if k is not None else config.TOP_K_HYBRID
    k = int(k)
    if generate_variants is None:
        variants = expand_query(query, max_variants = max_variants)
    else:
        variants = [v for v in generate_variants(query) if v and v.strip()]
        if not variants:
            variants = [query]

    rankings: list[list[retrieval.Hit]] = []
    for variant in variants:
        lexical = retrieval.retrieve_lexical(conn, scope, variant, config.TOP_K_LEXICAL)
        dense = retrieval.retrieve_dense(
            conn, scope, variant, config.TOP_K_DENSE, model_name = model_name
        )
        rankings.append(lexical)
        rankings.append(dense)
    return retrieval._rrf(rankings, config.RRF_K, k)
