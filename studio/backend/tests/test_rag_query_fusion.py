# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""RAG-Fusion tests: query expansion + multi-variant RRF wired into
``retrieval.retrieve_hybrid(mode="fusion")`` and the tool dispatcher.

Adapted from "RAG-Fusion: a New Take on Retrieval-Augmented Generation"
(Rackauckas, 2024 — arXiv:2402.03367).
"""

import math

import pytest

from core.rag import config, query_fusion, retrieval, store

VOCAB = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]


def _embed(text):
    v = [float(text.lower().count(w)) for w in VOCAB]
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


@pytest.fixture
def bow_embeddings(monkeypatch):
    """Bag-of-words embedder matching the vectors stored in the db."""
    from core.rag import embeddings

    monkeypatch.setattr(
        embeddings,
        "encode",
        lambda texts, *, model_name = None, normalize = True: [_embed(t) for t in texts],
    )
    monkeypatch.setattr(embeddings, "dim", lambda model_name = None: len(VOCAB))


def _add_doc(conn, scope, doc_id, filename, sha, text, page = None):
    from core.rag.chunking import Chunk

    store.create_document(
        conn, scope = scope, filename = filename, sha256 = sha, document_id = doc_id
    )
    chunk = Chunk(
        text = text,
        token_count = len(text.split()),
        page_number = page,
        source_page_index = 0,
        chunk_index = 0,
        page_char_start = 0,
        page_char_end = len(text),
    )
    store.add_chunks(conn, scope, doc_id, [chunk], [_embed(text)])


def test_expand_query_leads_with_original_and_dedups():
    variants = query_fusion.expand_query("How does alpha work?")
    assert variants[0] == "How does alpha work?"  # original always first
    # No duplicate phrasings survive (case-insensitive).
    lowered = [v.lower() for v in variants]
    assert len(lowered) == len(set(lowered))


def test_expand_query_splits_compound_question():
    variants = query_fusion.expand_query("alpha bravo and charlie delta")
    assert "alpha bravo" in variants and "charlie delta" in variants


def test_expand_query_emits_keyword_only_variant():
    variants = query_fusion.expand_query("what is the alpha bravo charlie")
    # Stopwords stripped into a keyword-dense reformulation.
    assert "alpha bravo charlie" in variants


def test_expand_query_respects_max_variants():
    variants = query_fusion.expand_query("alpha and bravo, charlie or delta", max_variants = 2)
    assert len(variants) == 2


def test_retrieve_fusion_fuses_variant_rankings(monkeypatch):
    # Each variant fans into lexical + dense rankings, all fed to one _rrf call.
    monkeypatch.setattr(retrieval, "retrieve_lexical", lambda c, s, q, k = None: [])
    monkeypatch.setattr(
        retrieval, "retrieve_dense", lambda c, s, q, k = None, *, model_name = None: []
    )
    seen = {}
    monkeypatch.setattr(
        retrieval,
        "_rrf",
        lambda rankings, rrf_k, top_k: seen.update(
            n = len(rankings), rrf_k = rrf_k, top_k = top_k
        )
        or [],
    )
    query_fusion.retrieve_fusion(
        None, "kb_a", "alpha and bravo", k = 7, generate_variants = lambda q: ["v1", "v2", "v3"]
    )
    assert seen["n"] == 6  # 3 variants x (lexical + dense)
    assert seen["rrf_k"] == config.RRF_K and seen["top_k"] == 7


def test_retrieve_fusion_falls_back_when_generator_empty(monkeypatch):
    # An empty generator must still retrieve for the raw query, not zero queries.
    calls = []
    monkeypatch.setattr(
        retrieval, "retrieve_lexical", lambda c, s, q, k = None: calls.append(q) or []
    )
    monkeypatch.setattr(
        retrieval, "retrieve_dense", lambda c, s, q, k = None, *, model_name = None: []
    )
    monkeypatch.setattr(retrieval, "_rrf", lambda rankings, rrf_k, top_k: [])
    query_fusion.retrieve_fusion(None, "kb_a", "alpha", generate_variants = lambda q: [])
    assert calls == ["alpha"]


def test_retrieve_hybrid_fusion_mode_dispatches(monkeypatch):
    # The call-site edit: mode="fusion" routes through retrieve_fusion.
    seen = {}

    def fake_fusion(conn, scope, query, *, k = None, model_name = None):
        seen.update(query = query, k = k)
        return [retrieval.Hit("d1:0", 1.0)]

    monkeypatch.setattr(query_fusion, "retrieve_fusion", fake_fusion)
    hits = retrieval.retrieve_hybrid(None, "kb_a", "alpha bravo", k = 5, mode = "fusion")
    assert seen == {"query": "alpha bravo", "k": 5}
    assert hits[0].chunk_id == "d1:0"


def test_fusion_mode_recovers_chunk_a_single_phrasing_misses(rag_conn, bow_embeddings):
    # End-to-end: a compound query whose halves each hit a different doc. Fusion
    # surfaces both; the keyword/clause variants give recall a single query lacks.
    _add_doc(rag_conn, "kb_a", "d1", "f1", "h1", "alpha bravo")
    _add_doc(rag_conn, "kb_a", "d2", "f2", "h2", "golf hotel")
    hits = retrieval.retrieve_hybrid(
        rag_conn, "kb_a", "alpha bravo and golf hotel", k = 5, mode = "fusion"
    )
    ids = {h.chunk_id for h in hits}
    assert {"d1:0", "d2:0"} <= ids


def test_tool_dispatcher_accepts_fusion_mode(monkeypatch):
    from core.inference import tools
    from storage import rag_db

    from core.rag import tool

    monkeypatch.setattr(rag_db, "RAG_AVAILABLE", True, raising = False)
    seen = {}

    def fake_search(**kw):
        seen.update(kw)
        return ("text", [])

    monkeypatch.setattr(tool, "search_knowledge_base_with_sources", fake_search)
    tools._search_knowledge_base({"query": "q"}, {"kb_id": "a", "mode": "fusion"})
    assert seen["mode"] == "fusion"  # no longer coerced back to hybrid
