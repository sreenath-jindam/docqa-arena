"""The golden set is the fixed thing everything else is measured against.

If it drifts from the corpus, two sweeps stop being comparable and nobody
notices, so these tests treat any drift as a build failure.
"""
import pytest

from eval.spans import load_golden, relevant_chunk_ids, resolve_spans
from src.chunkers import FixedSizeChunker, RecursiveChunker
from src.config import REPO_ROOT
from src.corpus import load_corpus, load_document_map

GOLDEN = REPO_ROOT / "eval" / "golden.csv"
CORPUS = REPO_ROOT / "data" / "corpus"


@pytest.fixture(scope="module")
def documents():
    return load_document_map(CORPUS)


@pytest.fixture(scope="module")
def examples(documents):
    return resolve_spans(load_golden(GOLDEN), documents)


def test_every_evidence_snippet_resolves(examples):
    # resolve_spans raises on drift, so reaching here means all rows resolved.
    assert len(examples) >= 30


def test_ids_are_unique(examples):
    ids = [e.id for e in examples]
    assert len(ids) == len(set(ids))


def test_spans_point_at_the_evidence(examples, documents):
    for example in examples:
        doc = documents[example.document_id]
        for span in example.spans:
            extracted = doc.content[span.char_start:span.char_end]
            assert extracted.strip(), f"{example.id} resolved to an empty span"


def test_every_question_has_a_relevant_chunk_under_both_chunkers(examples, documents):
    """The label must survive the chunking strategy.

    This is the check that makes an 18-config comparison meaningful: if a
    question has relevant chunks under fixed-size chunking but none under
    recursive, the two configurations are not being scored on the same thing.
    """
    docs = list(load_corpus(CORPUS))
    for chunker in (FixedSizeChunker(800, 120), RecursiveChunker(800, 120)):
        chunks = [c for doc in docs for c in chunker.chunk(doc)]
        for example in examples:
            assert relevant_chunk_ids(example, chunks), f"{example.id} has no relevant chunk under {chunker.name}"


def test_multi_hop_questions_have_two_spans(examples):
    multi = [e for e in examples if e.query_type == "multi-hop"]
    assert multi, "the golden set should contain at least one multi-hop question"
    for example in multi:
        assert len(example.spans) >= 2
