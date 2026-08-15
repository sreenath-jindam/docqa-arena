"""Chunkers must preserve exact character offsets — the retrieval metrics
depend on it, and an off-by-one here silently corrupts every score downstream.
"""
import pytest

from src.chunkers import FixedSizeChunker, RecursiveChunker, SemanticChunker, make_chunk_id
from src.embeddings import HashEmbedder


@pytest.mark.parametrize("chunker_factory", [
    lambda: FixedSizeChunker(200, 40),
    lambda: RecursiveChunker(200, 40),
    lambda: SemanticChunker(HashEmbedder(32), threshold=0.5, chunk_size=200),
])
def test_offsets_index_back_into_the_original(chunker_factory, long_doc):
    for chunk in chunker_factory().chunk(long_doc):
        assert long_doc.content[chunk.char_start:chunk.char_end] == chunk.content


@pytest.mark.parametrize("chunker_factory", [
    lambda: FixedSizeChunker(200, 40),
    lambda: RecursiveChunker(200, 40),
])
def test_whole_document_is_covered(chunker_factory, long_doc):
    chunks = sorted(chunker_factory().chunk(long_doc), key=lambda c: c.char_start)
    assert chunks[0].char_start == 0
    assert chunks[-1].char_end == len(long_doc.content)
    # No gaps: each chunk starts at or before the previous one ended.
    for previous, current in zip(chunks, chunks[1:]):
        assert current.char_start <= previous.char_end


def test_fixed_size_respects_the_window(long_doc):
    for chunk in FixedSizeChunker(200, 40).chunk(long_doc):
        assert chunk.char_end - chunk.char_start <= 200


def test_fixed_size_overlap_is_applied(long_doc):
    chunks = FixedSizeChunker(200, 40).chunk(long_doc)
    assert chunks[1].char_start == 160


def test_fixed_size_rejects_bad_overlap():
    with pytest.raises(ValueError):
        FixedSizeChunker(100, 100)
    with pytest.raises(ValueError):
        FixedSizeChunker(0, 0)


def test_recursive_prefers_heading_boundaries(doc):
    chunks = RecursiveChunker(120, 0).chunk(doc)
    # At least one chunk should begin at a heading rather than mid-sentence.
    assert any(c.content.lstrip().startswith("#") for c in chunks)


def test_chunk_ids_are_deterministic(doc):
    first = RecursiveChunker(200, 40).chunk(doc)
    second = RecursiveChunker(200, 40).chunk(doc)
    assert [c.id for c in first] == [c.id for c in second]


def test_chunk_id_changes_with_strategy():
    assert make_chunk_id("d", 0, 10, "fixed") != make_chunk_id("d", 0, 10, "recursive")


def test_semantic_chunker_splits_somewhere(long_doc):
    chunks = SemanticChunker(HashEmbedder(32), threshold=0.9, chunk_size=300).chunk(long_doc)
    assert len(chunks) > 1
