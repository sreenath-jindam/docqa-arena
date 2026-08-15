"""BM25 and RRF fusion, tested without touching Chroma or a real model."""
import pytest

from src.chunkers import RecursiveChunker
from src.retrievers import BM25Retriever, HybridRetriever, tokenize
from src.types import Chunk, RetrievalResult


def make_chunk(cid: str, content: str) -> Chunk:
    return Chunk(cid, "doc.md", content, "test", 0, len(content), len(content.split()))


@pytest.fixture
def chunks():
    return [
        make_chunk("c1", "Card refunds take five to ten business days to settle."),
        make_chunk("c2", "The processing fee is not returned when you refund a charge."),
        make_chunk("c3", "Disputes are decided by the issuing bank within seventy five days."),
        make_chunk("c4", "An invoice is a statement of amounts owed by a customer."),
    ]


def test_tokenize_drops_stopwords_and_punctuation():
    assert tokenize("The refund, of a charge!") == ["refund", "charge"]


def test_bm25_ranks_the_matching_chunk_first(chunks):
    results = BM25Retriever(chunks).retrieve("how long do card refunds take", 3)
    assert results[0].chunk_id == "c1"


def test_bm25_returns_no_more_than_k(chunks):
    assert len(BM25Retriever(chunks).retrieve("refund", 2)) <= 2


def test_bm25_returns_nothing_for_an_unrelated_query(chunks):
    assert BM25Retriever(chunks).retrieve("zebra photosynthesis", 5) == []


def test_bm25_ranks_are_sequential(chunks):
    results = BM25Retriever(chunks).retrieve("refund charge invoice", 4)
    assert [r.rank for r in results] == list(range(1, len(results) + 1))


class FakeVectorRetriever:
    name = "vector"

    def __init__(self, order):
        self.order = order

    def retrieve(self, query, k):
        return [
            RetrievalResult(cid, "doc.md", cid, 0, 1, rank, 1.0 / rank, "vector")
            for rank, cid in enumerate(self.order[:k], start=1)
        ]


def test_rrf_rewards_agreement_between_retrievers(chunks):
    # c2 is second for both retrievers; c1 is first for one and absent from the
    # other. RRF should put the consistently-ranked chunk near the top.
    vector = FakeVectorRetriever(["c1", "c2", "c3"])
    bm25 = BM25Retriever(chunks)
    hybrid = HybridRetriever(vector, bm25)
    results = hybrid.retrieve("processing fee refund charge", 3)
    assert results[0].chunk_id in {"c1", "c2"}
    assert [r.rank for r in results] == list(range(1, len(results) + 1))
    assert all(r.retriever_name == "hybrid" for r in results)


def test_hybrid_deduplicates(chunks):
    hybrid = HybridRetriever(FakeVectorRetriever(["c1", "c2"]), BM25Retriever(chunks))
    ids = [r.chunk_id for r in hybrid.retrieve("refund", 5)]
    assert len(ids) == len(set(ids))


def test_chunker_and_bm25_compose(doc):
    chunks = RecursiveChunker(120, 20).chunk(doc)
    results = BM25Retriever(chunks).retrieve("processing fee", 3)
    assert results
    assert "fee" in results[0].content.lower()
