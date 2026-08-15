"""Metric definitions, pinned by hand-checked values.

nDCG and MRR are easy to get subtly wrong and impossible to notice once they
are averaged over 30 questions, so they get exact expected numbers here.
"""
import math

from eval.metrics import (
    hit_at_k,
    mean,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_precision_counts_hits_over_k():
    assert precision_at_k(["a", "b", "c", "d", "e"], ["a", "c"], 5) == 0.4


def test_precision_divides_by_k_not_by_result_count():
    # Only three results returned but k=5: precision is still out of 5.
    assert precision_at_k(["a", "b", "c"], ["a"], 5) == 0.2


def test_recall_is_over_the_relevant_set():
    assert recall_at_k(["a", "b"], ["a", "c"], 5) == 0.5
    assert recall_at_k(["a", "c"], ["a", "c"], 5) == 1.0


def test_recall_respects_the_cutoff():
    assert recall_at_k(["x", "y", "a"], ["a"], 2) == 0.0


def test_reciprocal_rank_uses_the_first_hit():
    assert reciprocal_rank(["x", "a", "b"], ["a", "b"]) == 0.5
    assert reciprocal_rank(["x", "y"], ["a"]) == 0.0


def test_ndcg_is_one_when_hits_are_at_the_top():
    assert ndcg_at_k(["a", "b", "x"], ["a", "b"], 3) == 1.0


def test_ndcg_discounts_by_position():
    # One relevant item at rank 2: dcg = 1/log2(3), idcg = 1/log2(2) = 1
    expected = (1 / math.log2(3)) / 1.0
    assert abs(ndcg_at_k(["x", "a"], ["a"], 5) - expected) < 1e-9


def test_ndcg_is_zero_with_no_relevant_labels():
    assert ndcg_at_k(["a"], [], 5) == 0.0


def test_hit_at_k_is_binary():
    assert hit_at_k(["x", "a"], ["a"], 2) == 1.0
    assert hit_at_k(["x", "a"], ["a"], 1) == 0.0


def test_percentile_uses_nearest_rank():
    values = [float(v) for v in range(1, 101)]
    assert percentile(values, 50) == 50.0
    assert percentile(values, 95) == 95.0
    assert percentile(values, 100) == 100.0


def test_percentile_and_mean_handle_empty_input():
    assert percentile([], 95) == 0.0
    assert mean([]) == 0.0
