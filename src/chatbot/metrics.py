"""Lightweight text-similarity metrics for evaluating chatbot responses.

Pure-Python ROUGE-1/ROUGE-L F1 (no extra dependency) plus simple keyword
coverage, used by evaluate.py to score responses against reference answers.
"""
from collections import Counter
from typing import List


def _tokenize(text: str) -> List[str]:
    return text.lower().split()


def keyword_coverage(response: str, keywords: List[str]) -> float:
    if not keywords:
        return 0.0
    text = response.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text)
    return hits / len(keywords)


def rouge1_f1(response: str, reference: str) -> float:
    """Unigram overlap F1 between response and reference."""
    cand, ref = _tokenize(response), _tokenize(reference)
    if not cand or not ref:
        return 0.0
    overlap = sum((Counter(cand) & Counter(ref)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(cand)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: List[str], b: List[str]) -> int:
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def rougeL_f1(response: str, reference: str) -> float:
    """Longest-common-subsequence F1 between response and reference."""
    cand, ref = _tokenize(response), _tokenize(reference)
    if not cand or not ref:
        return 0.0
    lcs = _lcs_length(cand, ref)
    if lcs == 0:
        return 0.0
    precision = lcs / len(cand)
    recall = lcs / len(ref)
    return 2 * precision * recall / (precision + recall)
