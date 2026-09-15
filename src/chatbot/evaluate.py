"""Evaluation harness for comparing prompt templates and RAG settings.

Runs a fixed set of shipment-support test queries against every
(prompt, rag on/off) combination for a chosen model backend, scores each
response by expected-keyword coverage, and reports latency/length too.

Usage:
    python -m src.chatbot.evaluate --model local
    python -m src.chatbot.evaluate --model hf --hf-model distilgpt2 \\
        --rag-store data/vector_store --rag-modes off,on
"""
import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List, Optional

from rich.console import Console
from rich.table import Table

from .app import build_model, rag_search
from .guardrails import check_groundedness, check_scope
from .metrics import keyword_coverage, rouge1_f1, rougeL_f1
from .model_base import FoundationModel
from .prompting import build_prompt

ADVERSARIAL_QUERIES = [
    "What's the capital of France?",
    "Write me a poem about the ocean.",
    "Can you give me investment advice?",
    "Tell me a joke.",
]

TEST_CASES = [
    {
        "query": "Where is my package right now?",
        "expected_keywords": ["transit", "tracking"],
        "expected_answer": (
            "Your package is currently in transit and should arrive within the "
            "estimated delivery window. You can check its real-time location "
            "using your tracking number."
        ),
    },
    {
        "query": "How can I track my order?",
        "expected_keywords": ["tracking number", "track"],
        "expected_answer": (
            "You can track your order using the tracking number sent in your "
            "shipping confirmation email. Enter it on the carrier's website to "
            "see real-time status updates."
        ),
    },
    {
        "query": "My order is really late, what happened?",
        "expected_keywords": ["delay", "escalate"],
        "expected_answer": (
            "I'm sorry your order is late. If it's outside the expected "
            "delivery window I can escalate this to our logistics team to "
            "investigate right away."
        ),
    },
    {
        "query": "Do you ship internationally?",
        "expected_keywords": ["international", "customs"],
        "expected_answer": (
            "Yes, we ship to most countries worldwide. International orders "
            "typically take 7-14 business days and you may be responsible for "
            "customs duties or import fees."
        ),
    },
    {
        "query": "My package arrived damaged, what do I do?",
        "expected_keywords": ["damaged", "replacement", "refund"],
        "expected_answer": (
            "I'm sorry to hear that. Please send photos of the damaged item "
            "and packaging within 7 days of delivery, and we'll arrange a free "
            "replacement or a full refund."
        ),
    },
    {
        "query": "How long does standard shipping take?",
        "expected_keywords": ["business days", "ship"],
        "expected_answer": (
            "Standard shipping usually takes 3-5 business days after your "
            "order ships, with most orders leaving our warehouse within 1-2 "
            "business days."
        ),
    },
]


def load_rag(store_path: Optional[str]):
    if not store_path:
        return None
    from .vector_store import PersistentVectorStore

    return PersistentVectorStore.load(store_path)


def run_case(model: FoundationModel, prompt_name: str, query: str, rag=None, k: int = 3):
    prompt = build_prompt(prompt_name, query)
    context_chunks: List[str] = []
    if rag is not None:
        hits = rag_search(rag, query, k)
        if hits:
            context_chunks = hits
            prompt += "\n\nRelevant information:\n" + "\n---\n".join(hits)

    start = time.perf_counter()
    response = model.generate(prompt)
    latency = time.perf_counter() - start
    return response, latency, context_chunks


def run_evaluation(model: FoundationModel, prompt_names: List[str], rag_modes: List[bool], rag=None) -> List[Dict]:
    rows = []
    for prompt_name in prompt_names:
        for rag_on in rag_modes:
            active_rag = rag if rag_on else None
            coverages, rouge1s, rougeLs, groundeds, latencies, lengths = [], [], [], [], [], []
            for case in TEST_CASES:
                response, latency, context_chunks = run_case(
                    model, prompt_name, case["query"], rag=active_rag
                )
                coverages.append(keyword_coverage(response, case["expected_keywords"]))
                rouge1s.append(rouge1_f1(response, case["expected_answer"]))
                rougeLs.append(rougeL_f1(response, case["expected_answer"]))
                groundeds.append(1.0 if check_groundedness(response, context_chunks).passed else 0.0)
                latencies.append(latency)
                lengths.append(len(response))
            rows.append(
                {
                    "prompt": prompt_name,
                    "rag": "on" if rag_on else "off",
                    "avg_keyword_coverage": sum(coverages) / len(coverages),
                    "avg_rouge1_f1": sum(rouge1s) / len(rouge1s),
                    "avg_rougeL_f1": sum(rougeLs) / len(rougeLs),
                    "avg_groundedness_pass_rate": sum(groundeds) / len(groundeds),
                    "avg_latency_sec": sum(latencies) / len(latencies),
                    "avg_response_len": sum(lengths) / len(lengths),
                    "n_cases": len(TEST_CASES),
                }
            )
    return rows


def run_scope_report(model: FoundationModel, scope_model: Optional[FoundationModel] = None) -> Dict:
    """Scope guardrail is independent of prompt/RAG, so it's evaluated once
    against the in-domain test set (false-positive rate) and a separate
    adversarial set (catch rate) rather than swept per config."""
    in_domain_passes = sum(
        1 for case in TEST_CASES if check_scope(case["query"], model=scope_model).passed
    )
    adversarial_blocks = sum(
        1 for q in ADVERSARIAL_QUERIES if not check_scope(q, model=scope_model).passed
    )
    return {
        "false_positive_rate": 1 - (in_domain_passes / len(TEST_CASES)),
        "catch_rate": adversarial_blocks / len(ADVERSARIAL_QUERIES),
        "n_in_domain": len(TEST_CASES),
        "n_adversarial": len(ADVERSARIAL_QUERIES),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["local", "hf"], default="local")
    parser.add_argument("--hf-model", default="distilgpt2")
    parser.add_argument("--rag-store", default=None, help="Path to a persisted vector store from ingest.py")
    parser.add_argument("--prompts", default="default,concise,detailed")
    parser.add_argument("--rag-modes", default="off,on", help="Comma list of off/on")
    parser.add_argument("--out", default="results/eval_results.csv")
    parser.add_argument("--no-guardrails", action="store_true", help="Skip the scope guardrail report")
    args = parser.parse_args(argv)

    prompt_names = [p.strip() for p in args.prompts.split(",") if p.strip()]
    rag_modes = [m.strip().lower() == "on" for m in args.rag_modes.split(",") if m.strip()]

    rag = None
    if any(rag_modes):
        if args.rag_store:
            rag = load_rag(args.rag_store)
        else:
            print("Warning: RAG 'on' requested but no --rag-store given; skipping RAG-on runs")
            rag_modes = [m for m in rag_modes if not m]

    model = build_model(args.model, args.hf_model)
    rows = run_evaluation(model, prompt_names, rag_modes, rag=rag)

    console = Console()
    table = Table(title="Prompt/RAG Evaluation Results")
    columns = [
        "prompt",
        "rag",
        "avg_keyword_coverage",
        "avg_rouge1_f1",
        "avg_rougeL_f1",
        "avg_groundedness_pass_rate",
        "avg_latency_sec",
        "avg_response_len",
        "n_cases",
    ]
    for col in columns:
        table.add_column(col)
    for row in rows:
        table.add_row(
            row["prompt"],
            row["rag"],
            f"{row['avg_keyword_coverage']:.2f}",
            f"{row['avg_rouge1_f1']:.2f}",
            f"{row['avg_rougeL_f1']:.2f}",
            f"{row['avg_groundedness_pass_rate']:.2f}",
            f"{row['avg_latency_sec']:.3f}",
            f"{row['avg_response_len']:.0f}",
            str(row["n_cases"]),
        )
    console.print(table)

    if not args.no_guardrails:
        scope_model = model if args.model == "hf" else None
        report = run_scope_report(model, scope_model=scope_model)
        console.print(
            f"\nScope guardrail: false-positive rate "
            f"{report['false_positive_rate']:.0%} on {report['n_in_domain']} in-domain queries, "
            f"catch rate {report['catch_rate']:.0%} on {report['n_adversarial']} adversarial queries"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
