"""Simple CLI app to chat and experiment with prompt variants and RAG."""
import argparse
from typing import List, Optional

from .guardrails import annotate_unverified, check_groundedness, check_scope, SCOPE_REFUSAL
from .model_base import FoundationModel, HFModel, LocalDummyModel
from .prompting import build_prompt


def chat_loop(
    model: FoundationModel,
    prompt_name: str,
    rag=None,
    rag_k: int = 3,
    guardrails: bool = True,
    scope_model: Optional[FoundationModel] = None,
):
    print("Starting chat (type 'exit' to quit)")
    while True:
        msg = input("Customer: ").strip()
        if not msg:
            continue
        if msg.lower() in ("exit", "quit"):
            break

        if guardrails and not check_scope(msg, model=scope_model).passed:
            print("Assistant:", SCOPE_REFUSAL)
            continue

        prompt = build_prompt(prompt_name, msg)
        context_chunks: List[str] = []
        if rag:
            hits = rag_search(rag, msg, rag_k)
            if hits:
                context_chunks = hits
                prompt += "\n\nRelevant information:\n" + "\n---\n".join(hits)

        resp = model.generate(prompt)
        if guardrails:
            resp = annotate_unverified(resp, check_groundedness(resp, context_chunks))
        print("Assistant:", resp)


def rag_search(rag, query: str, k: int) -> List[str]:
    """Normalizes results from either SimpleRAG (list[str]) or
    PersistentVectorStore (list[(text, source, score)]) into plain strings."""
    hits = rag.search(query, k=k)
    if hits and isinstance(hits[0], tuple):
        return [text for text, _source, _score in hits]
    return hits


def build_model(name: str, hf_model: str) -> FoundationModel:
    if name == "hf":
        return HFModel(model_name=hf_model)
    return LocalDummyModel()


def main(argv: Optional[list] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["local", "hf"], default="local")
    parser.add_argument("--hf-model", default="distilgpt2", help="Hugging Face model name when --model hf")
    parser.add_argument("--prompt", choices=["default", "concise", "detailed"], default="default")
    parser.add_argument("--use-rag", action="store_true", help="Use a small in-memory RAG demo (ignored if --rag-store is set)")
    parser.add_argument("--rag-store", default=None, help="Path to a persistent vector store built with ingest.py")
    parser.add_argument("--no-guardrails", action="store_true", help="Disable the scope/groundedness guardrails")
    args = parser.parse_args(argv)

    model = build_model(args.model, args.hf_model)

    rag = None
    if args.rag_store:
        from .vector_store import PersistentVectorStore

        try:
            rag = PersistentVectorStore.load(args.rag_store)
        except FileNotFoundError as e:
            print(f"RAG store not found at {args.rag_store} (run ingest.py first): {e}")
    elif args.use_rag:
        try:
            from .rag import SimpleRAG

            rag = SimpleRAG()
            # example documents - replace with real shipment docs
            rag.add("Your shipment is handled by our logistics partner and usually ships in 1-2 days.")
            rag.add("Tracking numbers start with 'TR' followed by digits.")
            rag.build_index()
        except Exception as e:
            print("RAG unavailable:", e)
            rag = None

    scope_model = model if args.model == "hf" else None
    chat_loop(model, args.prompt, rag=rag, guardrails=not args.no_guardrails, scope_model=scope_model)


if __name__ == "__main__":
    main()
