Customer support chatbot scaffold

Overview
- Scaffold for a customer-support chatbot focused on shipment questions.
- Swappable foundation model (dummy or Hugging Face), prompt templates, and a
  persistent RAG vector store, plus an evaluation harness to compare them.

Quick start
1. Create and activate a Python virtualenv.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Chat with the dummy model (no downloads required):

```bash
python -m src.chatbot.app --model local
```

3. Chat with a real Hugging Face model instead:

```bash
python -m src.chatbot.app --model hf --hf-model distilgpt2
```

`distilgpt2` is small and runs on CPU but isn't instruction-tuned, so
replies are weak. For better quality, swap in a small instruct model, e.g.
`--hf-model Qwen/Qwen2.5-0.5B-Instruct`.

Retrieval-augmented generation (RAG)
1. Ingest the sample shipment docs in `data/shipment_docs/` into a
   persistent vector store (embeds with sentence-transformers; uses FAISS
   if installed, otherwise falls back to a numpy index — FAISS is skipped
   on macOS in `requirements.txt`):

```bash
python -m src.chatbot.ingest --docs data/shipment_docs --out data/vector_store
```

2. Chat with retrieval enabled:

```bash
python -m src.chatbot.app --model hf --hf-model distilgpt2 --rag-store data/vector_store
```

Add your own `.txt`/`.md` docs to `data/shipment_docs/` (or point `--docs`
elsewhere) and re-run `ingest.py` to refresh the store.

Guardrails
Two lightweight, heuristic guardrails are on by default (pass `--no-guardrails`
to disable):
- **Scope check** (input side): rejects messages that don't match a list of
  shipment/support keywords, without calling the model. With `--model hf`,
  an ambiguous rule-check result additionally falls back to asking the model
  itself yes/no whether the message is on-topic — that fallback only works
  reliably with instruction-tuned models, not `distilgpt2`.
- **Groundedness check** (output side): flags numeric claims (day counts,
  dollar amounts, hour counts) in the response that don't literally appear
  in the retrieved RAG context, and appends an `[unverified: ...]` marker
  when interacting via `app.py`.

Both are substring/regex matching, not semantic understanding — a
paraphrased number or an off-topic question that happens to use a shipment
keyword can slip past. That's an intentional, disclosed simplification; see
`src/chatbot/guardrails.py` for the exact rules.

Evaluating prompt/RAG tweaks
Run the fixed shipment-support test set across every (prompt, RAG on/off)
combination and score responses against a reference answer per query
(ROUGE-1/ROUGE-L F1, pure-Python, no extra dependency), plus keyword
coverage, latency, and length. ROUGE catches rambling/repetition/off-topic
replies that keyword matching alone would miss — e.g. a response that
echoes a RAG doc verbatim instead of answering scores low on ROUGE-L
despite hitting a keyword.

```bash
python -m src.chatbot.evaluate --model local
python -m src.chatbot.evaluate --model hf --hf-model distilgpt2 \
    --rag-store data/vector_store --rag-modes off,on
```

Results print as a table (now including `avg_groundedness_pass_rate`) and
save to `results/eval_results.csv` (`--out` to change the path), followed by
a scope-guardrail summary (false-positive rate on in-domain queries, catch
rate on a small adversarial set). Use this to A/B test prompt templates, RAG
on/off, or different models/checkpoints (e.g. before vs. after fine-tuning)
against the same metrics.

Files
- `src/chatbot/model_base.py`: model interface, dummy model, and `HFModel`
  (transformers-backed, with retry logic and chat-template support).
- `src/chatbot/prompting.py`: prompt templates and prompt switching.
- `src/chatbot/rag.py`: minimal in-memory RAG demo (used by `--use-rag`).
- `src/chatbot/vector_store.py`: persistent embeddings + FAISS/numpy index.
- `src/chatbot/ingest.py`: chunks docs and builds/saves a vector store.
- `src/chatbot/metrics.py`: keyword coverage and ROUGE-1/ROUGE-L F1.
- `src/chatbot/guardrails.py`: scope (input) and groundedness (output) checks.
- `src/chatbot/evaluate.py`: prompt/RAG/guardrail evaluation harness.
- `src/chatbot/app.py`: CLI to chat and wire together model/prompt/RAG.

Next steps
- Wire up an API-backed model (OpenAI, Anthropic, etc.) as an alternative
  `FoundationModel` implementation.
- Fine-tune a small open model on real support transcripts and compare it
  against the base model with `evaluate.py`.
- Add unit tests and a small web UI (FastAPI + frontend) for demos.
