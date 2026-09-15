"""Safety/scope guardrails for the customer support chatbot.

Two independent checks:
- Scope guardrail (input side): is this a shipment/support question at all?
- Groundedness guardrail (output side): does the response make numeric
  claims (day counts, dollar amounts) that aren't backed by the retrieved
  RAG context?

Both are deliberately simple/heuristic: substring and regex matching, not
semantic entailment. A paraphrased number ("3 to 5 days" vs. a doc's
"3-5 business days") can false-positive as unverified, and a paraphrased
off-topic question without any of the listed keywords can false-negative
past the rule-based scope check. That's an accepted, disclosed trade-off
for this project's scale, not something to over-engineer — a production
system would back these with a classifier model rather than regex.
"""
import re
from dataclasses import dataclass, field
from typing import List, Optional

from .model_base import FoundationModel

SCOPE_KEYWORDS = [
    "ship", "shipment", "shipping", "order", "package", "deliver", "delivery",
    "track", "tracking", "return", "refund", "exchange", "damaged", "lost",
    "international", "customs", "warehouse", "carrier",
]

SCOPE_REFUSAL = (
    "I can help with questions about shipping, orders, tracking, returns, "
    "and refunds. Could you rephrase your question around one of those?"
)

SCOPE_LLM_PROMPT = (
    "You are a strict classifier. Answer with exactly one word: YES or NO.\n"
    "Is the following customer message related to shipping, orders, tracking, "
    "deliveries, returns, or refunds?\n"
    "Message: {query}\nAnswer:"
)

NUMERIC_CLAIM_RE = re.compile(
    r"\b\d+(?:-\d+)?\s*(?:business\s+)?days?\b"  # "3-5 business days", "7 days"
    r"|\$\d+(?:\.\d{2})?"                         # "$50", "$19.99"
    r"|\b\d+\s*(?:hours?|hrs?)\b",                # "24 hours"
    re.IGNORECASE,
)


@dataclass
class GuardrailResult:
    passed: bool
    reason: Optional[str] = None
    details: List[str] = field(default_factory=list)


def check_scope_rules(query: str) -> GuardrailResult:
    text = query.lower()
    if any(kw in text for kw in SCOPE_KEYWORDS):
        return GuardrailResult(passed=True)
    return GuardrailResult(passed=False, reason="no shipment/support keywords matched")


def check_scope_llm(query: str, model: FoundationModel) -> GuardrailResult:
    resp = model.generate(SCOPE_LLM_PROMPT.format(query=query), max_tokens=5)
    if "yes" in resp.strip().lower():
        return GuardrailResult(passed=True)
    return GuardrailResult(
        passed=False, reason=f"LLM classifier said off-topic (raw: {resp.strip()[:40]!r})"
    )


def check_scope(query: str, model: Optional[FoundationModel] = None) -> GuardrailResult:
    """Rule-based check first; only falls back to an LLM call when the
    keyword check found nothing (ambiguous) and a model is supplied."""
    result = check_scope_rules(query)
    if result.passed or model is None:
        return result
    return check_scope_llm(query, model)


def extract_numeric_claims(text: str) -> List[str]:
    return [m.group(0) for m in NUMERIC_CLAIM_RE.finditer(text)]


def check_groundedness(response: str, context_chunks: List[str]) -> GuardrailResult:
    """Flags numeric claims in the response that don't literally appear in
    the retrieved context. Only meaningful when RAG context was retrieved."""
    claims = extract_numeric_claims(response)
    if not claims:
        return GuardrailResult(passed=True)
    if not context_chunks:
        return GuardrailResult(
            passed=False, reason="no RAG context to verify claims against", details=claims
        )

    context_text = " ".join(context_chunks).lower()
    unverified = [c for c in claims if c.lower() not in context_text]
    if not unverified:
        return GuardrailResult(passed=True)
    return GuardrailResult(
        passed=False, reason="claims not found in retrieved context", details=unverified
    )


def annotate_unverified(response: str, result: GuardrailResult) -> str:
    if result.passed:
        return response
    tags = ", ".join(result.details) if result.details else result.reason
    return f"{response}\n\n[unverified: {tags}]"
