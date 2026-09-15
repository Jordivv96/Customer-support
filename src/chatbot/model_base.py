import time
from typing import Optional, Dict, Any


class FoundationModel:
    """Abstract foundation model interface.

    Implement `generate` to connect to a real model (HF, API, or fine-tuned).
    """

    def generate(self, prompt: str, max_tokens: int = 256, **kwargs) -> str:
        raise NotImplementedError()


class LocalDummyModel(FoundationModel):
    """A tiny local baseline for development and testing."""

    def __init__(self, persona: Optional[str] = None):
        self.persona = persona or "SupportBot"

    def generate(self, prompt: str, max_tokens: int = 256, **kwargs) -> str:
        # Very simple canned reply logic for shipment-related queries.
        text = prompt.lower()
        if "where" in text and "ship" in text:
            return "Your order is in transit — expected delivery in 3-5 business days."
        if "tracking" in text:
            return "Please provide your tracking number and I'll check the status."
        if "delay" in text or "late" in text:
            return (
                "Sorry for the delay. I can escalate this to the logistics team if you want."
            )
        # Default echo with persona
        return f"{self.persona}: I received your message and will follow up shortly."


class HFModel(FoundationModel):
    """Foundation model backed by a local Hugging Face `transformers` model.

    Works with any causal LM on the Hub (instruct-tuned models will follow
    the prompt best). Defaults to a small public model so it runs on CPU
    without extra setup; swap in a larger instruct model via `model_name`
    for better quality.
    """

    def __init__(
        self,
        model_name: str = "distilgpt2",
        device: Optional[str] = None,
        max_new_tokens: int = 128,
        **generate_kwargs: Any,
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise RuntimeError(
                "transformers/torch are required for HFModel; run `pip install -r requirements.txt`"
            ) from e

        self._torch = torch
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.generate_kwargs = generate_kwargs

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _format_prompt(self, prompt: str) -> str:
        # Use the model's chat template when it has one (instruct-tuned models);
        # otherwise fall back to the raw prompt for plain causal LMs.
        if getattr(self.tokenizer, "chat_template", None):
            messages = [{"role": "user", "content": prompt}]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return prompt

    def generate(self, prompt: str, max_tokens: int = None, retries: int = 2, **kwargs) -> str:
        torch = self._torch
        max_new_tokens = max_tokens or self.max_new_tokens
        gen_kwargs = {"do_sample": True, "temperature": 0.7, **self.generate_kwargs, **kwargs}

        formatted = self._format_prompt(prompt)
        last_err: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                inputs = self.tokenizer(formatted, return_tensors="pt", truncation=True).to(self.device)
                with torch.no_grad():
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        pad_token_id=self.tokenizer.pad_token_id,
                        **gen_kwargs,
                    )
                new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
                return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            except Exception as e:  # transient OOM/generation errors are worth a retry
                last_err = e
                if attempt < retries:
                    time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"HFModel generation failed after {retries + 1} attempts: {last_err}")
