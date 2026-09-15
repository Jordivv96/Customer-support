from typing import Dict


PROMPT_TEMPLATES: Dict[str, str] = {
    "default": (
        "You are a helpful customer support assistant for shipment questions.\n"
        "Customer: {customer_message}\n"
        "Assistant:"
    ),
    "concise": (
        "You are a concise support assistant. Answer briefly.\n"
        "Customer: {customer_message}\n"
        "Assistant:"
    ),
    "detailed": (
        "You are a thorough support assistant. Give steps and possible causes.\n"
        "Customer: {customer_message}\n"
        "Assistant:"
    ),
}


def build_prompt(template_name: str, customer_message: str) -> str:
    template = PROMPT_TEMPLATES.get(template_name, PROMPT_TEMPLATES["default"])
    return template.format(customer_message=customer_message)
