"""Content-addressed template transport; no mutation of shared Harbor task files."""

import hashlib
from dataclasses import dataclass

DEFAULT_MAX_TEMPLATE_TOKENS = 1024
_OPEN = "<p2o_template"
_CLOSE = "</p2o_template>"


@dataclass(frozen=True)
class Template:
    text: str

    def __post_init__(self):
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("P2O template must be nonempty text")
        if len(self.text) > 65536 or _OPEN in self.text or _CLOSE in self.text:
            raise ValueError("P2O template contains reserved markers or exceeds transport size")

    @property
    def template_id(self):
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    @property
    def block(self):
        return f'\n\n{_OPEN} id="{self.template_id}">\n{self.text}\n{_CLOSE}\n'

    def to_dict(self):
        return {"version": 1, "template_id": self.template_id, "text": self.text}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {"version", "template_id", "text"}:
            raise ValueError("Invalid P2O template payload")
        if type(value["version"]) is not int or value["version"] != 1:
            raise ValueError("Unsupported P2O template version")
        template = cls(value["text"])
        if template.template_id != value["template_id"]:
            raise ValueError("P2O template content hash mismatch")
        return template

    def validate_token_budget(self, tokenizer, *, max_tokens=DEFAULT_MAX_TEMPLATE_TOKENS):
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError("P2O template token budget must be a positive integer")
        if any(token in self.text for token in tokenizer.all_special_tokens):
            raise ValueError("P2O template must not contain model control tokens")
        count = len(tokenizer.encode(self.block, add_special_tokens=False))
        if count > max_tokens:
            raise ValueError(f"P2O template including wrapper uses {count} tokens, budget={max_tokens}")
        return count


def augment_instruction(instruction: str, template: Template) -> str:
    if _OPEN in instruction or _CLOSE in instruction:
        raise ValueError("Task instruction already contains P2O markers")
    return instruction + template.block
