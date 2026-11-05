"""Exact two-context scoring inputs over Miles' retained response tokens.

Only the initial non-generated prompt is rendered again. Every retained
response token (actions, observations, delimiters) is copied from TITO. Native
stop trimming, truncation, and branch ownership have already run at this point.
Training integration consumes these inputs in G6; this module does not change
the live sample's tokens, masks, rewards, or behavior log probabilities.
"""

import math
from copy import deepcopy
from dataclasses import dataclass

from examples.seta.p2o.templates import DEFAULT_MAX_TEMPLATE_TOKENS, Template
from miles.utils.chat_template_utils.tito_tokenizer import extract_template_args
from miles.utils.types import Sample


@dataclass(frozen=True)
class ContextPair:
    augmented_tokens: tuple[int, ...]
    plain_tokens: tuple[int, ...]
    response_length: int
    loss_mask: tuple[int, ...]
    behavior_log_probs: tuple[float, ...]
    augmented_action_indices: tuple[int, ...]
    plain_action_indices: tuple[int, ...]


def _plain_messages(messages, template):
    if any(message["role"] == "assistant" for message in messages):
        raise ValueError("P2O root contains historical assistant text; cannot retokenize actions")
    plain = deepcopy(messages)
    matches = 0
    for message in plain:
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("P2O currently requires text-only root messages")
        if template.block in content:
            if message["role"] != "user" or content.count(template.block) != 1:
                raise ValueError("P2O template must occur exactly once in the initial user message")
            message["content"] = content.replace(template.block, "", 1)
            matches += 1
    if matches != 1:
        raise ValueError("P2O template missing or duplicated in actual remote input")
    return plain


def attach_contexts(samples, session_metadata, sample_context, *, max_template_tokens=DEFAULT_MAX_TEMPLATE_TOKENS):
    payload = (session_metadata.get("agent") or {}).get("p2o_template")
    if payload is None:
        return
    if sample_context is None:
        raise ValueError("P2O needs the v2 serving tokenizer and record snapshot")
    template = Template.from_dict(payload)
    tito = sample_context.tito_tokenizer
    template_tokens = template.validate_token_budget(tito.tokenizer, max_tokens=max_template_tokens)
    for sample in samples:
        path_ids = sample.metadata["leaf"]["path_node_ids"]
        if path_ids[0] != 0:
            raise ValueError("P2O history was rewritten into a new root; exact alignment is not supported")
        records = sample_context.records_by_leaf[sample.metadata["leaf"]["node_id"]]
        request = records[0].request
        messages = request["messages"]
        plain_messages = _plain_messages(messages, template)
        render_args = dict(add_generation_prompt=True, tokenize=True, template_args=extract_template_args(request))
        augmented_prefix = tito.apply_chat_template(messages, **render_args)
        plain_prefix = tito.apply_chat_template(plain_messages, **render_args)
        response_start = len(sample.tokens) - sample.response_length
        if augmented_prefix != request["input_ids"] or augmented_prefix != sample.tokens[:response_start]:
            raise ValueError("P2O root rendering differs from actual TITO prompt")
        # Boundary tokenization may change tokens adjacent to the textual block.
        # Record that exact replacement span, rather than deleting an approximate
        # independently-tokenized substring from the augmented prefix.
        start, aug_end, plain_end = _replacement_span(augmented_prefix, plain_prefix)
        sample.metadata["p2o_context"] = {
            "version": 1,
            "template_id": template.template_id,
            "template_token_count": template_tokens,
            "augmented_prompt_length": response_start,
            "plain_prompt_token_ids": plain_prefix,
            "augmented_replacement_span": [start, aug_end],
            "plain_replacement_span": [start, plain_end],
        }
        context_pair(sample)  # Validate now, before this sample can enter a replay buffer.


def _replacement_span(augmented, plain):
    start = 0
    while start < min(len(augmented), len(plain)) and augmented[start] == plain[start]:
        start += 1
    aug_end, plain_end = len(augmented), len(plain)
    while aug_end > start and plain_end > start and augmented[aug_end - 1] == plain[plain_end - 1]:
        aug_end -= 1
        plain_end -= 1
    return start, aug_end, plain_end


def context_pair(sample: Sample) -> ContextPair:
    """Build two causal-LM inputs; action indices address tokens, logits use index-1.

    Response-relative masks/log-probs are identical, including observation zeros
    and branch ownership zeros. Position IDs are 0..len(input)-1 independently
    for each input, not shifted copies of the augmented positions.
    """
    start = len(sample.tokens) - sample.response_length
    if start <= 0 or sample.response_length <= 0:
        raise ValueError("P2O requires a nonempty prompt and retained response")
    metadata = sample.metadata.get("p2o_context")
    if metadata is None:
        if sample.metadata.get("p2o_template") is not None:
            raise ValueError("Templated sample is missing its verified P2O context")
        plain_prefix = sample.tokens[:start]
    else:
        if metadata["version"] != 1 or metadata["augmented_prompt_length"] != start:
            raise ValueError("P2O context schema or prompt length mismatch")
        plain_prefix = metadata["plain_prompt_token_ids"]
    mask, behavior = sample.loss_mask, sample.rollout_log_probs
    if mask is None or behavior is None or len(mask) != sample.response_length or len(behavior) != len(mask):
        raise ValueError("P2O mask/behavior lengths must match the retained response")
    if any(value not in (0, 1) for value in mask) or not all(math.isfinite(value) for value in behavior):
        raise ValueError("P2O requires binary original masks and finite behavior log-probs")
    actions = tuple(index for index, active in enumerate(mask) if active)
    return ContextPair(
        augmented_tokens=tuple(sample.tokens),
        plain_tokens=tuple(plain_prefix) + tuple(sample.tokens[start:]),
        response_length=sample.response_length,
        loss_mask=tuple(mask),
        behavior_log_probs=tuple(behavior),
        augmented_action_indices=tuple(start + index for index in actions),
        plain_action_indices=tuple(len(plain_prefix) + index for index in actions),
    )
