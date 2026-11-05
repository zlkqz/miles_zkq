"""Frozen dual-context scoring and CRM, using native Megatron forward-only."""

import logging
import math
from dataclasses import dataclass

import torch

from examples.seta.p2o.contexts import context_pair
from miles.backends.training_utils.data.context_parallel import all_gather_with_cp, slice_log_prob_with_cp
from miles.backends.training_utils.loss.hub.corrections import icepop_function
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.function_registry import load_function
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CRMConfig:
    lower: float = 0.01
    upper: float = 10.0

    def __post_init__(self):
        if not (math.isfinite(self.lower) and math.isfinite(self.upper) and 0 < self.lower <= self.upper):
            raise ValueError("CRM requires finite 0 < lower <= upper")


def attach_training_context(sample: Sample):
    """Native train_metadata transports the verified prefix, without token copies."""
    if sample.status not in (Sample.Status.COMPLETED, Sample.Status.TRUNCATED) or sample.remove_sample:
        return
    pair = context_pair(sample)
    templated = sample.metadata["p2o"]["templated"]
    if templated != (sample.metadata.get("p2o_context") is not None):
        raise ValueError("P2O generation assignment disagrees with verified context")
    sample.train_metadata = {
        **(sample.train_metadata or {}),
        "p2o": {
            "templated": templated,
            "template_id": sample.metadata["p2o"].get("template_id"),
            "augmented_prompt_length": len(pair.augmented_tokens) - pair.response_length,
            "plain_prompt_token_ids": list(pair.plain_tokens[: -pair.response_length]) if templated else None,
        },
    }


def validate_training_args(args):
    incompatible = (
        "use_rollout_logprobs",
        "skip_actor_forward_only",
        "keep_old_actor",
        "use_critic",
        "use_kl_loss",
        "use_opd",
        "use_sampling_support_replay",
        "enable_witness",
        "use_routing_replay",
        "use_indexer_replay",
    )
    active = [name for name in incompatible if getattr(args, name, False)]
    if active or getattr(args, "kl_coef", 0) != 0:
        raise ValueError(f"P2O dual-context scoring is incompatible with {active} or nonzero KL reward")
    if args.loss_type != "policy_loss" or args.advantage_estimator != "grpo":
        raise ValueError("P2O requires native GRPO policy loss")
    if not args.compute_advantages_and_returns or not args.use_tis:
        raise ValueError("P2O requires frozen actor scoring, native advantages and IcePop")
    # Existing aliases may refer to the same native correction function.
    if load_function(args.custom_tis_function_path) is not icepop_function:
        raise ValueError("P2O requires the native IcePop correction")
    if args.allgather_cp and get_parallel_state().cp.size > 1:
        raise ValueError("P2O dual-context scoring currently supports native zigzag CP, not allgather CP")


def crm_weights(old_plain, old_augmented, *, templated, config):
    if old_plain.shape != old_augmented.shape:
        raise ValueError("CRM contexts must score the same response tokens")
    if not templated:
        return torch.ones_like(old_plain, requires_grad=False)
    log_ratio = old_plain.detach() - old_augmented.detach()
    return ((log_ratio >= math.log(config.lower)) & (log_ratio <= math.log(config.upper))).to(old_plain.dtype)


def _plain_data(data):
    plain = dict(data)
    tokens, flags = [], []
    metadata = data.get("metadata")
    if metadata is None or len(metadata) != len(data["tokens"]):
        raise ValueError("P2O scoring requires native train_metadata for every episode")
    for index, (original, response_length, meta) in enumerate(
        zip(data["tokens"], data["response_lengths"], metadata, strict=True)
    ):
        info = meta["p2o"]
        if info["augmented_prompt_length"] != len(original) - response_length:
            raise ValueError("P2O scoring prefix does not match the retained trajectory")
        flags.append(info["templated"])
        if info["templated"]:
            prefix = torch.as_tensor(info["plain_prompt_token_ids"], dtype=original.dtype, device=original.device)
            value = torch.cat((prefix, original[-response_length:]))
        else:
            value = original
        if "max_seq_lens" in data and len(value) > data["max_seq_lens"][index]:
            raise ValueError("P2O plain context exceeds the native padded sequence length")
        tokens.append(value)
    plain["tokens"] = tokens
    plain["total_lengths"] = [len(value) for value in tokens]
    return plain, flags


def _align_response(values, augmented, plain, args):
    """Move response scores between CP partitions when prefix lengths change."""
    result = []
    for index, value in enumerate(values):
        response = augmented["response_lengths"][index]
        aug_max = augmented["max_seq_lens"][index] if "max_seq_lens" in augmented else None
        plain_max = plain["max_seq_lens"][index] if "max_seq_lens" in plain else None
        full = all_gather_with_cp(
            value.detach(), augmented["total_lengths"][index], response, args.qkv_format, aug_max
        )
        result.append(
            slice_log_prob_with_cp(full, plain["total_lengths"][index], response, args.qkv_format, plain_max).detach()
        )
    return result


@torch.no_grad()
def score_contexts(args, rollout_data, *, score):
    """The actor selects old weights before this hook and never changes them here.

    Both forwards use native schedules on every rank, including all-plain local
    shards. This keeps DP/PP collectives aligned for mixed templated batches.
    """
    validate_training_args(args)
    config = CRMConfig(args.p2o_crm_lower, args.p2o_crm_upper)
    augmented = dict(rollout_data)
    plain, flags = _plain_data(augmented)
    augmented_scores = score(augmented, "p2o_aug_")
    if "rollout_log_probs" in augmented:
        plain["rollout_log_probs"] = _align_response(augmented["rollout_log_probs"], augmented, plain, args)
    plain_scores = score(plain, "")
    plain.update(plain_scores)
    if get_parallel_state().is_pp_last_stage:
        old_aug = _align_response(augmented_scores["p2o_aug_log_probs"], augmented, plain, args)
        plain["importance_log_probs"] = old_aug
        plain["pg_loss_weights"] = [
            crm_weights(old, aug, templated=flag, config=config)
            for old, aug, flag in zip(plain_scores["log_probs"], old_aug, flags, strict=True)
        ]
    # Original response masks, behavior values and rollout identities stay intact;
    # only CP ownership of response-aligned scores may change.
    rollout_data.update(plain)
    logger.info(
        "P2O dual-context scoring episodes=%d templated=%d augmented_tokens=%d plain_tokens=%d CRM=[%g,%g]",
        len(flags),
        sum(flags),
        sum(augmented["total_lengths"]),
        sum(plain["total_lengths"]),
        config.lower,
        config.upper,
    )
