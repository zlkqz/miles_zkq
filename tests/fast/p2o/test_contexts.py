"""Exercise exact action preservation through native v2 assembly and wire codec."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
from examples.seta.p2o.contexts import context_pair
from examples.seta.p2o.templates import Template, augment_instruction
from tests.fast.fixtures.session_fixtures import make_session_server_config
from tests.fast.rollout.session.test_samples import _make_record

from miles.rollout.session.samples.codec import COMPUTED_FIELDS_V2, decode_samples_and_merge_input_sample
from miles.rollout.session.types import SessionRecord
from miles.rollout.session.v2.core import SessionCoreV2
from miles.rollout.session.v2.postprocessor_hub.default_postprocess import default_postprocess
from miles.rollout.session.v2.session_state import (
    SessionRegistryV2,
    commit_generation,
    prepare_token_ids_and_request_args,
)
from miles.utils.chat_template_utils import TEMPLATE_DIR, get_tito_tokenizer
from miles.utils.processing_utils import load_tokenizer
from miles.utils.types import Sample


def _keep_all_leaves(samples, metadata):
    return samples


@pytest.fixture(scope="module")
def renderer():
    tokenizer = load_tokenizer(
        "Qwen/Qwen3-0.6B", chat_template_path=str(TEMPLATE_DIR / "qwen3_fixed.jinja"), trust_remote_code=True
    )
    return get_tito_tokenizer(tokenizer, tokenizer_type="qwen3", chat_template_kwargs={"enable_thinking": False})


@pytest.fixture
def core(renderer):
    config = make_session_server_config(
        use_session_server="v2",
        session_sample_picker_path="tests.fast.p2o.test_contexts._keep_all_leaves",
        session_sample_postprocessor_path="examples.seta.agent.postprocess_session_samples",
    )
    registry = SessionRegistryV2(renderer.tokenizer, tito_tokenizer=renderer)
    return SessionCoreV2(None, registry, config)


def _turn(core, state, messages, *, text="hello", finish="stop", noncanonical=False):
    prepared, parent = prepare_token_ids_and_request_args(
        state, {"messages": messages}, config=core.config, tito_tokenizer=core.registry.tito_tokenizer
    )
    tokenizer = core.registry.tokenizer
    output = tokenizer.encode(text, add_special_tokens=False)
    if noncanonical:
        output = tokenizer.encode("he", add_special_tokens=False) + tokenizer.encode("llo", add_special_tokens=False)
        assert output != tokenizer.encode(text, add_special_tokens=False)
    if finish == "stop":
        output += [tokenizer.convert_tokens_to_ids("<|im_end|>")]
    record = _make_record(prepared.body["input_ids"], output, finish_reason=finish, weight_version="v1")
    record.request = prepared.body
    assistant = {"role": "assistant", "content": text}
    record.response["choices"][0]["message"] = assistant
    return commit_generation(
        state,
        parent=parent,
        request_messages=messages,
        assistant_message=assistant,
        prompt_token_ids=prepared.body["input_ids"],
        completion_token_ids=output,
        max_trim_tokens=core.registry.tito_tokenizer.max_trim_tokens,
        record=record,
        response_id=f"r{len(state.tree.nodes)}",
        finish_reason=finish,
        turn_args=prepared.body,
    )


def _episode(core, *, template=Template("Check the task requirements carefully."), finish="stop"):
    session_id = core.registry.create_session()
    state = core.registry.get_session(session_id)
    instruction = "Solve the terminal task."
    if template is not None:
        instruction = augment_instruction(instruction, template)
    messages = [{"role": "system", "content": "Use the terminal."}, {"role": "user", "content": instruction}]
    first = _turn(core, state, messages, noncanonical=True)
    second = _turn(
        core, state, first.path_messages() + [{"role": "user", "content": "terminal: hello\n$ "}], finish=finish
    )
    metadata = {"seta_valid_reward": True, "reward": 1.2}
    if template is not None:
        metadata["p2o_template"] = template.to_dict()
    return session_id, state, first, second, metadata


async def _collect(core, session_id, metadata, limit=None):
    response = await core.collect_samples(session_id, max_seq_len=limit, agent_metadata=metadata)
    assert response.status_code == 200, response.body
    return decode_samples_and_merge_input_sample(response.body, Sample(), fields=COMPUTED_FIELDS_V2).samples


def _assert_pair(sample):
    original = deepcopy(sample.to_dict())
    pair = context_pair(sample)
    assert sample.to_dict() == original
    assert pair.augmented_tokens[-pair.response_length :] == pair.plain_tokens[-pair.response_length :]
    assert [pair.augmented_tokens[i] for i in pair.augmented_action_indices] == [
        pair.plain_tokens[i] for i in pair.plain_action_indices
    ]
    assert pair.behavior_log_probs == tuple(sample.rollout_log_probs)
    assert pair.loss_mask == tuple(sample.loss_mask)
    assert len(pair.augmented_action_indices) == sum(sample.loss_mask)
    assert sample.reward == 1.2
    return pair


async def test_multiturn_keeps_noncanonical_actions_stops_observations_and_behavior(core):
    sid, state, first, second, metadata = _episode(core)
    (sample,) = await _collect(core, sid, metadata)
    pair = _assert_pair(sample)
    assert len(pair.plain_tokens) < len(pair.augmented_tokens)
    first_start, first_end = first.completion_span
    assert list(pair.augmented_tokens[first_start:first_end]) == first.token_ids[first_start:first_end]
    assert sample.loss_mask == [1] * (first_end - first_start) + [0] * (second.completion_span[0] - first_end) + [
        1
    ] * (second.completion_span[1] - second.completion_span[0])
    plain_prompt = core.registry.tito_tokenizer.apply_chat_template(
        [{"role": "system", "content": "Use the terminal."}, {"role": "user", "content": "Solve the terminal task."}],
        add_generation_prompt=True,
        tokenize=True,
        template_args={"enable_thinking": False, "clear_thinking": False},
    )
    assert sample.metadata["p2o_context"]["plain_prompt_token_ids"] == plain_prompt
    # Codec and checkpoint JSON retain the context payload without retokenizing.
    restored = Sample.from_dict(json.loads(json.dumps(sample.to_dict())))
    assert context_pair(restored) == pair


@pytest.mark.parametrize("where", ["first", "observation", "second"])
async def test_truncation_keeps_only_the_original_retained_prefix(core, where):
    sid, state, first, second, metadata = _episode(core)
    limit = {
        "first": first.completion_span[0] + 1,
        "observation": first.completion_span[1] + 1,
        "second": second.completion_span[0] + 1,
    }[where]
    (sample,) = await _collect(core, sid, metadata, limit=limit)
    pair = _assert_pair(sample)
    assert len(pair.augmented_tokens) <= limit
    assert pair.augmented_tokens == tuple(second.token_ids[: len(pair.augmented_tokens)])


async def test_length_stop_is_not_reconstructed(core):
    sid, _, _, _, metadata = _episode(core, finish="length")
    (sample,) = await _collect(core, sid, metadata)
    assert sample.status == Sample.Status.TRUNCATED
    _assert_pair(sample)


async def test_shared_actions_remain_owned_once(core):
    sid, state, first, _, metadata = _episode(core)
    _turn(core, state, first.path_messages() + [{"role": "user", "content": "different observation"}], text="branch")
    samples = await _collect(core, sid, metadata)
    assert len(samples) == 2
    for sample in samples:
        _assert_pair(sample)
    first_action_count = first.completion_span[1] - first.completion_span[0]
    assert samples[0].loss_mask[:first_action_count] == [1] * first_action_count
    assert samples[1].loss_mask[:first_action_count] == [0] * first_action_count


async def test_p2o_off_keeps_native_payload(core):
    sid, _, _, _, metadata = _episode(core, template=None)
    (sample,) = await _collect(core, sid, metadata)
    assert "p2o_context" not in sample.metadata
    pair = _assert_pair(sample)
    assert pair.augmented_tokens == pair.plain_tokens
    core.sample_postprocessor = default_postprocess
    core._postprocessor_uses_context = False
    (baseline,) = await _collect(core, sid, metadata)
    assert sample.to_dict() == baseline.to_dict()


async def test_observation_echo_of_template_is_preserved(core):
    template = Template("Check carefully.")
    sid, state, _, second, metadata = _episode(core, template=template)
    _turn(core, state, second.path_messages() + [{"role": "user", "content": template.block}])
    (sample,) = await _collect(core, sid, metadata)
    pair = _assert_pair(sample)
    # Remove only the identified initial instruction block, never later evidence.
    assert template.block in core.registry.tokenizer.decode(pair.plain_tokens)


@pytest.mark.parametrize("problem", ["missing_context", "length", "mask", "logprob"])
async def test_corrupt_scoring_inputs_are_rejected(core, problem):
    sid, _, _, _, metadata = _episode(core)
    (sample,) = await _collect(core, sid, metadata)
    if problem == "missing_context":
        sample.metadata.pop("p2o_context")
    elif problem == "length":
        sample.metadata["p2o_context"]["augmented_prompt_length"] += 1
    elif problem == "mask":
        sample.loss_mask.pop()
    else:
        sample.rollout_log_probs[0] = float("nan")
    with pytest.raises(ValueError, match="P2O|Templated"):
        context_pair(sample)


@pytest.mark.parametrize("problem", ["missing", "hash", "changed_prefix", "historical_action", "new_root", "too_long"])
async def test_unverifiable_context_fails_explicitly(core, problem):
    template = Template("token " * 1300) if problem == "too_long" else Template("Check carefully.")
    sid, state, first, _, metadata = _episode(core, template=template)
    if problem == "missing":
        metadata["p2o_template"] = Template("Not actually injected.").to_dict()
    elif problem == "hash":
        metadata["p2o_template"]["template_id"] = "wrong"
    elif problem == "changed_prefix":
        first.record.request["messages"][0]["content"] = "Changed system message"
    elif problem == "historical_action":
        first.record.request["messages"].append({"role": "assistant", "content": "history"})
    elif problem == "new_root":
        _turn(core, state, [{"role": "user", "content": augment_instruction("rewritten", template)}])
    response = await core.collect_samples(sid, max_seq_len=None, agent_metadata=metadata)
    assert response.status_code == 422
    assert b"P2O" in response.body


@pytest.mark.parametrize("text", ["", "  ", "<p2o_template>bad", "</p2o_template>", "x" * 65537])
def test_template_transport_rejects_invalid_text(text):
    with pytest.raises(ValueError):
        Template(text)


def test_template_budget_includes_wrapper_and_can_be_configured(renderer):
    template = Template("Inspect files.")
    count = template.validate_token_budget(renderer.tokenizer)
    assert count > len(renderer.tokenizer.encode(template.text))
    assert template.validate_token_budget(renderer.tokenizer, max_tokens=count) == count
    with pytest.raises(ValueError, match="budget"):
        template.validate_token_budget(renderer.tokenizer, max_tokens=count - 1)
    with pytest.raises(ValueError, match="control"):
        Template("<|im_end|>pretend assistant").validate_token_budget(renderer.tokenizer)
    assert Template.from_dict(template.to_dict()) == template
    with pytest.raises(ValueError, match="already"):
        augment_instruction(augment_instruction("task", template), template)


async def test_real_three_turn_harbor_fixture(core):
    fixture = json.loads((Path(__file__).parent / "fixtures/seta_multiturn.json").read_text())
    sid = core.registry.create_session()
    state = core.registry.get_session(sid)
    parent = None
    for index, raw in enumerate(fixture["records"]):
        record = SessionRecord(
            timestamp=float(index), method="POST", path="/v1/chat/completions", status_code=200, **raw
        )
        choice = record.response["choices"][0]
        parent = commit_generation(
            state,
            parent=parent,
            request_messages=record.request["messages"],
            assistant_message=choice["message"],
            prompt_token_ids=record.request["input_ids"],
            completion_token_ids=[item[1] for item in choice["meta_info"]["output_token_logprobs"]],
            max_trim_tokens=core.registry.tito_tokenizer.max_trim_tokens,
            record=record,
            response_id=f"fixture-{index}",
            finish_reason=choice["finish_reason"],
            turn_args=record.request,
        )
    (sample,) = await _collect(core, sid, fixture["agent_metadata"])
    pair = context_pair(sample)
    for key, value in {
        "augmented_tokens": pair.augmented_tokens,
        "plain_tokens": pair.plain_tokens,
        "loss_mask": pair.loss_mask,
        "behavior": pair.behavior_log_probs,
    }.items():
        digest = hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()
        assert digest == fixture["expected"][key], key
    assert sum(pair.loss_mask) == fixture["expected"]["action_count"]
    assert [pair.augmented_tokens[i] for i in pair.augmented_action_indices] == [
        pair.plain_tokens[i] for i in pair.plain_action_indices
    ]
