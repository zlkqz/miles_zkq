import math
import random

import pytest

from examples.seta.p2o.evidence import EpisodeEvidence, TaskKey
from examples.seta.p2o.history import HistoryState, TemplateCycle, TemplateEvaluation, TemplateHistory
from examples.seta.p2o.scoring import DevScores, DevTaskScore
from examples.seta.p2o.templates import Template


def evaluation(text, reward, *, task=None, solved=False, policy="v1"):
    task = task or TaskKey(dataset_id="d", instance_id="dev")
    episode = EpisodeEvidence(
        task=task, attempt_id=f"{text}:0", valid=True, passed=10 if solved else 0, total=10, reward=reward
    )
    scores = DevScores(tasks=(DevTaskScore(task=task, episodes=(episode,)),))
    return TemplateEvaluation(text=text, scores=scores, policy_version=policy)


def test_normalization_is_per_cycle_not_per_task_or_global(task):
    history = TemplateHistory()
    history.commit(TemplateCycle(cycle_id=1, evaluations=(evaluation("a", 0.1), evaluation("b", 0.3))))
    history.commit(TemplateCycle(cycle_id=2, evaluations=(evaluation("c", 0.8), evaluation("d", 1.0))))
    ids, probabilities, reason = history.probabilities(task)
    probs = dict(zip(ids, probabilities))
    assert probs[Template("a").template_id] == pytest.approx(probs[Template("c").template_id])
    assert probs[Template("b").template_id] == pytest.approx(probs[Template("d").template_id])
    assert probs[Template("b").template_id] / probs[Template("a").template_id] == pytest.approx(math.exp(2))
    assert reason == "global_history"


def test_empty_baseline_excluded_zero_variance_uniform(task):
    history = TemplateHistory()
    history.commit(
        TemplateCycle(cycle_id=1, evaluations=(evaluation("", 0), evaluation("a", 0.8), evaluation("b", 0.8)))
    )
    ids, probabilities, _ = history.probabilities(task)
    assert len(ids) == 2 and probabilities == (0.5, 0.5)


def test_same_hash_uses_latest_score_but_keeps_earlier_solver_evidence(task):
    history = TemplateHistory()
    history.commit(TemplateCycle(cycle_id=1, evaluations=(evaluation("a", 1.2, task=task, solved=True),)))
    history.commit(TemplateCycle(cycle_id=2, evaluations=(evaluation("a", 0), evaluation("b", 0.5))))
    assert len(history) == 2 and len(history.snapshot().cycles) == 2
    choice = history.choose(task, rng=random.Random(1))
    assert choice.template.text == "a" and choice.score_cycle == 2
    assert choice.normalized_score == -1 and choice.eligibility == "known_solver"


def test_partial_progress_does_not_imply_solver(task):
    history = TemplateHistory()
    history.commit(
        TemplateCycle(cycle_id=1, evaluations=(evaluation("a", 0.9, task=task), evaluation("b", 1.0, task=task)))
    )
    assert history.probabilities(task)[2] == "global_history"


def test_old_assignment_survives_no_new_templates_and_new_solver_wins(task):
    history = TemplateHistory()
    history.commit(TemplateCycle(cycle_id=1, evaluations=(evaluation("old", 0.5),)))
    history.assign([task], repeats=4, rng=random.Random(1))
    history.commit(TemplateCycle(cycle_id=2, evaluations=(evaluation("", 0.0),)))
    history.assign([task], repeats=4, rng=random.Random(9))
    assert history.choose(task, rng=random.Random(0)).eligibility == "previous_assignment"
    history.commit(TemplateCycle(cycle_id=3, evaluations=(evaluation("new", 1.2, task=task, solved=True),)))
    choice = history.choose(task, rng=random.Random(0))
    assert choice.template.text == "new" and choice.eligibility == "known_solver"


def test_roundtrip_restores_scores_assignments_and_same_draws(task):
    history = TemplateHistory()
    history.commit(TemplateCycle(cycle_id=1, evaluations=tuple(evaluation(f"t{i}", i / 10) for i in range(5))))
    rng = random.Random(42)
    history.assign([task], repeats=16, rng=rng)
    state = HistoryState.model_validate_json(history.snapshot().model_dump_json())
    restored = TemplateHistory.restore(state)
    other_rng = random.Random()
    other_rng.setstate(rng.getstate())
    assert [history.choose(task, rng=rng) for _ in range(30)] == [
        restored.choose(task, rng=other_rng) for _ in range(30)
    ]
    assert state == restored.snapshot()


def test_empty_pool_falls_back_without_faking_template(task):
    history = TemplateHistory()
    history.commit(TemplateCycle(cycle_id=1, evaluations=()))
    history.assign([task], repeats=16, rng=random.Random(1))
    assert history.choose(task, rng=random.Random(1)) is None and not history.snapshot().assignments


def test_commit_is_idempotent_but_rejects_conflicting_replacement():
    history = TemplateHistory()
    cycle = TemplateCycle(cycle_id=1, evaluations=(evaluation("a", 0),))
    history.commit(cycle)
    history.commit(cycle)
    assert len(history.snapshot().cycles) == 1
    with pytest.raises(ValueError, match="in order"):
        history.commit(TemplateCycle(cycle_id=1, evaluations=()))


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf")])
def test_reject_invalid_temperature(temperature):
    with pytest.raises(ValueError):
        TemplateHistory(temperature=temperature)


def test_tiny_temperature_remains_finite(task):
    history = TemplateHistory(temperature=1e-320)
    history.commit(TemplateCycle(cycle_id=1, evaluations=(evaluation("a", 0), evaluation("b", 1))))
    assert sorted(history.probabilities(task)[1]) == [0.0, 1.0]


def test_search_train_success_is_retained_for_task_specific_assignment(task):
    solved = EpisodeEvidence(task=task, attempt_id="mini_after:0", valid=True, passed=1, total=1, reward=1.2)
    entry = evaluation("search solver", 0.1).model_copy(update={"search_episodes": (solved,)})
    history = TemplateHistory()
    history.commit(TemplateCycle(cycle_id=1, evaluations=(entry, evaluation("higher dev", 0.9))))
    restored = TemplateHistory.restore(HistoryState.model_validate_json(history.snapshot().model_dump_json()))
    chosen = restored.choose(task, rng=random.Random(2))
    assert chosen.template.text == "search solver" and chosen.eligibility == "known_solver"
