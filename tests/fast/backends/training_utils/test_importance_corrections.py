"""Numerical contracts for SETA's token importance correction and PPO clipping."""

from argparse import Namespace

import pytest
import torch

from miles.backends.training_utils.loss.hub.corrections import icepop_function, vanilla_tis_function
from miles.backends.training_utils.loss.hub.math_utils import compute_policy_loss


@pytest.mark.parametrize(
    ("lower", "ratios", "expected"),
    [
        (0.0, [0.0, 0.5, 1.0, 1.999, 2.0, 2.001, 5.0], [0.0, 0.5, 1.0, 1.999, 2.0, 0.0, 0.0]),
        (0.5, [0.499, 0.5, 0.501, 2.0, 2.001], [0.0, 0.5, 0.501, 2.0, 0.0]),
    ],
)
def test_icepop_preserves_inclusive_bounds_and_pops_outliers(lower, ratios, expected):
    ratio = torch.tensor(ratios, dtype=torch.float64)
    pg_loss = torch.ones_like(ratio, requires_grad=True)
    masks = [torch.ones(2), torch.ones(len(ratios) - 2)]
    masks_before = [mask.clone() for mask in masks]
    corrected, returned_masks, metrics = icepop_function(
        Namespace(tis_clip_low=lower, tis_clip=2),
        pg_loss=pg_loss,
        train_log_probs=list(ratio.log().split([2, len(ratios) - 2])),
        rollout_log_probs=[torch.zeros_like(mask, dtype=torch.float64) for mask in masks],
        loss_masks=masks,
    )
    expected = torch.tensor(expected, dtype=torch.float64)
    torch.testing.assert_close(corrected, expected)
    corrected.sum().backward()
    torch.testing.assert_close(pg_loss.grad, expected)
    assert returned_masks is masks
    for before, after in zip(masks_before, masks, strict=True):
        torch.testing.assert_close(before, after)
    torch.testing.assert_close(metrics["tis"], ratio)
    assert all(not metric.requires_grad for metric in metrics.values())


def test_icepop_zeroes_a_rejected_token_without_changing_the_reduction_denominator():
    kwargs = dict(
        pg_loss=torch.tensor([-2.0, -4.0, 99.0]),
        train_log_probs=[torch.tensor([1.0, 3.0, 1.0]).log()],
        rollout_log_probs=[torch.zeros(3)],
        loss_masks=[torch.tensor([1.0, 1.0, 0.0])],
    )
    args = Namespace(tis_clip_low=0, tis_clip=2)
    ice_loss, ice_masks, _ = icepop_function(args, **kwargs)
    tis_loss, tis_masks, _ = vanilla_tis_function(args, **kwargs)
    assert ((ice_loss * ice_masks[0]).sum() / ice_masks[0].sum()).item() == -1.0
    assert ((tis_loss * tis_masks[0]).sum() / tis_masks[0].sum()).item() == -5.0


def test_ppo_clip_is_asymmetric_and_depends_on_advantage_sign():
    # Positive advantages cap improvement above 1.28; negative ones cap below 0.8.
    # The opposite directions remain differentiable, rather than clamping all ratios.
    log_ratio = torch.tensor([0.7, 1.0, 1.4, 0.7, 1.0, 1.4]).log().requires_grad_()
    advantages = torch.tensor([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
    loss, clipfrac = compute_policy_loss(-log_ratio, advantages, eps_clip=0.2, eps_clip_high=0.28)
    torch.testing.assert_close(loss, torch.tensor([-0.7, -1.0, -1.28, 0.8, 1.0, 1.4]))
    torch.testing.assert_close(clipfrac, torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 0.0]))
    loss.sum().backward()
    torch.testing.assert_close(log_ratio.grad, torch.tensor([-0.7, -1.0, 0.0, 0.0, 1.0, 1.4]))
