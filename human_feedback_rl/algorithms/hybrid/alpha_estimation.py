"""The reliability weight, from the scatter of each channel's gradient.

alpha = CV2_pref / (CV2_pref + CV2_demo) is the weight on the demonstrations.
Two sizes matter and are not the same: N, the samples a channel has, estimates
how much the process scatters; B, the minibatch, divides it, because the
applied gradient averages B draws.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch as th

from human_feedback_rl.common.batching import (
    fragment_avg_rewards,
    fragment_sum_rewards,
)
from human_feedback_rl.common.preference_losses import (
    bradley_terry_probs,
    preference_nll_per_sample,
)

#: Below this many comparisons the preference dispersion cannot be estimated.
#: Estimating it anyway biases the value downwards, towards the channel that is
#: in fact less reliable, so alpha is pinned to 1: all weight on demonstrations.
ALPHA_MIN_PREFS = 5


@dataclass(frozen=True)
class ChannelDispersion:
    """How much a channel scatters, from the single sample to the minibatch."""

    n: int                  # samples available, used to estimate V
    batch: int              # minibatch size, the divisor of S
    process_var: float      # V: variance of the generating process
    mean_var: float         # S = V / batch: variance of the sample mean
    mean_norm_sq: float     # ||g_mean||^2, the denominator that makes CV^2 dimensionless
    cv2: float              # S / ||g_mean||^2


@dataclass(frozen=True)
class AlphaEstimate:
    alpha: float
    pref: Optional[ChannelDispersion]
    demo: Optional[ChannelDispersion]
    pinned: bool            # True when alpha is the fallback, not an estimate


def _flat_grads(make_scalar, items: Sequence, params: List[th.Tensor]) -> th.Tensor:
    """Flattened per-sample gradients, one per row.

    The graph is built and freed ONE SAMPLE AT A TIME. A single forward over all
    samples, slicing the outputs afterwards, would cost quadratically: every
    backward would walk the whole shared graph. This way the total cost is
    proportional to the number of transitions, like an ordinary
    forward-backward over the dataset.
    """
    rows = []
    for item in items:
        grads = th.autograd.grad(make_scalar(item), params, allow_unused=True)
        rows.append(
            th.cat([
                (th.zeros_like(p) if g is None else g).reshape(-1)
                for p, g in zip(params, grads)
            ])
        )
    return th.stack(rows)


def _dispersion(per_sample: th.Tensor, batch: int, eps: float) -> Optional[ChannelDispersion]:
    """The recipe: mean, squared distances, /(N-1), then /B."""
    n = per_sample.shape[0]
    if n < 2 or batch <= 0:
        return None
    mean = per_sample.mean(dim=0)
    # sum_i ||g_i - gbar||^2, without materialising the differences
    total_sq = float((per_sample - mean).pow(2).sum())
    process_var = total_sq / (n - 1)
    mean_var = process_var / batch
    mean_norm_sq = float(mean.pow(2).sum())
    if not np.isfinite(process_var) or not np.isfinite(mean_norm_sq):
        return None
    return ChannelDispersion(
        n=n,
        batch=batch,
        process_var=process_var,
        mean_var=mean_var,
        mean_norm_sq=mean_norm_sq,
        cv2=mean_var / max(mean_norm_sq, eps),
    )


def preference_sample_gradients(member, batch, smooth_labels, params) -> th.Tensor:
    """One gradient per comparison.

    The preference loss is a mean over independent comparisons, so the
    decomposition is exact: ``mean_i g_i`` is exactly the full-batch gradient.
    Each ``l_i`` depends on theta only through ``Delta_i``, the difference
    between the two fragment scores, hence

        g_i = (d l_i / d Delta_i) * grad_theta Delta_i

    The coefficient is not derived by hand but taken with autograd on the same
    ``preference_nll_per_sample`` used in training: it stays exact through the
    internal ``clamp``, and cannot drift if the loss ever changes.
    """
    pairs = batch.fragment_pairs

    # Delta values: one forward, no graph -- they only feed the coefficients.
    with th.no_grad():
        delta = (
            fragment_avg_rewards(member, [p.frag1 for p in pairs])
            - fragment_avg_rewards(member, [p.frag2 for p in pairs])
        )

    # Coefficients: derivative of the loss wrt the Deltas alone, theta aside.
    delta_leaf = delta.clone().requires_grad_(True)
    probs = bradley_terry_probs(delta_leaf, th.zeros_like(delta_leaf))
    losses = preference_nll_per_sample(probs, smooth_labels)
    (coeff,) = th.autograd.grad(losses.sum(), delta_leaf)

    def delta_of(pair):
        return (
            fragment_avg_rewards(member, [pair.frag1])[0]
            - fragment_avg_rewards(member, [pair.frag2])[0]
        )

    jac = _flat_grads(delta_of, pairs, params)      # grad_theta Delta_i, one per row
    return coeff.detach().unsqueeze(1) * jac


def demonstration_sample_gradients(
    member, expert_trajs, model_trajs, params
) -> th.Tensor:
    """One gradient per demonstration, under demo_2.

    demo_2 does not decompose, because its partition term contains the experts too.
    Sample i is the loss of that one demonstration against the frozen rollout. The
    rollout is held fixed: it is not feedback.
    """
    with th.no_grad():
        r_e = fragment_sum_rewards(member, expert_trajs)
        r_m = fragment_sum_rewards(member, model_trajs)

    def return_of(traj):
        return fragment_sum_rewards(member, [traj])[0]

    jac_expert = _flat_grads(return_of, expert_trajs, params)   # (N_d, P)
    jac_model = _flat_grads(return_of, model_trajs, params)     # (|M|, P)

    n_d, n_m = r_e.shape[0], r_m.shape[0]

    # row i = softmax over [R^M..., R_i^E]; the expert is the last element
    logits = th.cat([r_m.unsqueeze(0).expand(n_d, n_m), r_e.unsqueeze(1)], dim=1)
    weights = th.softmax(logits, dim=1)

    return (weights[:, -1:] - 1.0) * jac_expert + weights[:, :n_m] @ jac_model


def estimate_alpha(
    member,
    params: List[th.Tensor],
    pref_batch,
    smooth_labels,
    expert_trajs,
    model_trajs,
    batch_size_pref: int,
    batch_size_expert: int,
    min_prefs: int,
    eps: float,
) -> AlphaEstimate:
    """Weight on the demonstrations, estimated at the current parameters.

    Called BEFORE the gradient steps of the iteration, not after: the weight has
    to describe the point where it will be applied.
    """
    n_pref = 0 if pref_batch is None else len(pref_batch.fragment_pairs)
    if n_pref < min_prefs:
        return AlphaEstimate(alpha=1.0, pref=None, demo=None, pinned=True)

    pref_grads = preference_sample_gradients(member, pref_batch, smooth_labels, params)
    demo_grads = demonstration_sample_gradients(member, expert_trajs, model_trajs, params)

    pref = _dispersion(pref_grads, min(batch_size_pref, pref_grads.shape[0]), eps)
    demo = _dispersion(demo_grads, min(batch_size_expert, demo_grads.shape[0]), eps)
    if pref is None or demo is None:
        return AlphaEstimate(alpha=1.0, pref=pref, demo=demo, pinned=True)

    total = pref.cv2 + demo.cv2
    if not np.isfinite(total) or total <= eps:
        return AlphaEstimate(alpha=1.0, pref=pref, demo=demo, pinned=True)

    alpha = float(np.clip(pref.cv2 / total, 0.0, 1.0))
    return AlphaEstimate(alpha=alpha, pref=pref, demo=demo, pinned=False)
