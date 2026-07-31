"""Per-iteration statistics of the hybrid gradient channels.

The hybrid step composes two gradients on one shared reward net: ``g_pref``
from the Bradley-Terry loss and ``g_demo`` from the demonstration IRL loss.
The two are tracked SEPARATELY, plus ``g_total`` for the composed gradient
the optimizer actually receives. These accumulators answer, once per
reward-training iteration, how noisy each channel is and whether the two pull
in the same direction:

* ``var``            — total variance of the channel, tr(Cov), estimated over
  the gradient steps of the iteration;
* ``sq_norm``        — mean squared norm of the per-step gradients;
* ``mean_sq_norm``   — squared norm of the mean gradient;
* ``var_ratio``      — var / mean_sq_norm, a signal-to-noise ratio: large
  means the channel's per-step gradients are dominated by sampling noise
  rather than by a consistent direction. Already invariant to rescaling the
  channel by a constant, so it is directly comparable across channels whose
  norms differ by orders of magnitude;
* ``dir_var``        — the same variance after dividing EACH gradient by its
  own norm: dispersion of direction alone, with magnitude fluctuations
  removed. Reported only for the variance, because the other quantities
  degenerate under that normalization (see :class:`GradientChannelStats`);
* ``cosine``         — cos of the angle between the two channels' gradients,
  negative when the demonstration and preference signals conflict. Invariant
  to per-channel rescaling, so it is the same raw or normalized.

The population is the ``gradient_steps_rew`` gradients taken within one
iteration, for one ensemble member. The parameters move between those steps,
so the estimate mixes minibatch noise with movement along the loss landscape;
it is a diagnostic of the optimization as it actually runs, not an unbiased
estimate of the gradient estimator's variance at a fixed parameter point.

A second estimator (``reward/grad_adam_*``) reads the variance off Adam's own
exponential moving averages instead — see :class:`AdamVarianceEstimator`. Its
state persists for the whole run rather than resetting each iteration, so the
two estimators answer different questions and are logged side by side.

The gradients are already computed and flattened by the hybrid step, so
accumulating them costs a few vector operations and no extra backward pass.
"""

import math
from typing import Dict, List, Optional

import torch as th


class _Welford:
    """Streaming mean vector and total variance of a stream of vectors.

    Welford's algorithm: it keeps a running mean and a running sum of squared
    deviations, so the history is never stored and the variance does not lose
    precision to the cancellation of ``E[||x||^2] - ||E[x]||^2``.
    """

    def __init__(self) -> None:
        self.count = 0
        self._mean: Optional[th.Tensor] = None
        self._sum_squared_deviations = 0.0
        self._sum_squared_norms = 0.0

    def update(self, vector: th.Tensor) -> None:
        x = vector.detach().reshape(-1).to(th.float64)
        self._sum_squared_norms += float(x.dot(x))
        if self._mean is None:
            self._mean = th.zeros_like(x)
        self.count += 1
        delta = x - self._mean
        self._mean += delta / self.count
        self._sum_squared_deviations += float(delta.dot(x - self._mean))

    @property
    def mean(self) -> Optional[th.Tensor]:
        return self._mean

    @property
    def sq_norm(self) -> float:
        """Mean squared norm of the individual vectors."""
        if self.count == 0:
            return math.nan
        return self._sum_squared_norms / self.count

    @property
    def mean_sq_norm(self) -> float:
        """Squared norm of the mean vector."""
        if self._mean is None:
            return math.nan
        return float(self._mean.dot(self._mean))

    @property
    def var(self) -> float:
        """Total variance tr(Cov), unbiased (ddof=1)."""
        if self.count < 2:
            return math.nan
        return self._sum_squared_deviations / (self.count - 1)


class GradientChannelStats:
    """One gradient channel, measured raw and direction-only.

    Two parallel accumulators over the same stream:

    * ``raw`` — the gradients as they come out of ``backward()``. Its
      ``var / ||mean||^2`` is already invariant to rescaling the whole channel
      by a constant, and it counts BOTH direction and magnitude fluctuations.
    * ``unit`` — each gradient divided by its own norm. Its variance is the
      purely DIRECTIONAL dispersion, discarding magnitude fluctuations
      entirely. For unit vectors ``tr(Cov) = 1 - R^2`` (up to the ddof
      factor), where ``R`` is the length of the mean direction.

    The other three quantities are deliberately NOT reported on the
    normalized stream: ``||u||^2`` is identically 1, and
    ``var / ||mean u||^2 = 1/R^2 - 1`` is a reparametrization of the
    directional variance, so both would be redundant.
    """

    def __init__(self) -> None:
        self.raw = _Welford()
        self.unit = _Welford()

    def update(self, gradient: th.Tensor) -> None:
        # float64 throughout: the deviations are small next to the gradient
        # norms late in training, where float32 accumulation drifts.
        g = gradient.detach().reshape(-1).to(th.float64)
        self.raw.update(g)
        norm = float(g.norm())
        if norm > 0:  # a vanished gradient has no direction to record
            self.unit.update(g / norm)

    @property
    def count(self) -> int:
        return self.raw.count

    @property
    def mean(self) -> Optional[th.Tensor]:
        return self.raw.mean

    @property
    def mean_direction(self) -> Optional[th.Tensor]:
        """Mean of the unit gradients: directions weighted equally.

        Unlike ``mean``, which is a magnitude-weighted average of directions,
        this cannot be dragged around by a few unusually large steps.
        """
        return self.unit.mean

    @property
    def sq_norm(self) -> float:
        return self.raw.sq_norm

    @property
    def mean_sq_norm(self) -> float:
        return self.raw.mean_sq_norm

    @property
    def var(self) -> float:
        return self.raw.var

    @property
    def dir_var(self) -> float:
        """Directional variance: tr(Cov) of the unit gradients."""
        return self.unit.var

    @property
    def var_ratio(self) -> float:
        """var / ||mean gradient||^2."""
        mean_sq_norm = self.mean_sq_norm
        if not mean_sq_norm > 0:
            return math.nan
        return self.var / mean_sq_norm

    @property
    def noise_fraction(self) -> float:
        """var / E[||g||^2]: the share of the gradient's energy that is noise.

        The same information as :attr:`var_ratio` on a bounded scale — the two
        are related by ``noise_fraction = r / (1 + r*(T-1)/T)`` with
        ``r = var_ratio`` — but easier to read: 0.9 means nine tenths of the
        gradient's energy is dispersion around its mean.

        NOT bounded by exactly 1. ``var`` is the unbiased estimate while the
        identity ``E[||g||^2] = ||mean||^2 + var`` holds for the population
        one, so the ceiling is ``T/(T-1)``, reached when the mean gradient
        vanishes. Left uncorrected so the numerator matches ``grad_var_*``.
        """
        sq_norm = self.sq_norm
        if not sq_norm > 0:
            return math.nan
        return self.var / sq_norm

    def metrics(self, suffix: str) -> Dict[str, float]:
        return {
            f"grad_var_{suffix}": self.var,
            f"grad_sq_norm_{suffix}": self.sq_norm,
            f"grad_mean_sq_norm_{suffix}": self.mean_sq_norm,
            f"grad_var_ratio_{suffix}": self.var_ratio,
            f"grad_noise_fraction_{suffix}": self.noise_fraction,
            f"grad_dir_var_{suffix}": self.dir_var,
        }


class AdamVarianceEstimator:
    """Gradient variance read off Adam's own moment estimates.

    Adam already maintains, per parameter, exponential moving averages of the
    gradient and of its square::

        m_t = b1 m_{t-1} + (1 - b1) g_t
        v_t = b2 v_{t-1} + (1 - b2) g_t^2          (elementwise)
        m_hat = m_t / (1 - b1^t),  v_hat = v_t / (1 - b2^t)
        var_t = v_hat - m_hat^2                     (elementwise)

    Summing ``var_t`` over the parameters gives a tr(Cov) directly comparable
    to :class:`GradientChannelStats`, and the state persists across
    iterations exactly as the optimizer's does.

    Two caveats, both inherent to the estimator rather than to this code:

    * with ``b1 != b2`` the two averages cover very different effective
      windows (~1/(1-b) steps: ~10 and ~1000 at Adam's defaults), so the
      difference compares two timescales and CAN GO NEGATIVE — when the
      gradient GROWS, because the short-window mean has already risen while
      the long-window second moment is still held down by older, smaller
      values. It is reported raw; clipping at zero would bias it. Setting
      ``b1 == b2`` puts both moments on one window, making it a genuine
      variance that Jensen keeps non-negative.
    * at ``t == 1`` the bias correction makes ``m_hat = g`` and
      ``v_hat = g^2``, so the variance is exactly zero whatever the betas.
    """

    def __init__(self, beta1: float, beta2: float) -> None:
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError(f"betas must be in [0, 1), got ({beta1}, {beta2}).")
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.count = 0
        self._m: Optional[th.Tensor] = None
        self._v: Optional[th.Tensor] = None

    def update(self, gradient: th.Tensor) -> None:
        g = gradient.detach().reshape(-1).to(th.float64)
        if self._m is None:
            self._m = th.zeros_like(g)
            self._v = th.zeros_like(g)
        self.count += 1
        self._m.mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)
        self._v.mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)

    def _corrected(self):
        """Bias-corrected (m_hat, v_hat), or (None, None) before any update."""
        if self._m is None:
            return None, None
        m_hat = self._m / (1.0 - self.beta1 ** self.count)
        v_hat = self._v / (1.0 - self.beta2 ** self.count)
        return m_hat, v_hat

    @property
    def var(self) -> float:
        """tr(Cov): the elementwise variance summed over the parameters."""
        m_hat, v_hat = self._corrected()
        if m_hat is None:
            return math.nan
        return float((v_hat - m_hat * m_hat).sum())

    @property
    def sq_norm(self) -> float:
        """Running estimate of E[||g||^2], i.e. v_hat summed."""
        _, v_hat = self._corrected()
        return math.nan if v_hat is None else float(v_hat.sum())

    @property
    def mean_sq_norm(self) -> float:
        """||m_hat||^2, the squared norm of the running mean gradient."""
        m_hat, _ = self._corrected()
        return math.nan if m_hat is None else float(m_hat.dot(m_hat))

    @property
    def var_ratio(self) -> float:
        mean_sq_norm = self.mean_sq_norm
        if not mean_sq_norm > 0:
            return math.nan
        return self.var / mean_sq_norm

    def metrics(self, prefix: str, suffix: str) -> Dict[str, float]:
        return {
            f"{prefix}_var_{suffix}": self.var,
            f"{prefix}_sq_norm_{suffix}": self.sq_norm,
            f"{prefix}_mean_sq_norm_{suffix}": self.mean_sq_norm,
            f"{prefix}_var_ratio_{suffix}": self.var_ratio,
        }


class AdamGradientStats:
    """Both channels' Adam-style estimators, in two beta variants.

    One instance per ensemble member, living for the whole run: unlike the
    per-iteration Welford statistics, Adam's averages are meaningful only if
    they are never reset.

    ``grad_adam_*``    uses the betas of the Adam optimizer actually training
                       the reward model, so it describes that optimizer.
    ``grad_adam_eq_*`` uses a single beta for both moments, which makes
                       ``v_hat - m_hat^2`` a clean, non-negative variance over
                       one window at the cost of no longer matching Adam.
    """

    CHANNELS = ("pref", "demo", "total")

    def __init__(self, betas=(0.9, 0.999), eq_beta: float = 0.99) -> None:
        beta1, beta2 = betas
        self.variants = {
            "grad_adam": {
                channel: AdamVarianceEstimator(beta1, beta2)
                for channel in self.CHANNELS
            },
            "grad_adam_eq": {
                channel: AdamVarianceEstimator(eq_beta, eq_beta)
                for channel in self.CHANNELS
            },
        }

    def update(
        self,
        g_pref: Optional[th.Tensor] = None,
        g_demo: Optional[th.Tensor] = None,
        g_total: Optional[th.Tensor] = None,
    ) -> None:
        gradients = {"pref": g_pref, "demo": g_demo, "total": g_total}
        for channels in self.variants.values():
            for channel, gradient in gradients.items():
                if gradient is not None:
                    channels[channel].update(gradient)

    def metrics(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for prefix, channels in self.variants.items():
            for suffix, estimator in channels.items():
                if estimator.count:
                    metrics.update(estimator.metrics(prefix, suffix))
        return metrics


class HybridGradientStats:
    """The two channels plus the angle between them, over one iteration.

    ``adam``, when given, is the run-long :class:`AdamGradientStats` of the
    same ensemble member: this object forwards every recorded gradient to it
    and merges its metrics in, so the two estimators always see exactly the
    same gradients.
    """

    def __init__(self, adam: Optional[AdamGradientStats] = None) -> None:
        self.pref = GradientChannelStats()
        self.demo = GradientChannelStats()
        # The gradient actually handed to the optimizer: g_pref + scale*g_demo
        # in the hybrid case, and simply the one channel in the degenerate
        # arms. Its noise is what the update step really experiences.
        self.total = GradientChannelStats()
        self._cosines: List[float] = []
        self._adam = adam

    def update(
        self,
        g_pref: Optional[th.Tensor] = None,
        g_demo: Optional[th.Tensor] = None,
        g_total: Optional[th.Tensor] = None,
    ) -> None:
        """Record one step. Any channel may be absent (degenerate arms)."""
        if g_pref is not None:
            self.pref.update(g_pref)
        if g_demo is not None:
            self.demo.update(g_demo)
        if g_total is not None:
            self.total.update(g_total)
        if g_pref is not None and g_demo is not None:
            cosine = _cosine(g_pref, g_demo)
            if cosine is not None:
                self._cosines.append(cosine)
        if self._adam is not None:
            self._adam.update(g_pref=g_pref, g_demo=g_demo, g_total=g_total)

    @property
    def count(self) -> int:
        return max(self.pref.count, self.demo.count, self.total.count)

    def metrics(self) -> Dict[str, float]:
        """Metric name -> value for this member's iteration.

        Names carry no logger prefix; the algorithm adds ``reward/``.
        """
        metrics: Dict[str, float] = {}
        if self.pref.count:
            metrics.update(self.pref.metrics("pref"))
        if self.demo.count:
            metrics.update(self.demo.metrics("demo"))
        if self.total.count:
            metrics.update(self.total.metrics("total"))
        if self._cosines:
            cosines = th.tensor(self._cosines, dtype=th.float64)
            metrics["grad_cosine"] = float(cosines.mean())
            # Spread of the per-step angle: a mean near zero can mean either
            # consistently orthogonal channels or two that alternate between
            # agreeing and conflicting, and those are different failures.
            metrics["grad_cosine_std"] = (
                float(cosines.std(unbiased=True)) if cosines.numel() > 1 else math.nan
            )
        if self.pref.mean is not None and self.demo.mean is not None:
            cosine = _cosine(self.pref.mean, self.demo.mean)
            if cosine is not None:
                metrics["grad_cosine_of_means"] = cosine
        # Same angle between the two channels' systematic components, but with
        # every step weighted equally instead of by its gradient magnitude:
        # a handful of unusually large steps cannot set the sign on their own.
        if self.pref.mean_direction is not None and self.demo.mean_direction is not None:
            cosine = _cosine(self.pref.mean_direction, self.demo.mean_direction)
            if cosine is not None:
                metrics["grad_dir_cosine_of_means"] = cosine
        if self._adam is not None:
            metrics.update(self._adam.metrics())
        return metrics


def _cosine(a: th.Tensor, b: th.Tensor) -> Optional[float]:
    """Cosine of the angle between two gradients, or None if one vanishes."""
    a = a.detach().reshape(-1).to(th.float64)
    b = b.detach().reshape(-1).to(th.float64)
    norms = float(a.norm()) * float(b.norm())
    if not norms > 0:
        return None
    return float(a.dot(b) / norms)


def average_metrics(per_member: List[Dict[str, float]]) -> Dict[str, float]:
    """Average each metric across ensemble members, ignoring non-finite values.

    Averaging per-member statistics rather than pooling the members' gradients
    keeps disagreement *between* ensemble members out of the within-member
    variance being reported.
    """
    averaged: Dict[str, float] = {}
    keys = {key for metrics in per_member for key in metrics}
    for key in sorted(keys):
        values = [
            metrics[key]
            for metrics in per_member
            if key in metrics and math.isfinite(metrics[key])
        ]
        if values:
            averaged[key] = sum(values) / len(values)
    return averaged
