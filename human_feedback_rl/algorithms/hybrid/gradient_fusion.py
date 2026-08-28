"""Turning two gradients into one optimizer step."""

from typing import Dict, List

import numpy as np
import torch as th


class GradientFusionMixin:
    """The two fusion rules: norm balancing and the reliability weight."""

    def _reward_step(self, member, optimizer, pref_loss, demo_loss,
                     alpha=None) -> Dict[str, float]:
        """One optimizer step from the preference and/or demonstration losses.

        With a single channel the loss goes straight to ``backward()``. With
        both, each gradient is computed on its own and the two are combined the
        way ``gcl_fusion`` says.
        """
        nan = float("nan")
        stats = {
            "pref_loss": nan, "demo_loss": nan, "scale": nan, "pref_norm": nan,
            "demo_norm": nan, "grad_norm": nan, "alpha": nan,
        }

        if pref_loss is None and demo_loss is None:
            return stats
        if demo_loss is None:
            return self._single_channel_step(member, optimizer, pref_loss, "pref_loss", stats)
        if pref_loss is None:
            return self._single_channel_step(member, optimizer, demo_loss, "demo_loss", stats)

        params = list(member.parameters())
        g_pref = self._channel_gradient(optimizer, pref_loss, params)
        g_demo = self._channel_gradient(optimizer, demo_loss, params)

        flat_pref = self._flatten(g_pref, params)
        flat_demo = self._flatten(g_demo, params)
        pref_norm = float(flat_pref.norm())
        demo_norm = float(flat_demo.norm())

        if self.gcl_fusion == "alpha_norm_single_adam":
            fused = self._fuse_by_reliability_weight(
                member, params, flat_pref, flat_demo, pref_norm, demo_norm, alpha)
        else:
            fused = self._fuse_by_norm_balance(
                member, params, g_pref, g_demo, pref_norm, demo_norm)
        optimizer.step()

        stats.update(
            pref_loss=float(pref_loss.detach()),
            demo_loss=float(demo_loss.detach()),
            pref_norm=pref_norm, demo_norm=demo_norm,
            **fused,
        )
        return stats

    def _single_channel_step(self, member, optimizer, loss, loss_key: str,
                             stats: Dict[str, float]) -> Dict[str, float]:
        """The one-channel baselines: a single loss, a single backward."""
        optimizer.zero_grad()
        loss.backward()
        stats.update({loss_key: float(loss.detach()), "grad_norm": self._grad_norm(member)})
        optimizer.step()
        return stats

    @staticmethod
    def _channel_gradient(optimizer, loss, params) -> List:
        """The gradient of one channel alone, detached from the graph."""
        optimizer.zero_grad()
        loss.backward()
        return [None if p.grad is None else p.grad.detach().clone() for p in params]

    def _fuse_by_reliability_weight(self, member, params, flat_pref, flat_demo,
                                    pref_norm, demo_norm, alpha) -> Dict[str, float]:
        """Mix the two unit directions by alpha, for a single Adam step.

        The channel norms are discarded on purpose: only the direction
        survives, which is why alpha has to be dimensionless.
        """
        if alpha is None:
            alpha = self._alpha_weight(member)
        alpha = float(np.clip(alpha, 0.0, 1.0))
        unit_pref = flat_pref / (pref_norm + self.balance_eps)
        unit_demo = flat_demo / (demo_norm + self.balance_eps)
        direction = (1.0 - alpha) * unit_pref + alpha * unit_demo
        self._set_flat_grad(params, direction)
        return {"grad_norm": float(direction.norm()), "alpha": alpha}

    def _fuse_by_norm_balance(self, member, params, g_pref, g_demo,
                              pref_norm, demo_norm) -> Dict[str, float]:
        """Rescale the demo gradient to ``demo_weight`` times the preference one."""
        scale = min(
            self.demo_weight * pref_norm / (demo_norm + self.balance_eps),
            self.max_balance_scale,
        )
        for p, gp, gd in zip(params, g_pref, g_demo):
            if gp is None and gd is None:
                p.grad = None
                continue
            grad = th.zeros_like(p)
            if gp is not None:
                grad += gp
            if gd is not None:
                grad += scale * gd
            p.grad = grad
        return {"scale": scale, "grad_norm": self._grad_norm(member)}

    @staticmethod
    def _set_flat_grad(params, direction: th.Tensor) -> None:
        """Write a flat vector into the ``.grad`` fields for ``step()``."""
        offset = 0
        for p in params:
            k = p.numel()
            p.grad = direction[offset:offset + k].view_as(p).clone()
            offset += k

    @staticmethod
    def _flatten(grads, params) -> th.Tensor:
        parts = [
            g.reshape(-1) if g is not None else th.zeros(p.numel())
            for g, p in zip(grads, params)
        ]
        return th.cat(parts) if parts else th.zeros(0)

    def _log_hybrid_step_stats(self, all_stats: List[Dict[str, float]]) -> None:
        def nanmean(key):
            values = np.asarray([s[key] for s in all_stats], dtype=float)
            finite = values[np.isfinite(values)]
            return float(finite.mean()) if finite.size else None

        pairs = {
            "reward/hybrid_pref_loss": "pref_loss",
            "reward/hybrid_demo_loss": "demo_loss",
            "reward/hybrid_demo_scale": "scale",
            "reward/grad_norm_pref": "pref_norm",
            "reward/grad_norm_demo": "demo_norm",
            "reward/grad_norm": "grad_norm",
        }
        for log_key, stat_key in pairs.items():
            value = nanmean(stat_key)
            if value is not None:
                self.logger.record(log_key, value, exclude="stdout")

        norms = {k: nanmean(k2) for k, k2 in (("pref", "pref_norm"), ("demo", "demo_norm"))}
        if norms["pref"] is not None and norms["demo"] is not None:
            self.logger.record(
                "reward/grad_norm_demo_pref_ratio",
                norms["demo"] / (norms["pref"] + self.balance_eps),
                exclude="stdout",
            )
        grad_norms = [s["grad_norm"] for s in all_stats if np.isfinite(s["grad_norm"])]
        if grad_norms:
            self.logger.record("reward/grad_norm_max", float(np.max(grad_norms)), exclude="stdout")
