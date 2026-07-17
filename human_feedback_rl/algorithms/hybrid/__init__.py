"""Building blocks of :class:`~human_feedback_rl.algorithms.HybridAlgorithm`.

* ``demonstration_losses`` — the two demonstration IRL losses (demo_1, demo_2)
  and the batch-sampling mixin that dispatches them.
* ``reward_training`` — reward-model optimization helpers (gradient norms,
  agent-facing reward normalization).
* ``reward_diagnostics`` — validation, ranking and replay-buffer diagnostics.
* ``imitation_metrics`` — agent-vs-expert imitation error logging.

The preference side (Bradley-Terry losses, fragmenters, synthetic oracle)
lives in :mod:`human_feedback_rl.common`.
"""
