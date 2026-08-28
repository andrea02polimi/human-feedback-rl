# HybridAlgorithm — pseudocode

Learning ONE reward net from demonstrations and preferences. It is the only
algorithm in the package: with `demo_weight=0` it degenerates into the
preference-only baseline, with `total_queries=0` into the demonstration-only
one.

Two ways of taking in the demonstration signal (`demo_mode`):

- `"gcl"` — the demo loss (`demo_1`, difference of means, or `demo_2`, MaxEnt
  surrogate) and the Bradley-Terry preference loss are fused on the SAME
  network. How they are fused is `gcl_fusion`.
- `"preferences"` — the demonstrations become preference pairs (expert ≻
  agent): a single BT objective over mixed batches, so no scale conflict by
  construction. This is the literature hybrid (Ibarz et al. 2018).

```text
────────────────────────────────────────────────────────────────────
INIT(...)
  validate the hyperparameters (batch_size_pref>0, demo_weight≥0,
      pref_temperature>0, demo_mode ∈ {gcl, preferences},
      gcl_fusion ∈ {norm_balance, alpha_norm_single_adam}, ...)
  super().__init__(...)              # reward net, IRL loss, agent
  split the master seed into four independent streams:
      query   -- which fragments get compared
      oracle  -- oracle labels, Bernoulli draws included
      train   -- preference and demonstration minibatches, bootstrap
      probe   -- the rollout the alpha estimate needs
  build:
      fragmenter (pairs)             random, or active on ensemble disagreement
      preference_gatherer            synthetic oracle, own pref_temperature
      dataset_train                  oracle preferences
      dataset_demo_prefs_train       expert-vs-agent pairs (mode "preferences")

────────────────────────────────────────────────────────────────────
TRAIN(total_timesteps, timesteps_per_iteration, ...):
  n_iterations = total_timesteps / timesteps_per_iteration
  schedule = build_query_schedule(n_iterations, total_queries)

  IF initial_agent_timesteps > 0:                  # optional bootstrap
      collect bootstrap transitions -> _collect_feedback(initial_queries)
      subtract those queries from the first slot of the schedule
      train the reward model -> normalize -> refresh the replay cache
      pre-warm the agent on the learned reward

  FOR each (iteration, num_queries) in the schedule:
      collect the agent rollout (+ exploration transitions)
      _collect_feedback(num_queries)
      log imitation errors and the "pre_update" validation snapshot

      _train_reward_model()                        # the hybrid core
      normalize the agent-facing reward
      log "post_update", outcome returns, replay staleness
      (periodically) log predicted-vs-true return scatter

      refresh the replay relabelling cache
      train_agent(timesteps_per_iteration)         # RL on the current reward
      log the iteration; checkpoint every checkpoint_interval
  RETURN the trained agent

────────────────────────────────────────────────────────────────────
_collect_feedback(num_queries):
  _collect_preference_feedback(num_queries)
  IF demo_mode == "preferences":
      _collect_demo_preference_pairs(demo_pref_pairs_per_iteration)

_collect_preference_feedback(num_queries):
  IF num_queries ≤ 0: return
  fragments   = fragmenter(trajectories, fragment_length, num_queries)
  preferences = preference_gatherer(fragments)     # oracle labels
  dataset_train.push(fragments, preferences)
  _count_duplicate_comparisons(fragments)          # diagnostic only

  Everything collected is trained on: there is no validation split. Feedback
  is the scarce resource, and holding a share back took it away from training
  while no decision really depended on that measurement.

_collect_demo_preference_pairs(num_pairs):         # Ibarz et al. 2018
  expert_frags = fragment(expert_trajectories)
  agent_frags  = fragment(trajectories)
  pairs = zip(expert, agent); labels fixed at expert ≻ agent (no reward used)

────────────────────────────────────────────────────────────────────
_train_reward_model():
  IF no trajectories: return
  IF demo_mode == "preferences": _train_reward_model_pure_preferences()
  ELSE:                          _train_reward_model_gcl()

── GCL mode ─────────────────────────────────────────────────────────
_train_reward_model_gcl():
  has_prefs = |dataset_train| > 0
  IF not has_prefs AND demo_weight == 0: return

  _estimate_alpha()          # BEFORE any step: the weight describes THIS theta

  member_step(member, optimizer):                  # per ensemble member
      alpha = this member's estimate, fixed for the whole iteration
      repeat gradient_steps_rew times:
          pref_loss = BT loss on a preference minibatch    (if has_prefs)
          demo_loss = IRL/GCL loss                         (if demo_weight>0)
          stats += _reward_step(member, optimizer, pref_loss, demo_loss, alpha)

  run member_step on every member
  log the loss diagnostics, the step statistics and the weight norm

── PREFERENCES mode ─────────────────────────────────────────────────
_train_reward_model_pure_preferences():
  split the batch: n_demo = fraction·batch (expert-vs-agent pairs),
                   n_oracle = the rest (oracle pairs)
  member_step: repeat gradient_steps_rew times:
      sample n_oracle from dataset_train + n_demo from dataset_demo_prefs_train
      concatenate into one PreferenceBatch
      loss = BT loss; zero_grad -> backward -> step

────────────────────────────────────────────────────────────────────
_estimate_alpha():                                 # alpha_norm_single_adam only
  # alpha = CV²_pref / (CV²_pref + CV²_demo), the weight on demonstrations
  IF gcl_fusion == "norm_balance": return          # that fusion ignores alpha
  IF demo_weight ≤ 0 or no trajectories: return
  IF loss_type != "demo_2": raise NotImplementedError

  n_pref = comparisons collected
  IF n_pref < ALPHA_MIN_PREFS (5):
      alpha = 1, pinned                            # all weight on demos
  ELSE, per member:
      per-sample gradients for each channel
      V = Σ‖gᵢ - ḡ‖² / (n-1)      # how the generating process scatters
      S = V / B,  B = min(batch_size, N)   # noise of the gradient applied
      CV² = S / ‖ḡ‖²                       # dimensionless
      alpha = CV²_pref / (CV²_pref + CV²_demo)

  The rollout the demo loss needs is drawn ONCE, from the probe stream, and
  shared by every sample: it is not feedback, so its noise must not enter the
  channel's variance, and the estimate must not move the training draws.

────────────────────────────────────────────────────────────────────
_reward_step(member, optimizer, pref_loss, demo_loss, alpha) → stats:
  # ONE optimizer step composing the two gradients

  IF both None: return (no-op)
  IF only one of them: zero_grad -> backward -> step   (single-channel methods)

  g_pref = grad(pref_loss);  g_demo = grad(demo_loss)
  flatten; pref_norm, demo_norm

  IF gcl_fusion == "alpha_norm_single_adam":
      g = (1-alpha)·g_pref/‖g_pref‖ + alpha·g_demo/‖g_demo‖
      # the channel norms are discarded on purpose: only the direction
      # survives, which is why alpha has to be dimensionless
  ELSE:                       # norm_balance
      scale = min(demo_weight · pref_norm / (demo_norm + eps), max_balance_scale)
      g = g_pref + scale·g_demo

  write g into p.grad for every parameter; optimizer.step()
  return stats (losses, alpha or scale, norms, grad_norm)

────────────────────────────────────────────────────────────────────
Helpers:
  _preference_loss(member, batch):
      r1, r2 = mean fragment rewards; preference_nll(BT(r1, r2), labels)

  _smoothed_labels: only for sampled binary labels, which otherwise have their
      optimum at Delta = ±inf; soft labels are left alone

  diagnostics: demo/pref norm ratio, preference accuracy, and the reminder
      that ln(2) is the expected BT floor with soft labels

  _save_checkpoint_extras: demo_mode, demo_weight and the preference datasets
```
