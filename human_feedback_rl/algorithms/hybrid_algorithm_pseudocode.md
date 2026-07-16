# HybridAlgorithm — pseudocodice

Apprendimento di UNA reward net da dimostrazioni + preferenze.
Estende `DemoAlgorithm`. Due meccanismi (`demo_mode`):
  - `"gcl"`         : loss IRL/GCL sulle demo  +  loss Bradley-Terry sulle preferenze,
                      fuse sulla STESSA rete con bilanciamento di norma dei gradienti.
  - `"preferences"` : le demo diventano coppie di preferenza (expert ≻ agent),
                      unico obiettivo BT su batch misti (nessun conflitto di scala).

```text
────────────────────────────────────────────────────────────────────
INIT(...)
  valida gli iperparametri (batch_size_pref>0, train_frac∈(0,1),
      demo_weight≥0, pref_temperature>0, demo_mode∈{gcl,preferences}, ...)
  super().__init__(...)                      # DemoAlgorithm: reward net, IRL loss, agente
  salva parametri (demo_mode, demo_weight, max_balance_scale, balance_eps,
      frazioni batch demo/pref)
  crea:
      fragmenter (a coppie)
      preference_gatherer  (oracolo soft, con pref_temperature propria)
      dataset_train / dataset_val                 # preferenze oracolo
      dataset_demo_prefs_train / _val             # coppie expert-vs-agent (solo mode "preferences")

────────────────────────────────────────────────────────────────────
TRAIN(total_timesteps, timesteps_per_iteration, ...):
  n_iterations = total_timesteps / timesteps_per_iteration
  schedule = build_query_schedule(n_iterations, query_budget)   # #query per iterazione

  SE initial_agent_timesteps > 0:                 # bootstrap opzionale
      raccogli transizioni bootstrap → colleziona feedback iniziale
      allena reward model (bootstrap) → normalizza reward → rinfresca cache replay
      pre-allena l'agente sulla reward appresa

  PER ogni (iteration, num_queries) nello schedule:
      raccogli rollout dell'agente (+ transizioni di esplorazione)
      _collect_feedback(num_queries)
      logga errori di imitazione + snapshot validazione "pre_update"

      _train_reward_model()                        # cuore ibrido
      normalizza reward agente
      logga snapshot "post_update", returns, staleness replay
      (periodicamente) logga scatter return predetti vs veri

      rinfresca cache di relabel del replay
      train_agent(timesteps_per_iteration)         # RL sulla reward corrente
      logga iterazione; salva checkpoint ogni checkpoint_interval
  RITORNA l'agente allenato

────────────────────────────────────────────────────────────────────
_collect_feedback(num_queries):
  _collect_preference_feedback(num_queries)        # coppie oracolo etichettate
  SE demo_mode == "preferences":
      _collect_demo_preference_pairs(demo_pref_pairs_per_iteration)

_collect_preference_feedback(num_queries):
  SE num_queries ≤ 0: return
  fragments = fragmenter(trajectories, len, num_queries)
  preferences = preference_gatherer(fragments)     # etichette oracolo soft
  push_split → dataset_train / dataset_val

_collect_demo_preference_pairs(num_pairs):         # Ibarz et al. 2018
  expert_frags = fragmenta(expert_trajectories)
  agent_frags  = fragmenta(trajectories)
  pairs  = zip(expert, agent);  preferences = [expert≻agent] fisse (nessuna reward)
  push_split → dataset_demo_prefs_train / _val

push_split(train_ds, val_ds, fragments, prefs):
  shuffle; n_train = train_comparison_frac·N; distribuisci in train/val

────────────────────────────────────────────────────────────────────
_train_reward_model():
  SE nessuna traiettoria: return
  SE demo_mode == "preferences": _train_reward_model_pure_preferences()
  ALTRIMENTI:                    _train_reward_model_gcl()

── modalità GCL ─────────────────────────────────────────────────────
_train_reward_model_gcl():
  has_prefs = |dataset_train| > 0
  SE non has_prefs E demo_weight==0: return        # niente da allenare

  definisci member_step(member, optimizer):        # per ogni membro dell'ensemble
      per gradient_steps_rew volte:
          pref_loss = BT loss su batch bootstrap di preferenze   (se has_prefs)
          demo_loss = IRL/GCL loss                               (se demo_weight>0)
          stats += _reward_step(member, optimizer, pref_loss, demo_loss)

  esegui member_step su tutti i membri
  logga diagnostica loss/maxent/preferenze + statistiche step ibride + norma pesi

── modalità PREFERENCES ─────────────────────────────────────────────
_train_reward_model_pure_preferences():
  ripartisci batch: n_demo = frazione·batch (coppie expert-vs-agent),
                    n_oracle = resto (coppie oracolo)
  member_step: per gradient_steps_rew volte:
      campiona n_oracle da dataset_train + n_demo da dataset_demo_prefs_train
      concatena in un unico PreferenceBatch
      loss = BT loss;  zero_grad → backward → step
  logga diagnostica preferenze + loss media

────────────────────────────────────────────────────────────────────
_reward_step(member, optimizer, pref_loss, demo_loss)  → stats:
  # UN passo di ottimizzatore che compone i due gradienti

  SE entrambi None: return  (no-op)

  SE solo pref_loss:  zero_grad → backward → step            (arm solo-pref)
  SE solo demo_loss:  zero_grad → backward → step            (arm solo-demo)

  # --- entrambi presenti: calcola i due gradienti separatamente ---
  g_pref = grad(pref_loss)
  g_demo = grad(demo_loss)
  appiattisci; pref_norm, demo_norm

  # bilanciamento di norma: porta g_demo a demo_weight·||g_pref||
  scale = min(demo_weight · pref_norm / (demo_norm+eps), max_balance_scale)

  scrivi p.grad = g_pref + scale·g_demo  per ogni parametro
  optimizer.step()
  ritorna stats (loss, scale, norme, grad_norm)

────────────────────────────────────────────────────────────────────
Helper:
  _preference_loss(member, batch):
      r1,r2 = reward media dei frammenti;  return preference_nll(BT(r1,r2), labels)

  diagnostica: rapporto norme demo/pref, accuratezza preferenze train/val,
               ln(2) è il floor atteso della BT loss con etichette soft

  _save_checkpoint_extras: salva demo_mode, demo_weight,
               e i quattro dataset di preferenze
```
