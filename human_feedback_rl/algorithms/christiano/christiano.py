"""
ChristianoRLHF — Christiano et al. (2017) RLHF algorithm.

Like SB3 algorithms, all configuration parameters is passed to __init__ constructor.
Call train(output_dir) to start the full asynchronous pipeline.
"""

import functools
import multiprocessing as mp
import os
import signal
import time
from multiprocessing import Process, Queue
from pathlib import Path

import wandb
from omegaconf import OmegaConf
from tqdm import tqdm

from human_feedback_rl.algorithms.christiano.policy_worker import (
    _policy_worker,
    _preference_worker,
    _demonstration_worker,
    _demo_preference_worker,
    _set_thread_limits,
)
from human_feedback_rl.common.pref_db import PrefDB, PrefBuffer
from human_feedback_rl.common.demo_db import DemoDatabase
from human_feedback_rl.common.reward_predictor.ensemble import RewardPredictorEnsemble
from human_feedback_rl.common.reward_predictor.networks import SumoRewardNetwork
from human_feedback_rl.common.utils.env_setup import build_single_env
from human_feedback_rl.common.utils.checkpoints import keep_latest_checkpoints, drain_demo_pipe


class ChristianoRLHF:
    """
    Christiano et al. (2017) RLHF algorithm.

    Like SB3 algorithms, all configuration parameters is passed to __init__ constructor.
    Call train(output_dir) to start the full asynchronous pipeline.

    Three concurrent processes:
      Policy process    — SB3 A2C rollouts + segment generation
      Preference process — preference labeling via oracle
      Main process      — PrefDB management + reward predictor training

    Args:
        expert_model_path: path to the expert model directory (relative to project root)
        seed: random seed
        n_envs: number of parallel environments for the policy worker
        device: torch device ("cpu" or "cuda")

        oracle: preference labeling mode — "env_reward" | "qnet" | "human"
        label_mode: "hard" (original Christiano) or "soft" (softmax over scores)

        use_demonstrations: enable margin ranking loss on expert vs agent demos
        use_demo_preferences: treat expert-correction demos as preference pairs (1.0, 0.0)

        n_reward_predictors: ensemble size (paper: 3)
        rp_lr: reward predictor Adam learning rate
        rp_val_interval: validate reward predictor every N gradient steps
        demo_weight: weight of demo margin loss relative to preference loss
        demo_margin: margin for the ranking loss

        policy_lr: A2C learning rate
        gamma: discount factor
        rollout_steps: A2C n_steps per update
        entropy_coef: A2C entropy coefficient
        value_coef: A2C value function coefficient
        max_gradient_norm: A2C gradient clipping

        initial_prefs: number of preferences to collect before pretraining RP
        segment_len: number of frames per segment
        max_segs: maximum segments in circular buffer
        db_train_maxlen: maximum entries in train PrefDB
        db_val_maxlen: maximum entries in val PrefDB
        seg_pipe_maxsize: max segments queued in segment_pipe
        demo_seg_pipe_maxsize: max segments queued in agent_demo_pipe
        demo_db_maxlen: max entries in DemoDatabase (use_demonstrations path)
        disagreement_candidates: candidate pairs for disagreement-based selection
        max_query_interval: max sleep between queries for annealing (seconds)

        total_env_steps: A2C training budget (Phase 2+)
        rp_reload_interval: reload RP checkpoint every N A2C gradient steps
        policy_save_interval: save policy checkpoint every N A2C gradient steps

        wandb_project: wandb project name
        wandb_entity: wandb entity (username/org), None for default
        wandb_tags: list of tags for wandb run
    """

    def __init__(
        self,
        expert_model_path: str,
        seed: int = 0,
        n_envs: int = 4,
        device: str = "cpu",
        # Oracle
        oracle: str = "env_reward",
        label_mode: str = "hard",
        oracle_temperature: float = 1.0,
        # Demo modes
        use_demonstrations: bool = False,
        use_demo_preferences: bool = False,
        # Reward predictor
        n_reward_predictors: int = 3,
        rp_lr: float = 2e-4,
        rp_val_interval: int = 50,
        demo_weight: float = 1.0,
        demo_margin: float = 1.0,
        # Policy (A2C)
        policy_lr: float = 7e-4,
        gamma: float = 0.99,
        rollout_steps: int = 20,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_gradient_norm: float = 0.5,
        # Preferences
        initial_prefs: int = 500,
        segment_len: int = 25,
        max_segs: int = 1000,
        db_train_maxlen: int = 3000,
        db_val_maxlen: int = 750,
        seg_pipe_maxsize: int = 100,
        demo_seg_pipe_maxsize: int = 100,
        demo_db_maxlen: int = 1500,
        disagreement_candidates: int = 20,
        max_query_interval: float = 5.0,
        # Training
        total_env_steps: int = 1_000_000,
        rp_reload_interval: int = 50,
        rp_retrain_min_new_prefs: int = 50,
        policy_save_interval: int = 100,
        # Performance: max PyTorch intra-op threads per process.
        # Each spawned subprocess (policy, preference, demo) uses this limit,
        # preventing thread-pool contention when multiple processes share few cores.
        # Rule of thumb: floor(n_cores / n_processes). With 9 cores and 3 processes: 3.
        torch_num_threads: int = 2,
        # wandb
        wandb_project: str = "sumo-rlhf",
        wandb_entity: str = None,
        wandb_tags: list = None,
    ):
        self.expert_model_path    = expert_model_path
        self.seed                 = seed
        self.n_envs               = n_envs
        self.device               = device
        self.oracle               = oracle
        self.label_mode           = label_mode
        self.oracle_temperature   = oracle_temperature
        self.use_demonstrations   = use_demonstrations
        self.use_demo_preferences = use_demo_preferences
        self.n_reward_predictors  = n_reward_predictors
        self.rp_lr                = rp_lr
        self.rp_val_interval      = rp_val_interval
        self.demo_weight          = demo_weight
        self.demo_margin          = demo_margin
        self.policy_lr            = policy_lr
        self.gamma                = gamma
        self.rollout_steps        = rollout_steps
        self.entropy_coef         = entropy_coef
        self.value_coef           = value_coef
        self.max_gradient_norm    = max_gradient_norm
        self.initial_prefs        = initial_prefs
        self.segment_len          = segment_len
        self.max_segs             = max_segs
        self.db_train_maxlen      = db_train_maxlen
        self.db_val_maxlen        = db_val_maxlen
        self.seg_pipe_maxsize     = seg_pipe_maxsize
        self.demo_seg_pipe_maxsize = demo_seg_pipe_maxsize
        self.demo_db_maxlen       = demo_db_maxlen
        self.disagreement_candidates = disagreement_candidates
        self.max_query_interval   = max_query_interval
        self.total_env_steps             = total_env_steps
        self.rp_reload_interval          = rp_reload_interval
        self.rp_retrain_min_new_prefs    = rp_retrain_min_new_prefs
        self.policy_save_interval        = policy_save_interval
        self.torch_num_threads    = torch_num_threads
        self.wandb_project        = wandb_project
        self.wandb_entity         = wandb_entity
        self.wandb_tags           = wandb_tags or []

    def _build_config_dict(self) -> dict:
        return {
            "seed": self.seed,
            "env": {
                "expert_model": self.expert_model_path,
                "n_envs": self.n_envs,
            },
            "resources": {"device": self.device, "torch_num_threads": self.torch_num_threads},
            "reward_predictor": {
                "lr": self.rp_lr,
                "n_preds": self.n_reward_predictors,
                "val_interval": self.rp_val_interval,
                "demo_weight": self.demo_weight,
                "demo_margin": self.demo_margin,
            },
            "policy": {
                "lr": self.policy_lr,
                "gamma": self.gamma,
                "rollout_steps": self.rollout_steps,
                "entropy_coef": self.entropy_coef,
                "value_coef": self.value_coef,
                "max_gradient_norm": self.max_gradient_norm,
            },
            "training": {
                "reward_predictor_reload_interval": self.rp_reload_interval,
                "rp_retrain_min_new_prefs": self.rp_retrain_min_new_prefs,
                "policy_save_interval": self.policy_save_interval,
                "total_env_steps": self.total_env_steps,
            },
            "preferences": {
                "oracle": self.oracle,
                "label_mode": self.label_mode,
                "oracle_temperature": self.oracle_temperature,
                "use_demonstrations": self.use_demonstrations,
                "use_demo_preferences": self.use_demo_preferences,
                "initial_prefs": self.initial_prefs,
                "segment_len": self.segment_len,
                "max_segs": self.max_segs,
                "db_train_maxlen": self.db_train_maxlen,
                "db_val_maxlen": self.db_val_maxlen,
                "seg_pipe_maxsize": self.seg_pipe_maxsize,
                "demo_seg_pipe_maxsize": self.demo_seg_pipe_maxsize,
                "demo_db_maxlen": self.demo_db_maxlen,
                "disagreement_candidates": self.disagreement_candidates,
                "max_query_interval": self.max_query_interval,
            },
        }

    def train(self, output_dir: str) -> None:
        """Run the full training pipeline."""

        # Limit all CPU thread pools in the main process to avoid competing
        # with worker subprocesses for the same CPU cores.
        _set_thread_limits(self.torch_num_threads)

        config_dict = self._build_config_dict()

        # Save config to output_dir so eval.py / play.py can reload it.
        run_directory = Path(output_dir)
        run_directory.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(OmegaConf.create(config_dict), run_directory / "config.yaml")

        print("\nConfiguration:")
        print(OmegaConf.to_yaml(OmegaConf.create(config_dict)))

        wandb.init(
            project=self.wandb_project,
            entity=self.wandb_entity,
            name=run_directory.name,
            tags=list(self.wandb_tags),
            config=config_dict,
        )
        # Give reward-predictor metrics their own x-axis so they don't
        # conflict with the policy's a2c_steps x-axis (wandb drops
        # out-of-order steps otherwise).
        wandb.define_metric("rp/*",     step_metric="rp_step")
        wandb.define_metric("policy/*", step_metric="a2c_step")
        wandb.define_metric("prefs/*",  step_metric="a2c_step")

        reward_predictor_checkpoint_dir = str(run_directory / "reward_predictor_checkpoints")
        policy_checkpoint_path          = str(run_directory / "models" / "policy_christiano")

        use_demonstrations   = self.use_demonstrations
        use_demo_preferences = self.use_demo_preferences

        # ── Communication channels ────────────────────────────────────────────
        segment_pipe                 = Queue(maxsize=self.seg_pipe_maxsize)
        preference_pipe              = Queue()
        reward_predictor_ready_event = mp.Event()
        shutdown_event               = mp.Event()

        agent_demo_pipe = Queue(maxsize=self.demo_seg_pipe_maxsize) if use_demonstrations else None
        demo_pipe       = Queue()                                    if use_demonstrations else None
        demo_db         = DemoDatabase(maxlen=self.demo_db_maxlen)  if use_demonstrations else None

        if use_demo_preferences and not use_demonstrations:
            agent_demo_pipe = Queue(maxsize=self.demo_seg_pipe_maxsize)

        shared_env_steps     = mp.Value("l", 0)
        a2c_steps            = mp.Value("l", 0)
        policy_metrics_queue = Queue()

        # ── Preference databases ──────────────────────────────────────────────
        train_database      = PrefDB(maxlen=self.db_train_maxlen)
        validation_database = PrefDB(maxlen=self.db_val_maxlen)
        preference_buffer   = PrefBuffer(
            train_database,
            validation_database,
            shared_steps=a2c_steps,
        )
        preference_buffer.start_recv_thread(preference_pipe)

        # ── Reward predictor ──────────────────────────────────────────────────
        cfg_obj         = OmegaConf.create(config_dict)
        temp_env        = build_single_env(cfg_obj)
        observation_dim = temp_env.observation_space.shape[0]
        temp_env.close()

        reward_predictor = RewardPredictorEnsemble(
            core_network=functools.partial(SumoRewardNetwork, obs_dim=observation_dim),
            lr=self.rp_lr,
            n_preds=self.n_reward_predictors,
            log_dir=str(run_directory),
            device=self.device,
        )

        # ── Launch worker processes ───────────────────────────────────────────
        policy_process = Process(
            target=_policy_worker,
            args=(
                config_dict,
                segment_pipe,
                reward_predictor_ready_event,
                shutdown_event,
                reward_predictor_checkpoint_dir,
                policy_checkpoint_path,
                str(run_directory),
                shared_env_steps,
                agent_demo_pipe,
                policy_metrics_queue,
                a2c_steps,
            ),
        )
        preference_process = Process(
            target=_preference_worker,
            args=(
                config_dict,
                segment_pipe,
                preference_pipe,
                reward_predictor_ready_event,
                shutdown_event,
                reward_predictor_checkpoint_dir,
                shared_env_steps,
            ),
        )
        if use_demonstrations:
            demo_process = Process(
                target=_demonstration_worker,
                args=(
                    config_dict,
                    agent_demo_pipe,
                    demo_pipe,
                    shutdown_event,
                ),
            )
        if use_demo_preferences:
            demo_pref_process = Process(
                target=_demo_preference_worker,
                args=(
                    config_dict,
                    agent_demo_pipe,
                    preference_pipe,
                    shutdown_event,
                ),
            )

        policy_process.start()
        preference_process.start()
        if use_demonstrations:
            demo_process.start()
        if use_demo_preferences:
            demo_pref_process.start()

        def _shutdown(*_):
            print("\n[main] Shutting down…", flush=True)
            shutdown_event.set()
            workers = [policy_process, preference_process]
            if use_demonstrations:
                workers.append(demo_process)
            if use_demo_preferences:
                workers.append(demo_pref_process)
            for proc in workers:
                proc.join(timeout=15)
            for proc in workers:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=3)
            preference_buffer.stop_recv_thread()
            print(f"[main] Policy saved to {policy_checkpoint_path}")
            wandb.finish()
            os._exit(0)

        signal.signal(signal.SIGINT, _shutdown)

        # ── Phase 1: collect initial preferences ──────────────────────────────
        target_preferences = self.initial_prefs
        print(f"\n[Phase 1] Collecting {target_preferences} initial preferences…")
        with tqdm(total=target_preferences, desc="preferences", unit="pref", ncols=80) as progress_bar:
            previous_count = 0
            while True:
                current_train_db, current_val_db = preference_buffer.get_dbs()
                current_count = len(current_train_db)
                if current_count > previous_count:
                    progress_bar.update(current_count - previous_count)
                    previous_count = current_count
                if current_count >= target_preferences and len(current_val_db) > 0:
                    break
                time.sleep(2.0)
        train_db, val_db = preference_buffer.get_dbs()
        print(f"  Done — train={len(train_db)}, validation={len(val_db)}")

        # ── Phase 2: pretrain reward predictor ────────────────────────────────
        print("\n[Phase 2] Pretraining reward predictor…")
        if use_demonstrations:
            drain_demo_pipe(demo_pipe, demo_db)
        reward_predictor.train(
            train_db, val_db,
            demo_db=demo_db if (use_demonstrations and len(demo_db) > 0) else None,
            val_interval=self.rp_val_interval,
            demo_weight=self.demo_weight,
            demo_margin=self.demo_margin,
            global_step=0,
        )
        reward_predictor.save()
        keep_latest_checkpoints(reward_predictor_checkpoint_dir)
        reward_predictor_ready_event.set()
        print("  Reward predictor ready — A2C training with predicted rewards unlocked.")

        # ── Phase 3: continuous reward predictor retraining ───────────────────
        print("\n[Phase 3] Reward predictor training continuously…")

        # Retrain RP only when enough new preferences have arrived.
        # This avoids both busy-wait and unnecessary retraining on stale data.
        _rp_retrain_min_new_prefs = self.rp_retrain_min_new_prefs
        rp_retrain_count = 0
        _prefs_at_last_retrain = 0

        while not shutdown_event.is_set():
            while True:
                try:
                    metrics = policy_metrics_queue.get_nowait()
                    metrics["a2c_step"] = a2c_steps.value
                    wandb.log(metrics)
                except Exception:
                    break

            # Use total-ever-received count (monotonically increasing) as the
            # retrain trigger, NOT DB sizes which plateau at maxlen and would
            # permanently stall retraining once the DB is full.
            current_prefs = preference_buffer.step

            train_db, val_db = preference_buffer.get_dbs()

            if current_prefs == 0 or len(val_db) == 0:
                time.sleep(1.0)
                continue

            # Skip retrain if not enough new preferences have arrived.
            if current_prefs - _prefs_at_last_retrain < _rp_retrain_min_new_prefs:
                time.sleep(0.5)
                continue

            if use_demonstrations:
                drain_demo_pipe(demo_pipe, demo_db)

            reward_predictor.train(
                train_db, val_db,
                demo_db=demo_db if (use_demonstrations and len(demo_db) > 0) else None,
                val_interval=self.rp_val_interval,
                demo_weight=self.demo_weight,
                demo_margin=self.demo_margin,
                global_step=a2c_steps.value,
            )
            reward_predictor.save()
            keep_latest_checkpoints(reward_predictor_checkpoint_dir)
            rp_retrain_count += 1
            _prefs_at_last_retrain = current_prefs

            if use_demonstrations and wandb.run is not None:
                wandb.log({"prefs/demo_db_size": len(demo_db)}, step=a2c_steps.value)

            demo_info = f"  demo={len(demo_db)}" if use_demonstrations else ""
            print(
                f"[rp] retrain #{rp_retrain_count}"
                f"  train={len(train_db)}  val={len(val_db)}{demo_info}",
                flush=True,
            )

        # ── Shutdown ──────────────────────────────────────────────────────────
        _shutdown()
