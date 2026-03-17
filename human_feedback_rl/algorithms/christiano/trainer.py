"""
ChristianoTrainer — orchestrates the full Christiano et al. (2017) RLHF pipeline.
"""

import functools
import multiprocessing as mp
import os
import signal
import time
from multiprocessing import Process, Queue
from pathlib import Path

import numpy as np
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from human_feedback_rl.algorithms.base_trainer import BaseTrainer
from human_feedback_rl.algorithms.christiano.workers import (
    _policy_worker,
    _preference_worker,
    _demonstration_worker,
)
from human_feedback_rl.feedback.pref_db import PrefDB, PrefBuffer
from human_feedback_rl.feedback.demo_db import DemoDatabase
from human_feedback_rl.reward_models.ensemble import RewardPredictorEnsemble
from human_feedback_rl.reward_models.networks import SumoRewardNetwork
from human_feedback_rl.utils.env_setup import build_single_env
from human_feedback_rl.utils.checkpoints import keep_latest_checkpoints, drain_demo_pipe


class ChristianoTrainer(BaseTrainer):
    """
    Runs the Christiano et al. asynchronous RLHF training pipeline.

    Three concurrent activities:
      Policy process    — SB3 A2C rollouts + segment generation
      Preference process — preference labeling via oracle
      Main process      — PrefDB management + RP training
    """

    def train(self, cfg: DictConfig) -> None:
        """Run the full training pipeline."""

        print("\nConfiguration:")
        print(OmegaConf.to_yaml(cfg))

        run_directory = Path(HydraConfig.get().runtime.output_dir)

        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.get("entity") or None,
            name=run_directory.name,
            tags=list(cfg.wandb.get("tags", [])),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        reward_predictor_checkpoint_dir = str(run_directory / "reward_predictor_checkpoints")
        policy_checkpoint_path          = str(run_directory / "models" / "policy_christiano")

        use_demonstrations = cfg.preferences.use_demonstrations

        # ── Communication channels ────────────────────────────────────────────
        segment_pipe                 = Queue(maxsize=cfg.preferences.seg_pipe_maxsize) # segmenti generati ad alta velocità, consumo lento, scarto i meno recenti
        preference_pipe              = Queue() # non voglio scartare nulla
        reward_predictor_ready_event = mp.Event()   # main → policy: reward predictor ready
        shutdown_event               = mp.Event()   # main → all:    time to stop

        # Demo pipes/DB only created when demonstrations are enabled.
        agent_demo_pipe = Queue(maxsize=cfg.preferences.demo_seg_pipe_maxsize) if use_demonstrations else None
        demo_pipe       = Queue()                                               if use_demonstrations else None
        demo_db         = DemoDatabase(maxlen=cfg.preferences.demo_db_maxlen) if use_demonstrations else None

        # Shared step counters.
        # shared_env_steps: total steps across Phase 1 + Phase 2 (used for query annealing)
        # a2c_steps:        A2C-only steps starting from 0 at Phase 2 (used as wandb X axis)
        shared_env_steps     = mp.Value("l", 0)
        a2c_steps            = mp.Value("l", 0)
        policy_metrics_queue = Queue()   # subprocess → main: A2C metrics for wandb

        # ── Preference databases (owned by main process) ──────────────────────
        train_database      = PrefDB(maxlen=cfg.preferences.db_train_maxlen)
        validation_database = PrefDB(maxlen=cfg.preferences.db_val_maxlen)
        preference_buffer   = PrefBuffer(
            train_database,
            validation_database,
            shared_steps=a2c_steps,
        )
        preference_buffer.start_recv_thread(preference_pipe)

        # ── Reward predictor (trained in main process) ────────────────────────
        temp_env        = build_single_env(cfg)
        observation_dim = temp_env.observation_space.shape[0]
        temp_env.close()

        reward_predictor = RewardPredictorEnsemble(
            core_network=functools.partial(SumoRewardNetwork, obs_dim=observation_dim),
            lr=cfg.reward_predictor.lr,
            n_preds=cfg.reward_predictor.n_preds,
            log_dir=str(run_directory),
            device=cfg.resources.device,
        )

        # ── Launch worker processes ───────────────────────────────────────────
        config_dict = OmegaConf.to_container(cfg, resolve=True)

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

        policy_process.start()
        preference_process.start()
        if use_demonstrations:
            demo_process.start()

        def _shutdown(*_):
            print("\n[main] Shutting down…", flush=True)
            shutdown_event.set()
            workers = [policy_process, preference_process]
            if use_demonstrations:
                workers.append(demo_process)
            for proc in workers:
                proc.join(timeout=15)
            for proc in workers:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=3)
            preference_buffer.stop_recv_thread()
            print(f"[main] Policy saved to {policy_checkpoint_path}")
            wandb.finish()
            os._exit(0)   # bypass atexit so non-daemon workers don't block exit

        signal.signal(signal.SIGINT, _shutdown)

        # ── Phase 1: collect initial preferences (random/untrained policy) ────
        target_preferences = cfg.preferences.initial_prefs
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
            val_interval=cfg.reward_predictor.val_interval,
            demo_weight=cfg.reward_predictor.demo_weight,
            demo_margin=cfg.reward_predictor.demo_margin,
            global_step=0,
        )
        reward_predictor.save()
        keep_latest_checkpoints(reward_predictor_checkpoint_dir)
        reward_predictor_ready_event.set()
        print("  Reward predictor ready — A2C training with predicted rewards unlocked.")

        # ── Phase 3: continuous reward predictor retraining ───────────────────
        print("\n[Phase 3] Reward predictor training continuously…")

        rp_retrain_count = 0
        while not shutdown_event.is_set():
            # Forward any pending A2C metrics from the policy subprocess to wandb.
            while True:
                try:
                    metrics = policy_metrics_queue.get_nowait()
                    wandb.log(metrics, step=a2c_steps.value)
                except Exception:
                    break

            train_db, val_db = preference_buffer.get_dbs()
            if len(train_db) == 0 or len(val_db) == 0:
                time.sleep(1.0)
                continue

            if use_demonstrations:
                drain_demo_pipe(demo_pipe, demo_db)

            reward_predictor.train(
                train_db, val_db,
                demo_db=demo_db if (use_demonstrations and len(demo_db) > 0) else None,
                val_interval=cfg.reward_predictor.val_interval,
                demo_weight=cfg.reward_predictor.demo_weight,
                demo_margin=cfg.reward_predictor.demo_margin,
                global_step=a2c_steps.value,
            )
            reward_predictor.save()
            keep_latest_checkpoints(reward_predictor_checkpoint_dir)
            rp_retrain_count += 1
            demo_info = f"  demo={len(demo_db)}" if use_demonstrations else ""
            print(
                f"[rp] retrain #{rp_retrain_count}"
                f"  train={len(train_db)}  val={len(val_db)}{demo_info}",
                flush=True,
            )

        # ── Shutdown ──────────────────────────────────────────────────────────
        _shutdown()
