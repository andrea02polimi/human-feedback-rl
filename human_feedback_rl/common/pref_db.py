"""
Preference database — PrefDB and PrefBuffer.

"""

import copy
import gzip
import pickle
import queue
import time
import zlib
from threading import Lock, Thread
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import wandb


# ─────────────────────────────────────────────────────────────────────────────
# Compressed dictionary
# ─────────────────────────────────────────────────────────────────────────────


class _CompressedDict:
    """Internal dictionary that stores values compressed with zlib + pickle."""

    def __init__(self):
        self._store: Dict[int, bytes] = {}

    def __getitem__(self, key: int):
        return pickle.loads(zlib.decompress(self._store[key]))

    def __setitem__(self, key: int, value) -> None:
        self._store[key] = zlib.compress(pickle.dumps(value))

    def __delitem__(self, key: int) -> None:
        del self._store[key]

    def __iter__(self) -> Iterator[int]:
        return iter(self._store)

    def __len__(self) -> int:
        return len(self._store)

    def keys(self):
        return self._store.keys()


# ─────────────────────────────────────────────────────────────────────────────
# PrefDB
# ─────────────────────────────────────────────────────────────────────────────


class PrefDB:
    """
    Circular database storing preferences over pairs of segments.

    Segments are stored compressed; reference counting ensures that a segment
    is deleted only when no preference still references it.

    Args:
        maxlen: maximum number of preference pairs to keep; when exceeded the
                oldest pair (and its segments if unreferenced) are evicted.
    """

    def __init__(self, maxlen: int):
        self.segments: _CompressedDict = _CompressedDict()
        self.segment_references: Dict[int, int] = {}
        self.preferences: List[Tuple[int, int, object]] = []
        self.maxlen = maxlen

    # ------------------------------------------------------------------

    def append(self, seg1_frames, seg1_actions, seg2_frames, seg2_actions, preference) -> None:
        """Add a labeled pair to the database.

        Segments are stored as (frames, actions) tuples so that an
        action-conditioned reward predictor can distinguish which action was
        taken at each observation.
        """
        seg1_frames  = np.asarray(seg1_frames)
        seg1_actions = np.asarray(seg1_actions)
        seg2_frames  = np.asarray(seg2_frames)
        seg2_actions = np.asarray(seg2_actions)

        # Include actions in the key so that two segments with identical
        # observations but different actions are stored separately.
        key1 = zlib.adler32(seg1_frames.tobytes() + seg1_actions.tobytes())
        key2 = zlib.adler32(seg2_frames.tobytes() + seg2_actions.tobytes())

        for key, frames, actions in [
            (key1, seg1_frames, seg1_actions),
            (key2, seg2_frames, seg2_actions),
        ]:
            if key not in self.segments.keys():
                self.segments[key] = (frames, actions)
                self.segment_references[key] = 1
            else:
                self.segment_references[key] += 1

        self.preferences.append((key1, key2, preference))

        if len(self.preferences) > self.maxlen:
            self._delete_first()

    def _delete_first(self) -> None:
        self.delete_preference(0)

    def delete_preference(self, index: int) -> None:
        """Remove preference at index and decrement segment reference counts."""
        if index >= len(self.preferences):
            raise IndexError(f"Preference {index} does not exist")

        key1, key2, _ = self.preferences[index]

        for key in [key1, key2]:
            if self.segment_references[key] == 1:
                del self.segments[key]
                del self.segment_references[key]
            else:
                self.segment_references[key] -= 1

        del self.preferences[index]

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.preferences)

    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        with gzip.open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "PrefDB":
        with gzip.open(path, "rb") as f:
            return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# PrefBuffer
# ─────────────────────────────────────────────────────────────────────────────


class PrefBuffer:
    """
    Background-thread receiver that routes incoming preferences into
    PrefDB train/val splits (80/20).

    Args:
        db_train:  PrefDB for training preferences
        db_val:    PrefDB for validation preferences
        log_dir:   optional directory for TensorBoard logging
    """

    def __init__(
        self,
        db_train: PrefDB,
        db_val: PrefDB,
        shared_steps=None,
    ):
        self.train_db = db_train
        self.val_db = db_val
        self.lock = Lock()
        self._stop_flag = False
        self._thread: Optional[Thread] = None
        self.step = 0
        self._shared_steps = shared_steps
        self._oracle_received = 0
        self._demo_received = 0

    # ------------------------------------------------------------------

    def start_recv_thread(self, pref_queue) -> None:
        """Start the background thread that receives from pref_queue."""
        self._stop_flag = False
        self._thread = Thread(
            target=self._recv_preferences,
            args=(pref_queue,),
            daemon=True,
        )
        self._thread.start()

    def stop_recv_thread(self) -> None:
        """Signal the background thread to stop and wait for it."""
        self._stop_flag = True
        if self._thread is not None:
            self._thread.join()

    # ------------------------------------------------------------------

    def _recv_preferences(self, pref_queue) -> None:
        received = 0

        while not self._stop_flag:
            try:
                seg1_frames, seg1_actions, seg2_frames, seg2_actions, preference, source = pref_queue.get(timeout=1)
            except queue.Empty:
                continue

            received += 1
            if source == "demo":
                self._demo_received += 1
            else:
                self._oracle_received += 1

            # 80 / 20 train / val split based on configured maxlens
            validation_fraction = self.val_db.maxlen / (
                self.val_db.maxlen + self.train_db.maxlen
            )

            with self.lock:
                self.step += 1
                if np.random.rand() < validation_fraction:
                    self.val_db.append(seg1_frames, seg1_actions, seg2_frames, seg2_actions, preference)
                else:
                    self.train_db.append(seg1_frames, seg1_actions, seg2_frames, seg2_actions, preference)

                if wandb.run is not None:
                    a2c_step = self._shared_steps.value if self._shared_steps is not None else 0
                    wandb.log({
                        "prefs/train_db_size":   len(self.train_db),
                        "prefs/val_db_size":     len(self.val_db),
                        "prefs/pref_db_size":    len(self.train_db) + len(self.val_db),
                        "prefs/total_received":  received,
                        "prefs/oracle_received": self._oracle_received,
                        "prefs/demo_received":   self._demo_received,
                        "a2c_step":              a2c_step,
                    })

    # ------------------------------------------------------------------

    def train_db_len(self) -> int:
        return len(self.train_db)

    def val_db_len(self) -> int:
        return len(self.val_db)

    def get_dbs(self) -> Tuple[PrefDB, PrefDB]:
        """Return deep copies of the train and val databases (thread-safe)."""
        with self.lock:
            train_copy = copy.deepcopy(self.train_db)
            val_copy = copy.deepcopy(self.val_db)
        return train_copy, val_copy

    def wait_until_len(self, minimum_length: int) -> None:
        """Block until train_db has at least minimum_length entries."""
        while True:
            with self.lock:
                train_len = len(self.train_db)
                val_len = len(self.val_db)
            if train_len >= minimum_length and val_len > 0:
                break
            print(f"Waiting for preferences; {train_len} collected")
            time.sleep(5.0)
