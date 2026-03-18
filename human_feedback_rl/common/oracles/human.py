"""
Human preference oracle with side-by-side visual comparison.

"""

import queue
from multiprocessing import Process, Queue
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")   # must be set before any pyplot import
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np

from .base import BaseOracle


# ─────────────────────────────────────────────────────────────────────────────
# Frame generation from vector observations
# ─────────────────────────────────────────────────────────────────────────────


def _segment_to_rgb(frames, title: str, w: int, h: int) -> np.ndarray:
    """
    Render the observation time series of one segment as an (H, W, 3) RGB image.

    Each observation feature is drawn as a separate line. Up to 10 features
    are shown (covers the full highway_discrete_v2 observation space).
    """
    data = np.array(frames, dtype=np.float32)   # (T, obs_dim)
    T, D = data.shape

    dpi = 80
    fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)

    colors = plt.cm.tab10.colors
    for i in range(min(D, 10)):
        ax.plot(data[:, i], color=colors[i % 10], lw=1.0, alpha=0.85, label=f"f{i}")

    ax.set_title(title, fontsize=11, pad=4)
    ax.set_xlabel("step", fontsize=8)
    ax.set_ylabel("obs value", fontsize=8)
    ax.set_ylim(-1.15, 1.15)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.legend(loc="upper right", fontsize=6, ncol=2, framealpha=0.6)
    ax.tick_params(labelsize=7)
    fig.tight_layout(pad=0.6)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
    plt.close(fig)

    return img[:, :, :3].copy()   # (H, W, 3) RGB


def _make_comparison_image(
    seg1_frames,
    seg2_frames,
    frame_w: int = 360,
    frame_h: int = 420,
) -> np.ndarray:
    """
    Produce a side-by-side (H, 2W+4, 3) RGB comparison image.
    The thin grey separator makes the boundary between segments visible.
    """
    img_l = _segment_to_rgb(seg1_frames, "Segment  L", frame_w, frame_h)
    img_r = _segment_to_rgb(seg2_frames, "Segment  R", frame_w, frame_h)
    sep = np.full((frame_h, 4, 3), 180, dtype=np.uint8)
    return np.hstack([img_l, sep, img_r])   # (H, 2W+4, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Pyglet RGB renderer  (subprocess — macOS Cocoa requires main thread)
# ─────────────────────────────────────────────────────────────────────────────


def _rgb_render_worker(img_queue: Queue) -> None:
    """
    Top-level function (required for spawn pickling on macOS).

    Opens a pyglet window and keeps it updated with whatever RGB image arrives
    on img_queue. Sending None is the shutdown sentinel.
    """
    import pyglet

    window: Optional[pyglet.window.Window] = None

    while True:
        try:
            img = img_queue.get(timeout=0.5)
        except queue.Empty:
            if window is not None:
                window.dispatch_events()
            continue

        if img is None:  # shutdown sentinel
            if window is not None:
                window.close()
            return

        h, w = img.shape[:2]

        if window is None:
            window = pyglet.window.Window(
                width=w,
                height=h,
                caption="Preference — L / R / E / S",
            )

        # pyglet expects bottom-up RGB; np.flipud reverses row order
        rgb_bytes = np.ascontiguousarray(np.flipud(img)).tobytes()
        image_data = pyglet.image.ImageData(w, h, "RGB", rgb_bytes)

        window.switch_to()
        window.dispatch_events()
        window.clear()
        image_data.blit(0, 0)
        window.flip()


class _RGBRenderer:
    """Wraps the pyglet subprocess and exposes a simple show() / close() API."""

    def __init__(self):
        self._queue: Queue = Queue()
        self._proc = Process(target=_rgb_render_worker, args=(self._queue,))
        self._proc.start()

    def show(self, img: np.ndarray) -> None:
        """Send a new RGB image to the display window (replaces the previous one)."""
        # Drain any queued-but-not-yet-rendered frames
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._queue.put(img)

    def close(self) -> None:
        if self._proc.is_alive():
            self._queue.put(None)   # sentinel
            self._proc.join(timeout=3)
            if self._proc.is_alive():
                self._proc.terminate()

    def __del__(self):
        self.close()


# ─────────────────────────────────────────────────────────────────────────────
# Public preference oracle
# ─────────────────────────────────────────────────────────────────────────────


class HumanOracle(BaseOracle):
    """
    Preference oracle for human annotation.

    Displays a side-by-side matplotlib visualization of each segment's
    observation time series in a pyglet window, then reads the annotator's
    choice from the terminal.

    Keyboard shortcuts:
        L  — left segment is better
        R  — right segment is better
        E  — equal / no preference
        S  — skip this pair

    Args:
        frame_w: width of each individual segment panel in pixels
        frame_h: height of each individual segment panel in pixels
    """

    def __init__(self, frame_w: int = 360, frame_h: int = 420):
        self._renderer = _RGBRenderer()
        self._frame_w = frame_w
        self._frame_h = frame_h

    # ------------------------------------------------------------------

    def label(self, seg1, seg2) -> Optional[Tuple[float, float]]:
        """
        Generate and display the comparison image, then prompt for input.

        Returns:
            (1.0, 0.0)  left segment preferred
            (0.0, 1.0)  right segment preferred
            (0.5, 0.5)  equal
            None        skip
        """
        img = _make_comparison_image(
            seg1.frames,
            seg2.frames,
            self._frame_w,
            self._frame_h,
        )
        self._renderer.show(img)

        print("\n" + "─" * 58)
        print("  Segment comparison displayed in window.")
        print(
            f"  L — steps={len(seg1.frames):3d}"
            f"  R — steps={len(seg2.frames):3d}"
        )
        print("─" * 58)

        while True:
            try:
                raw = input("  [L] left  [R] right  [E] equal  [S] skip > ")
            except EOFError:
                return None

            choice = raw.strip().upper()

            if choice == "L":
                return (1.0, 0.0)
            elif choice == "R":
                return (0.0, 1.0)
            elif choice == "E":
                return (0.5, 0.5)
            elif choice in ("S", "SKIP", ""):
                return None
            else:
                print(f"  Invalid input '{raw}'. Use L, R, E, or S.")
