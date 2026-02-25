"""
Esempio eseguibile: Human Feedback via GUI (Tkinter)

Scenario: guida autonoma semplificata.
- Un agente percorre 5 step con stato (posizione_laterale, velocità)
- Per ogni step, l'umano può:
    1. Assegnare una reward scalare (StepRewardExpert)
    2. Correggere lo sterzo (StepCorrectionExpert)
    3. Preferire uno tra due step proposti (StepPreferenceExpert)
"""

import tkinter as tk
import random

from src.Core import Step, History
from src.Expert import (
    StepRewardExpert,
    StepCorrectionExpert,
    StepPreferenceExpert,
)


# ──────────────────────────────────────────────
# ENVIRONMENT (stub minimale)
# ──────────────────────────────────────────────

class DrivingEnv:
    """Ambiente di guida simulato (stub)."""
    lane_width = 3.5   # metri
    speed_limit = 30   # km/h


# ──────────────────────────────────────────────
# FUNZIONI DI FEEDBACK UMANO (con GUI)
# ──────────────────────────────────────────────

def human_reward_fn(step: Step, env: DrivingEnv, history: History) -> float:
    """
    Apre una finestra Tkinter e chiede all'umano di assegnare
    una reward da -1.0 a +1.0 per lo step osservato.
    """
    result = {}

    root = tk.Tk()
    root.title("Feedback: Reward")
    root.resizable(False, False)

    # Info sullo step
    frame_info = tk.LabelFrame(root, text="Step osservato", padx=10, pady=8)
    frame_info.pack(padx=15, pady=10, fill="x")

    lat, spd = step.state
    steer = step.action

    tk.Label(frame_info, text=f"Posizione laterale:  {lat:+.2f} m  "
                               f"({'sinistra' if lat < 0 else 'destra'} della corsia)",
             font=("Helvetica", 11)).pack(anchor="w")
    tk.Label(frame_info, text=f"Velocità:            {spd:.1f} km/h",
             font=("Helvetica", 11)).pack(anchor="w")
    tk.Label(frame_info, text=f"Sterzo agente:       {steer:+.2f}",
             font=("Helvetica", 11)).pack(anchor="w")

    if history:
        tk.Label(frame_info, text=f"Step valutati fin ora: {len(history)}",
                 font=("Helvetica", 9), fg="gray").pack(anchor="w", pady=(5, 0))

    # Slider reward
    frame_reward = tk.LabelFrame(root, text="Assegna una reward", padx=10, pady=8)
    frame_reward.pack(padx=15, pady=5, fill="x")

    tk.Label(frame_reward,
             text="−1.0 = pessimo    0.0 = neutro    +1.0 = ottimo",
             font=("Helvetica", 9), fg="gray").pack()

    slider = tk.Scale(frame_reward, from_=-1.0, to=1.0, resolution=0.1,
                      orient=tk.HORIZONTAL, length=300, font=("Helvetica", 11))
    slider.set(0.0)
    slider.pack()

    def confirm():
        result["value"] = float(slider.get())
        root.destroy()

    tk.Button(root, text="Conferma", font=("Helvetica", 12, "bold"),
              bg="#4CAF50", fg="black", command=confirm).pack(pady=10)

    root.mainloop()
    return result.get("value", 0.0)


def human_correction_fn(step: Step, env: DrivingEnv, history: History):
    """
    Mostra lo sterzo corrente dell'agente e permette all'umano
    di correggerlo con uno slider.
    Restituisce il valore di sterzo corretto.
    """
    result = {}

    root = tk.Tk()
    root.title("🔧 Feedback: Correzione")
    root.resizable(False, False)

    lat, spd = step.state
    steer = step.action

    frame_info = tk.LabelFrame(root, text="Step da correggere", padx=10, pady=8)
    frame_info.pack(padx=15, pady=10, fill="x")

    tk.Label(frame_info, text=f"Posizione laterale: {lat:+.2f} m",
             font=("Helvetica", 11)).pack(anchor="w")
    tk.Label(frame_info, text=f"Sterzo attuale agente: {steer:+.2f}",
             font=("Helvetica", 11), fg="red").pack(anchor="w")

    frame_corr = tk.LabelFrame(root, text="Sterzo corretto", padx=10, pady=8)
    frame_corr.pack(padx=15, pady=5, fill="x")

    slider = tk.Scale(frame_corr, from_=-1.0, to=1.0, resolution=0.05,
                      orient=tk.HORIZONTAL, length=300, font=("Helvetica", 11))
    slider.set(round(steer, 2))   # parte dal valore attuale
    slider.pack()

    def confirm():
        result["action"] = float(slider.get())
        root.destroy()

    tk.Button(root, text="Correggi", font=("Helvetica", 12, "bold"),
              bg="#2196F3", fg="black", command=confirm).pack(pady=10)

    root.mainloop()
    return result.get("action", steer)


def human_preference_fn(steps: list, env: DrivingEnv, history: History) -> int:
    """
    Mostra due step proposti e chiede all'umano quale preferisce.
    Restituisce l'indice (0 o 1) dello step preferito.
    """
    result = {}

    root = tk.Tk()
    root.title("⚖️  Feedback: Preferenza")
    root.resizable(False, False)

    tk.Label(root, text="Quale step preferisci?",
             font=("Helvetica", 13, "bold")).pack(pady=10)

    container = tk.Frame(root)
    container.pack(padx=15, pady=5)

    for i, step in enumerate(steps):
        lat, spd = step.state
        steer = step.action

        frame = tk.LabelFrame(container, text=f"  Opzione {i + 1}  ",
                              padx=12, pady=10, font=("Helvetica", 11, "bold"))
        frame.grid(row=0, column=i, padx=15)

        tk.Label(frame, text=f"Posizione: {lat:+.2f} m",
                 font=("Helvetica", 11)).pack(anchor="w")
        tk.Label(frame, text=f"Velocità:  {spd:.1f} km/h",
                 font=("Helvetica", 11)).pack(anchor="w")
        tk.Label(frame, text=f"Sterzo:    {steer:+.2f}",
                 font=("Helvetica", 11)).pack(anchor="w")

        def choose(idx=i):
            result["preferred"] = idx
            root.destroy()

        tk.Button(frame, text=f"Scelgo {i + 1}", font=("Helvetica", 11, "bold"),
                  bg="#FF9800", fg="black", command=choose, width=12).pack(pady=(10, 0))

    root.mainloop()
    return result.get("preferred", 0)


# ──────────────────────────────────────────────
# AGENTE (genera step casuali)
# ──────────────────────────────────────────────

def generate_step() -> Step:
    """Simula uno step dell'agente con piccoli errori."""
    lateral_pos = round(random.uniform(-0.8, 0.8), 2)   # errore laterale
    speed       = round(random.uniform(25.0, 35.0), 1)  # km/h
    steering    = round(random.uniform(-0.6, 0.6), 2)   # azione di sterzo
    return Step(state=(lateral_pos, speed), action=steering)


# ──────────────────────────────────────────────
# DEMO PRINCIPALE
# ──────────────────────────────────────────────

def run_demo():
    env = DrivingEnv()

    # Crea i tre expert con le funzioni GUI
    reward_expert     = StepRewardExpert(env, human_reward_fn)
    correction_expert = StepCorrectionExpert(env, human_correction_fn)
    preference_expert = StepPreferenceExpert(env, human_preference_fn)

    print("\n" + "═" * 50)
    print("Demo feedback umano via GUI")
    print("═" * 50)
    print("Verranno mostrate 3 finestre, una per tipo di feedback.")
    print("Chiudi ogni finestra dopo aver interagito.\n")

    collected = []

    # 1. Reward
    print("[1/3] Apertura GUI per REWARD...")
    step_r = generate_step()
    fb_reward = reward_expert.query(step_r)
    collected.append(("Reward", step_r, fb_reward))
    print(f"       Ricevuto: {fb_reward}")

    # 2. Correzione
    print("[2/3] Apertura GUI per CORREZIONE...")
    step_c = generate_step()
    fb_correction = correction_expert.query(step_c)
    collected.append(("Correction", step_c, fb_correction))
    print(f"       Ricevuto: {fb_correction}")

    # 3. Preferenza tra due step
    print("[3/3] Apertura GUI per PREFERENZA...")
    step_a = generate_step()
    step_b = generate_step()
    fb_preference = preference_expert.query([step_a, step_b])
    collected.append(("Preference", [step_a, step_b], fb_preference))
    print(f"       Ricevuto: {fb_preference}")

    # Riepilogo
    print("\n" + "─" * 50)
    print("RIEPILOGO FEEDBACK RACCOLTI")
    print("─" * 50)
    for fb_type, obj, fb in collected:
        print(f"\n  [{fb_type}]")
        if isinstance(obj, list):
            for i, s in enumerate(obj):
                print(f"    Step {i+1}: stato={s.state}  azione={s.action}")
        else:
            print(f"    Step: stato={obj.state}  azione={obj.action}")
        print(f"    Feedback: {fb}")

    print("\n  History expert reward:     ", reward_expert.history)
    print("  History expert correction: ", correction_expert.history)
    print("  History expert preference: ", preference_expert.history)
    print("\n Demo completata.\n")


if __name__ == "__main__":
    run_demo()