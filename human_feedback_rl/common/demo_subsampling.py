"""Selezione riproducibile del sottocampione di dimostrazioni.

Ogni braccio che consuma dimostrazioni — i baseline solo-dimostrazioni e
l'algoritmo ibrido — deve vedere le STESSE dimostrazioni allo stesso budget.
Altrimenti una differenza fra bracci potrebbe venire da quali traiettorie sono
capitate, non dal metodo.

Due proprieta' danno la garanzia:

* il seed della selezione e' una costante condivisa, **indipendente dal seed di
  training**: cambiare ``run.seed`` cambia l'inizializzazione della rete e il
  rollout, non l'insieme di dimostrazioni;
* si permuta l'INTERO dataset e poi se ne prende un prefisso, quindi i budget
  sono annidati: passare da 10 a 100 aggiunge 90 dimostrazioni senza scambiare
  le prime 10.

Le impronte servono a verificarlo a posteriori: due run allo stesso budget
devono scrivere lo stesso ``fingerprint``.
"""
from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Sequence

import numpy as np

# Costante condivisa: NON e' il seed di training.
DEMO_SUBSAMPLE_SEED = 1000


def select_demo_indices(
    n_available: int,
    lengths: Optional[Sequence[int]] = None,
    n_trajectories: Optional[int] = None,
    n_transitions: Optional[int] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Indici delle dimostrazioni da tenere, nell'ordine di selezione.

    ``n_trajectories`` fissa il budget in traiettorie intere; ``n_transitions``
    lo fissa in transizioni (si prendono traiettorie intere finche' la lunghezza
    cumulata sta nel cap, sempre almeno una). I due sono mutuamente esclusivi.
    Senza nessuno dei due si restituisce l'intero dataset.

    ``seed=None`` significa :data:`DEMO_SUBSAMPLE_SEED`, cosi' chi dimentica di
    passarlo ottiene comunque il sottocampione condiviso e non uno privato.
    """
    if n_trajectories is not None and n_transitions is not None:
        raise ValueError(
            "n_trajectories and n_transitions are mutually exclusive; "
            f"got {n_trajectories} and {n_transitions}."
        )
    if n_available < 1:
        raise ValueError(f"n_available must be >= 1, got {n_available}.")
    if n_trajectories is None and n_transitions is None:
        return np.arange(n_available)

    if seed is None:
        seed = DEMO_SUBSAMPLE_SEED
    # Permutare tutto il dataset (non solo il budget) e' cio' che rende i
    # budget annidati: ogni budget legge un prefisso dello stesso ordine.
    order = np.random.default_rng(seed).permutation(n_available)

    if n_transitions is not None:
        if lengths is None:
            raise ValueError("lengths is required when budgeting by n_transitions.")
        if len(lengths) != n_available:
            raise ValueError(
                f"lengths has {len(lengths)} entries but n_available is {n_available}."
            )
        if n_transitions < 1:
            raise ValueError(f"n_transitions must be >= 1, got {n_transitions}.")
        selected: List[int] = []
        total = 0
        for index in order:
            length = int(lengths[index])
            if selected and total + length > n_transitions:
                continue
            selected.append(int(index))
            total += length
            if total >= n_transitions:
                break
        return np.asarray(selected, dtype=int)

    if not 1 <= n_trajectories <= n_available:
        raise ValueError(
            f"n_trajectories must be in [1, {n_available}], got {n_trajectories}."
        )
    return order[:n_trajectories]


def indices_fingerprint(indices: Sequence[int]) -> str:
    """Hash dell'INSIEME selezionato, non del suo ordine.

    Due bracci allo stesso budget possono elencare le stesse dimostrazioni in
    ordine diverso: cio' che deve coincidere e' l'insieme.
    """
    payload = ",".join(str(int(i)) for i in sorted(indices))
    return hashlib.sha1(payload.encode()).hexdigest()


def dataset_fingerprint(lengths: Sequence[int]) -> str:
    """Hash della forma del dataset: quante traiettorie e lunghe quanto.

    Serve a distinguere "stesso budget, stesso dataset" da "stesso budget, ma
    qualcuno ha rigenerato le dimostrazioni".
    """
    payload = ",".join(str(int(x)) for x in lengths)
    return hashlib.sha1(payload.encode()).hexdigest()


def subsample_manifest(
    indices: Sequence[int],
    lengths: Sequence[int],
    seed: Optional[int],
    n_trajectories: Optional[int] = None,
    n_transitions: Optional[int] = None,
    dataset_name: str = "",
) -> Dict:
    """Descrive una selezione abbastanza da riprodurla e da confrontarla."""
    idx = [int(i) for i in indices]
    # I nomi delle chiavi sono un'interfaccia: scripts/train_hybrid_sac.py e
    # scripts/verify_demo_subsample.py leggono subsample_seed, fingerprint,
    # dataset_fingerprint e n_transitions_selected. Rinominarle rompe entrambi.
    return {
        "dataset_name": dataset_name,
        "dataset_size": len(lengths),
        "dataset_fingerprint": dataset_fingerprint(lengths),
        "subsample_seed": DEMO_SUBSAMPLE_SEED if seed is None else int(seed),
        "budget_n_trajectories": n_trajectories,
        "budget_n_transitions": n_transitions,
        "n_selected": len(idx),
        "n_transitions_selected": int(sum(int(lengths[i]) for i in idx)),
        "indices": idx,
        "fingerprint": indices_fingerprint(idx),
    }
