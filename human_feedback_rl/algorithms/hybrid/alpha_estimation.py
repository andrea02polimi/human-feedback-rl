"""Stima di alpha dalla varianza di campionamento dei due canali.

Il peso ``alpha`` sulle dimostrazioni deve dire quale dei due canali produce il
gradiente meno affidabile, dove "affidabile" significa: se rifacessi girare
l'algoritmo pescando altri campioni di feedback, quanto cambierebbe il gradiente
che ottengo?

Si parte dal gradiente indotto dal **singolo campione**, se ne misura la
dispersione attorno al gradiente medio, e solo alla fine si divide per la
dimensione del minibatch che l'ottimizzatore usa davvero.

Le due quantita' hanno ruoli distinti e non vanno confuse:

``N``
    numero di campioni disponibili nel canale. Serve a stimare quanto disperde il
    processo che genera i dati, e va usato tutto: piu' campioni, stima migliore.
``B``
    dimensione del minibatch, ``min(batch_size, N)``. Determina quanto rumore ha
    il gradiente effettivamente applicato, perche' la varianza di una media di
    ``B`` estrazioni scala come ``1/B``.

Ne segue che l'asimmetria fra i due canali (256 preferenze contro 64
dimostrazioni) **resta**, e deve restare: il gradiente delle preferenze e'
davvero mediato su quattro volte piu' campioni, quindi e' davvero meno rumoroso.

Sanity check: a budget piccolo ``N`` e ``B`` sono piccoli e ``S`` risulta grande;
a budget grande ``S`` cala. Si legge da ``alpha/S_pref`` e ``alpha/S_demo``.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch as th

from human_feedback_rl.common.batching import (
    fragment_avg_rewards,
    fragment_sum_rewards,
)
from human_feedback_rl.common.preference_losses import (
    bradley_terry_probs,
    preference_nll_per_sample,
)


@dataclass(frozen=True)
class ChannelDispersion:
    """Dispersione di un canale, dal singolo campione al minibatch."""

    n: int                  # campioni disponibili, usati per stimare V
    batch: int              # dimensione del minibatch, il divisore di S
    process_var: float      # V: varianza del processo generativo
    mean_var: float         # S = V / batch: varianza della media campionaria
    mean_norm_sq: float     # ||g_medio||^2, il denominatore che rende CV^2 adimensionale
    cv2: float              # S / ||g_medio||^2


@dataclass(frozen=True)
class AlphaEstimate:
    alpha: float
    pref: Optional[ChannelDispersion]
    demo: Optional[ChannelDispersion]
    pinned: bool            # True quando alpha e' il fallback, non una stima


def _flat_grads(make_scalar, items: Sequence, params: List[th.Tensor]) -> th.Tensor:
    """Gradienti per-campione appiattiti, uno per riga.

    Il grafo viene costruito e liberato UN CAMPIONE PER VOLTA. Fare invece un
    forward unico su tutti i campioni e poi ritagliarne le uscite costerebbe
    quadratico: ogni backward ripercorrerebbe l'intero grafo condiviso. Cosi'
    il costo totale e' proporzionale al numero di transizioni, come un normale
    forward-backward sull'intero dataset.
    """
    rows = []
    for item in items:
        grads = th.autograd.grad(make_scalar(item), params, allow_unused=True)
        rows.append(
            th.cat([
                (th.zeros_like(p) if g is None else g).reshape(-1)
                for p, g in zip(params, grads)
            ])
        )
    return th.stack(rows)


def _dispersion(per_sample: th.Tensor, batch: int, eps: float) -> Optional[ChannelDispersion]:
    """Applica la ricetta: media, distanze al quadrato, /(N-1), poi /B."""
    n = per_sample.shape[0]
    if n < 2 or batch <= 0:
        return None
    mean = per_sample.mean(dim=0)
    # sum_i ||g_i - gbar||^2, calcolata senza materializzare le differenze
    total_sq = float((per_sample - mean).pow(2).sum())
    process_var = total_sq / (n - 1)
    mean_var = process_var / batch
    mean_norm_sq = float(mean.pow(2).sum())
    if not np.isfinite(process_var) or not np.isfinite(mean_norm_sq):
        return None
    return ChannelDispersion(
        n=n,
        batch=batch,
        process_var=process_var,
        mean_var=mean_var,
        mean_norm_sq=mean_norm_sq,
        cv2=mean_var / max(mean_norm_sq, eps),
    )


def preference_sample_gradients(member, batch, smooth_labels, params) -> th.Tensor:
    """Un gradiente per confronto.

    La loss delle preferenze e' una media su confronti indipendenti, quindi la
    decomposizione e' esatta: ``mean_i g_i`` e' esattamente il gradiente
    full-batch. Ogni ``l_i`` dipende da theta solo attraverso ``Delta_i``, cioe'
    la differenza dei punteggi dei due frammenti, quindi

        g_i = (d l_i / d Delta_i) * grad_theta Delta_i

    Il coefficiente non e' derivato a mano ma preso con autograd sulla stessa
    ``preference_nll_per_sample`` usata in training: resta esatto anche col
    ``clamp`` interno, e non si disallinea se un domani la loss cambia.
    """
    pairs = batch.fragment_pairs

    # Valori dei Delta: un solo forward, senza grafo, servono solo ai coefficienti.
    with th.no_grad():
        delta = (
            fragment_avg_rewards(member, [p.frag1 for p in pairs])
            - fragment_avg_rewards(member, [p.frag2 for p in pairs])
        )

    # Coefficienti: derivata della loss rispetto ai soli Delta, senza theta.
    delta_leaf = delta.clone().requires_grad_(True)
    probs = bradley_terry_probs(delta_leaf, th.zeros_like(delta_leaf))
    losses = preference_nll_per_sample(probs, smooth_labels)
    (coeff,) = th.autograd.grad(losses.sum(), delta_leaf)

    def delta_of(pair):
        return (
            fragment_avg_rewards(member, [pair.frag1])[0]
            - fragment_avg_rewards(member, [pair.frag2])[0]
        )

    jac = _flat_grads(delta_of, pairs, params)      # grad_theta Delta_i, per riga
    return coeff.detach().unsqueeze(1) * jac


def demonstration_sample_gradients(
    member, expert_trajs, model_trajs, params
) -> th.Tensor:
    """Un gradiente per dimostrazione, sotto ``demo_2``.

    ``demo_2`` non si decompone: il termine di partizione contiene anche gli
    esperti. Il campione i-esimo e' definito come la loss che si vedrebbe se il
    batch fosse quella sola dimostrazione piu' tutto il rollout congelato,

        L_i = -R_i^E + logsumexp({R_j^M} u {R_i^E}) - log(|M| + 1)

    che e' la piu' vicina alla loss vera: la media dei ``g_i`` coincide col
    gradiente full-batch a meno della non-linearita' del logsumexp.

    Il rollout NON e' feedback, quindi e' tenuto fisso: la sua variabilita' non
    deve entrare nella varianza del canale dimostrazioni. Entra invece nei
    gradienti, perche' e' cio' che la loss usa davvero.
    """
    with th.no_grad():
        r_e = fragment_sum_rewards(member, expert_trajs)
        r_m = fragment_sum_rewards(member, model_trajs)

    def return_of(traj):
        return fragment_sum_rewards(member, [traj])[0]

    jac_expert = _flat_grads(return_of, expert_trajs, params)   # (N_d, P)
    jac_model = _flat_grads(return_of, model_trajs, params)     # (|M|, P)

    n_d, n_m = r_e.shape[0], r_m.shape[0]

    # riga i = softmax su [R^M..., R_i^E]; l'esperto e' l'ultimo elemento
    logits = th.cat([r_m.unsqueeze(0).expand(n_d, n_m), r_e.unsqueeze(1)], dim=1)
    weights = th.softmax(logits, dim=1)

    return (weights[:, -1:] - 1.0) * jac_expert + weights[:, :n_m] @ jac_model


def estimate_alpha(
    member,
    params: List[th.Tensor],
    pref_batch,
    smooth_labels,
    expert_trajs,
    model_trajs,
    batch_size_pref: int,
    batch_size_expert: int,
    min_prefs: int,
    eps: float,
) -> AlphaEstimate:
    """Peso sulle dimostrazioni, stimato ai parametri correnti.

    Chiamata PRIMA dei passi di gradiente dell'iterazione, non dopo: il peso
    deve descrivere il punto in cui verra' applicato.
    """
    # Confronti RACCOLTI, non necessariamente distinti: il fragmenter estrae
    # con rimpiazzo e nulla vieta che una coppia si ripeta, o che un frammento
    # venga confrontato con se stesso. Con L=1 e un pool di ~20.000 transizioni
    # la probabilita' e' ~1/20.000 per coppia, quindi in pratica non accade;
    # ma la soglia conta elementi, e il nome non deve promettere altro.
    n_pref = 0 if pref_batch is None else len(pref_batch.fragment_pairs)
    if n_pref < min_prefs:
        # Sotto pochissimi confronti la dispersione delle preferenze non e'
        # stimabile e la stima sarebbe distorta verso il basso, cioe' verso il
        # canale meno affidabile. Tutto il peso alle dimostrazioni.
        return AlphaEstimate(alpha=1.0, pref=None, demo=None, pinned=True)

    pref_grads = preference_sample_gradients(member, pref_batch, smooth_labels, params)
    demo_grads = demonstration_sample_gradients(member, expert_trajs, model_trajs, params)

    pref = _dispersion(pref_grads, min(batch_size_pref, pref_grads.shape[0]), eps)
    demo = _dispersion(demo_grads, min(batch_size_expert, demo_grads.shape[0]), eps)
    if pref is None or demo is None:
        return AlphaEstimate(alpha=1.0, pref=pref, demo=demo, pinned=True)

    total = pref.cv2 + demo.cv2
    if not np.isfinite(total) or total <= eps:
        return AlphaEstimate(alpha=1.0, pref=pref, demo=demo, pinned=True)

    alpha = float(np.clip(pref.cv2 / total, 0.0, 1.0))
    return AlphaEstimate(alpha=alpha, pref=pref, demo=demo, pinned=False)
