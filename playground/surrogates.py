"""Training data, training and loading of the playground surrogates.

    python playground/surrogates.py            # retrain both materials (about 4 + 15 min on a laptop)

The training data are the stored PBE points of the paper (playground/data/<material>.npz: strain, the
eight valley energies above the VBM, and h). 5% of the points are held out and the held-out errors are
written next to the weights. Weights are saved as plain tensors (playground/models/<material>.pt), so
loading them never unpickles code.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from eseagent.surrogate import MLP, STAB, VALLEYS, ValleySurrogate, _Head  # noqa: E402

N_MODELS = 3


def records(mat: str) -> list[dict]:
    d = np.load(HERE / "data" / f"{mat}.npz")
    assert list(d["valleys"]) == VALLEYS
    return [dict(strain_gl=x.astype(float).tolist(), valleys={v: {"e": float(e)} for v, e in zip(VALLEYS, ve)},
                 h_mev_a3=float(h)) for x, ve, h in zip(d["strain_gl"], d["valley_e"], d["h"])]


def train(mat: str, seed: int = 0) -> dict:
    recs = records(mat)
    zero = [r for r in recs if not any(r["strain_gl"])]              # always train on the unstrained point
    idx = np.random.default_rng(seed).permutation(len(recs))
    n_test = len(recs) // 20
    test = [recs[i] for i in idx[:n_test] if any(recs[i]["strain_gl"])]
    train_ = [recs[i] for i in idx[n_test:]] + zero
    t0 = time.time()
    m = ValleySurrogate(n_models=N_MODELS).fit(train_)
    e = np.array([r["strain_gl"] for r in test])
    p = m.predict(e)
    eg = np.array([min(r["valleys"][v]["e"] for v in VALLEYS) for r in test])
    z = m.predict(np.zeros((1, 6)))
    stats = dict(material=mat, n_train=len(train_), n_test=len(test), fit_minutes=round((time.time() - t0) / 60, 1),
                 test_gap_mae_mev=round(float(np.abs(p["eg"] - eg).mean() * 1000), 1),
                 test_h_mae_mev_a3=round(float(np.abs(p["h"] - [r["h_mev_a3"] for r in test]).mean()), 3),
                 test_h_mean_mev_a3=round(float(np.mean([r["h_mev_a3"] for r in test])), 2),
                 unstrained_gap_pred_ev=round(float(z["eg"][0]), 3), unstrained_gap_dft_ev=round(min(
                     zero[0]["valleys"][v]["e"] for v in VALLEYS), 3) if zero else None)
    save(m, mat, stats)
    return stats


def save(m: ValleySurrogate, mat: str, stats: dict) -> None:
    (HERE / "models").mkdir(exist_ok=True)
    blob = {k: dict(mu=h.mu, sd=h.sd, states=[mm.state_dict() for mm in h.models]) for k, h in m.heads.items()}
    torch.save(dict(n_models=m.n_models, heads=blob), HERE / "models" / f"{mat}.pt")
    (HERE / "models" / f"{mat}.json").write_text(json.dumps(stats, indent=1))


def load(mat: str) -> ValleySurrogate:
    blob = torch.load(HERE / "models" / f"{mat}.pt", weights_only=True)
    m = ValleySurrogate(n_models=blob["n_models"])
    m.heads = {}
    for k, h in blob["heads"].items():
        head = _Head(STAB[k], blob["n_models"])
        head.mu, head.sd = h["mu"], h["sd"]
        head.models = []
        for st in h["states"]:
            net = MLP()
            net.load_state_dict(st)
            net.eval()
            head.models.append(net)
        m.heads[k] = head
    return m


if __name__ == "__main__":
    for mat in sys.argv[1:] or ["Si", "C"]:
        print(json.dumps(train(mat)), flush=True)
