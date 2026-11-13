"""Small demo that needs no DFT code: fit the valley-resolved surrogate on stored PBE points for
strained Si, test it on held-out points, then solve one inverse-design problem on the surrogate
(the least-elastic-energy strain that makes Si a direct-gap semiconductor, cf. task T2).

usage: python demo/run_demo.py [--quick]      (--quick: 2-member ensemble, about 3x faster)

The optimum printed here is a SURROGATE prediction from 255 training points. In the paper every
candidate is then re-computed with DFT and checked by an independent verifier; that step needs
Quantum ESPRESSO and is not part of this demo.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eseagent.optimize import optimize_strain  # noqa: E402
from eseagent.surrogate import ValleySurrogate  # noqa: E402

torch.set_num_threads(2)
quick = "--quick" in sys.argv
recs = json.load(open(ROOT / "data" / "refsets" / "Si.json"))
idx = np.random.default_rng(0).permutation(len(recs))
ntr = int(0.8 * len(recs))
train, test = [recs[i] for i in idx[:ntr]], [recs[i] for i in idx[ntr:]]

t0 = time.time()
model = ValleySurrogate(n_models=2 if quick else 5).fit(train)
p = model.predict(np.array([r["strain_gl"] for r in test]))
out = dict(
    n_train=len(train), n_test=len(test), fit_seconds=round(time.time() - t0),
    gap_mae_mev=round(float(np.abs(p["eg"] - [r["eg_ev"] for r in test]).mean() * 1000), 1),
    h_mae_mev_a3=round(float(np.abs(p["h"] - [r["h_mev_a3"] for r in test]).mean()), 3),
    cbm_valley_accuracy=round(float(np.mean(p["cbm"] == np.array([r["cbm_valley"] for r in test]))), 3))
print("held-out test:", json.dumps(out), flush=True)

# Inverse design: minimise h subject to "Gamma is the conduction-band minimum by >= 20 meV".
sols = optimize_strain(model, [dict(kind="direct", value=0.02)], bounds=(-0.10, 0.10), n_starts=16)
best = sols[0] if sols else None
if best:
    best = {k: ([round(x, 4) for x in v] if isinstance(v, list) else round(v, 4) if isinstance(v, float) else v)
            for k, v in best.items()}
out["direct_gap_si_surrogate_optimum"] = best
print("lowest-h direct-gap Si (surrogate, not DFT-verified):", json.dumps(best))
(ROOT / "demo" / "output.json").write_text(json.dumps(out, indent=1))
