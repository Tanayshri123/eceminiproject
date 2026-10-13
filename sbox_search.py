import json
import random
import sys
import time
from collections import Counter

import numpy as np

from gf256 import aes_sbox, irreducible_polys, random_inverse_sbox
from sbox_metrics import classify, evaluate, evaluate_in_chunks

TOTAL = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
SEED = 2024

rng = random.Random(SEED)
polys = irreducible_polys()
AES = aes_sbox()
base = {k: v[0] for k, v in evaluate(AES).items()}


def family_affine_inverse():
    return random_inverse_sbox(rng, polys)


def family_perturbed_aes():
    s = list(AES)
    for _ in range(rng.randint(1, 3)):
        i, j = rng.sample(range(256), 2)
        s[i], s[j] = s[j], s[i]
    return s


def family_random_perm():
    s = list(range(256))
    rng.shuffle(s)
    return s


FAMILIES = [
    ("affine-equivalent inverse maps", family_affine_inverse, 0.70),
    ("AES with 1-3 swaps", family_perturbed_aes, 0.20),
    ("uniform random permutations", family_random_perm, 0.10),
]

rank_key = lambda m, i: (-m["nl"][i], m["du"][i], -m["degree"][i], m["fixed"][i], m["sac_dev"][i], m["autocorr"][i])

print(f"AES baseline: {base}")
summary = {}
best = None
best_key = None
t0 = time.time()

for name, gen, frac in FAMILIES:
    n = int(TOTAL * frac)
    cands = [gen() for _ in range(n)]
    m = evaluate_in_chunks(cands)
    verdict = classify(m, base)
    summary[name] = dict(Counter(verdict.tolist()))
    summary[name]["nl_hist"] = {int(k): int(v) for k, v in sorted(Counter(m["nl"].tolist()).items())}
    summary[name]["du_hist"] = {int(k): int(v) for k, v in sorted(Counter(m["du"].tolist()).items())}
    for i in np.flatnonzero(verdict != "worse"):
        if cands[i] == AES:
            continue
        k = rank_key(m, i)
        if best_key is None or k < best_key:
            best_key, best, best_metrics = k, cands[i], {a: float(b[i]) for a, b in m.items()}
    print(f"{name}: {n} candidates, {summary[name]['better'] if 'better' in summary[name] else 0} better, "
          f"{summary[name].get('equivalent', 0)} equivalent, {summary[name].get('worse', 0)} worse "
          f"({time.time() - t0:.0f}s)")

tot = Counter()
for v in summary.values():
    for k in ("better", "equivalent", "worse"):
        tot[k] += v.get(k, 0)
print(f"\nTOTAL over {sum(tot.values())} S-boxes: {dict(tot)}")
print(f"best S-box metrics: {best_metrics}")

json.dump({"total": dict(tot), "families": summary, "aes_baseline": {k: float(v) for k, v in base.items()},
           "best_metrics": best_metrics}, open("search_results.json", "w"), indent=1)
json.dump(best, open("best_sbox.json", "w"))
