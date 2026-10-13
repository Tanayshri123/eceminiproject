import numpy as np

X = np.arange(256)
POP = np.array([bin(i).count("1") for i in range(256)])
H = (1 - 2 * (POP[X[:, None] & X[None, :]] & 1)).astype(np.float32)
XS = np.array([[x ^ s for x in range(256)] for s in range(1, 256)])
MASK_BITS = np.array([[(m >> j) & 1 for j in range(8)] for m in range(1, 256)], dtype=np.float32)


def evaluate(sboxes):
    S = np.asarray(sboxes, dtype=np.int64)
    if S.ndim == 1:
        S = S[None]
    B = len(S)
    out = {}

    out["bijective"] = np.array([len(set(r)) == 256 for r in S.tolist()])

    F = H[S]
    W = np.matmul(H, F)
    absW = np.abs(W[:, :, 1:])
    out["nl"] = (128 - absW.max(axis=(1, 2)) / 2).astype(int)

    AC = np.matmul(H, W * W) / 256.0
    out["autocorr"] = np.abs(AC[:, 1:, 1:]).max(axis=(1, 2)).astype(int)

    D = S[:, XS] ^ S[:, None, :]
    idx = (np.arange(B)[:, None, None] * 65280 + np.arange(255)[None, :, None] * 256 + D).ravel()
    ddt = np.bincount(idx, minlength=B * 65280).reshape(B, 255, 256)
    out["du"] = ddt.max(axis=(1, 2))

    bits = ((S[:, None, :] >> np.arange(8)[None, :, None]) & 1).astype(np.int8)
    a = bits.copy()
    for i in range(8):
        v = a.reshape(B, 8, 1 << (7 - i), 2, 1 << i)
        v[:, :, :, 1, :] ^= v[:, :, :, 0, :]
    comp = (np.matmul(MASK_BITS, a.astype(np.float32)).astype(np.int64) & 1)
    out["degree"] = (comp * POP[None, None, :]).max(axis=2).min(axis=1)

    flips = np.stack([S ^ S[:, X ^ (1 << i)] for i in range(8)], axis=1)
    ones = ((flips[:, :, None, :] >> np.arange(8)[None, None, :, None]) & 1).sum(axis=3)
    out["sac_dev"] = np.abs(ones - 128).sum(axis=(1, 2)) / (64 * 128)

    out["fixed"] = (S == X).sum(axis=1) + (S == (X ^ 255)).sum(axis=1)
    return out


def evaluate_in_chunks(sboxes, chunk=64):
    parts = [evaluate(sboxes[i:i + chunk]) for i in range(0, len(sboxes), chunk)]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def classify(m, base, sac_margin=0.05):
    n = len(m["nl"])
    worse = (~m["bijective"]) | (m["nl"] < base["nl"]) | (m["du"] > base["du"]) | (m["degree"] < base["degree"])
    prim_better = (m["nl"] > base["nl"]) | (m["du"] < base["du"]) | (m["degree"] > base["degree"])
    prim_equal = ~worse & ~prim_better
    sec_better = (m["fixed"] == 0) & (m["sac_dev"] < base["sac_dev"] * (1 - sac_margin)) & (m["autocorr"] <= base["autocorr"])
    better = ~worse & (prim_better | (prim_equal & sec_better))
    return np.where(worse, "worse", np.where(better, "better", "equivalent"))
