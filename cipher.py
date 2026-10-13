import json
import os
import random
from itertools import combinations

from gf256 import aes_sbox, gmul

RCON = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36]
MUL = [[gmul(a, b) for b in range(256)] for a in range(256)]
AES_MIX_ROW = (2, 3, 1, 1)
AES_SHIFTS = (0, 1, 2, 3)


def circulant(first_row):
    return [[first_row[(c - r) % 4] for c in range(4)] for r in range(4)]


def det(m):
    if len(m) == 1:
        return m[0][0]
    d = 0
    for j, a in enumerate(m[0]):
        if a:
            minor = [row[:j] + row[j + 1:] for row in m[1:]]
            d ^= MUL[a][det(minor)]
    return d


def is_mds(m):
    for k in range(1, 5):
        for rs in combinations(range(4), k):
            for cs in combinations(range(4), k):
                if det([[m[r][c] for c in cs] for r in rs]) == 0:
                    return False
    return True


def mat_inverse(m):
    n = len(m)
    a = [list(m[i]) + [int(i == j) for j in range(n)] for i in range(n)]
    inv = [0] * 256
    for x in range(1, 256):
        inv[x] = next(y for y in range(1, 256) if MUL[x][y] == 1)
    for col in range(n):
        piv = next(r for r in range(col, n) if a[r][col])
        a[col], a[piv] = a[piv], a[col]
        iv = inv[a[col][col]]
        a[col] = [MUL[iv][v] for v in a[col]]
        for r in range(n):
            if r != col and a[r][col]:
                f = a[r][col]
                a[r] = [v ^ MUL[f][w] for v, w in zip(a[r], a[col])]
    return [row[n:] for row in a]


class BlockCipher:
    def __init__(self, key, sbox, shifts=AES_SHIFTS, mix_row=AES_MIX_ROW, rounds=10):
        assert len(key) == 16 and sorted(sbox) == list(range(256))
        self.sbox = list(sbox)
        self.inv_sbox = [0] * 256
        for i, v in enumerate(sbox):
            self.inv_sbox[v] = i
        self.shifts = shifts
        self.mix = circulant(mix_row)
        assert is_mds(self.mix)
        self.inv_mix = mat_inverse(self.mix)
        self.rounds = rounds
        self.round_keys = self._expand(list(key))

    def _expand(self, key):
        w = [key[4 * i:4 * i + 4] for i in range(4)]
        for i in range(4, 4 * (self.rounds + 1)):
            t = list(w[i - 1])
            if i % 4 == 0:
                t = [self.sbox[b] for b in t[1:] + t[:1]]
                t[0] ^= RCON[i // 4 - 1]
            w.append([a ^ b for a, b in zip(w[i - 4], t)])
        return [sum(w[4 * r:4 * r + 4], []) for r in range(self.rounds + 1)]

    @staticmethod
    def _xor(s, k):
        return [a ^ b for a, b in zip(s, k)]

    def _shift(self, s, inverse=False):
        out = [0] * 16
        for r in range(4):
            sh = -self.shifts[r] if inverse else self.shifts[r]
            for c in range(4):
                out[r + 4 * c] = s[r + 4 * ((c + sh) % 4)]
        return out

    @staticmethod
    def _mix(s, m):
        out = [0] * 16
        for c in range(4):
            col = s[4 * c:4 * c + 4]
            for r in range(4):
                v = 0
                for j in range(4):
                    v ^= MUL[m[r][j]][col[j]]
                out[r + 4 * c] = v
        return out

    def encrypt(self, block):
        s = self._xor(list(block), self.round_keys[0])
        for r in range(1, self.rounds + 1):
            s = [self.sbox[b] for b in s]
            s = self._shift(s)
            if r != self.rounds:
                s = self._mix(s, self.mix)
            s = self._xor(s, self.round_keys[r])
        return s

    def decrypt(self, block):
        s = list(block)
        for r in range(self.rounds, 0, -1):
            s = self._xor(s, self.round_keys[r])
            if r != self.rounds:
                s = self._mix(s, self.inv_mix)
            s = self._shift(s, inverse=True)
            s = [self.inv_sbox[b] for b in s]
        return self._xor(s, self.round_keys[0])


def find_mix_rows():
    found = []
    for row in combinations_with_replacement_range():
        if is_mds(circulant(row)):
            found.append(row)
    return found


def combinations_with_replacement_range():
    for a in range(1, 9):
        for b in range(1, 9):
            for c in range(1, 9):
                for d in range(1, 9):
                    yield (a, b, c, d)


PROJECT_SHIFTS = (0, 2, 3, 1)
PROJECT_MIX_ROW = (3, 1, 4, 2)


def project_cipher(key):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_sbox.json")
    return BlockCipher(key, json.load(open(path)), PROJECT_SHIFTS, PROJECT_MIX_ROW)


def avalanche(make, trials=300, seed=7):
    rng = random.Random(seed)
    total = 0
    for _ in range(trials):
        c = make([rng.randrange(256) for _ in range(16)])
        p = [rng.randrange(256) for _ in range(16)]
        q = list(p)
        q[rng.randrange(16)] ^= 1 << rng.randrange(8)
        total += sum(bin(a ^ b).count("1") for a, b in zip(c.encrypt(p), c.encrypt(q)))
    return total / trials


if __name__ == "__main__":
    aes = BlockCipher(bytes(range(16)), aes_sbox())
    pt = bytes.fromhex("00112233445566778899aabbccddeeff")
    assert bytes(aes.encrypt(pt)).hex() == "69c4e0d86a7b0430d8cdb78070b4c55a"
    print("AES-128 FIPS-197 test vector: OK (cipher framework verified)")

    print("MDS check  AES circ(2,3,1,1):", is_mds(circulant(AES_MIX_ROW)),
          "| project circ(3,1,4,2):", is_mds(circulant(PROJECT_MIX_ROW)))

    rng = random.Random(1)
    for t in range(1000):
        key = [rng.randrange(256) for _ in range(16)]
        p = [rng.randrange(256) for _ in range(16)]
        c = project_cipher(key)
        ct = c.encrypt(p)
        assert c.decrypt(ct) == p
    print("1000 random key/block round-trips: OK")

    key = [rng.randrange(256) for _ in range(16)]
    p = [rng.randrange(256) for _ in range(16)]
    c = project_cipher(key)
    ct = c.encrypt(p)
    print("key        ", bytes(key).hex())
    print("plaintext  ", bytes(p).hex())
    print("ciphertext ", bytes(ct).hex())
    print("decrypted  ", bytes(c.decrypt(ct)).hex())

    print("avalanche (ideal 64 of 128 bits)  AES: %.2f  project: %.2f" % (
        avalanche(lambda k: BlockCipher(k, aes_sbox())), avalanche(project_cipher)))
