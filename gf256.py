import random

AES_POLY = 0x11B


def pmod(a, b):
    db = b.bit_length()
    while a.bit_length() >= db:
        a ^= b << (a.bit_length() - db)
    return a


def irreducible_polys():
    return [p for p in range(0x100, 0x200) if all(pmod(p, q) for q in range(2, 32))]


def gmul(a, b, poly=AES_POLY):
    r = 0
    while b:
        if b & 1:
            r ^= a
        a <<= 1
        if a & 0x100:
            a ^= poly
        b >>= 1
    return r


def ginv_table(poly=AES_POLY):
    inv = [0] * 256
    for x in range(1, 256):
        r, base, e = 1, x, 254
        while e:
            if e & 1:
                r = gmul(r, base, poly)
            base = gmul(base, base, poly)
            e >>= 1
        inv[x] = r
    return inv


def parity(x):
    return bin(x).count("1") & 1


def apply_matrix(rows, x):
    return sum(parity(rows[i] & x) << i for i in range(8))


def matrix_rank(rows):
    rows = list(rows)
    rank = 0
    for bit in range(8):
        piv = next((i for i in range(rank, 8) if rows[i] >> bit & 1), None)
        if piv is None:
            continue
        rows[rank], rows[piv] = rows[piv], rows[rank]
        for i in range(8):
            if i != rank and rows[i] >> bit & 1:
                rows[i] ^= rows[rank]
        rank += 1
    return rank


def random_invertible_matrix(rng):
    while True:
        rows = [rng.randrange(256) for _ in range(8)]
        if matrix_rank(rows) == 8:
            return rows


AES_AFFINE_ROWS = [
    sum(1 << ((i + k) % 8) for k in (0, 4, 5, 6, 7)) for i in range(8)
]
AES_AFFINE_CONST = 0x63


def sbox_from_inverse(poly, out_rows, out_const, in_rows=None, in_const=0):
    inv = ginv_table(poly)
    s = []
    for x in range(256):
        if in_rows is not None:
            x = apply_matrix(in_rows, x) ^ in_const
        s.append(apply_matrix(out_rows, inv[x]) ^ out_const)
    return s


def aes_sbox():
    return sbox_from_inverse(AES_POLY, AES_AFFINE_ROWS, AES_AFFINE_CONST)


def random_inverse_sbox(rng, polys):
    poly = rng.choice(polys)
    return sbox_from_inverse(
        poly,
        random_invertible_matrix(rng), rng.randrange(256),
        random_invertible_matrix(rng), rng.randrange(256),
    )


if __name__ == "__main__":
    s = aes_sbox()
    assert s[0] == 0x63 and s[1] == 0x7C and s[0x53] == 0xED
    print(len(irreducible_polys()), "irreducible polynomials; AES S-box generated OK")
