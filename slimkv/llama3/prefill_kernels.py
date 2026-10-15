"""Prefill rotary embedding with unchanged BF16 operation boundaries."""
import triton
import triton.language as tl


@triton.jit
def prefill_rope_kernel(Q, K, COS, SIN, POS, ROLES, OQ, OK,
                        BATCH, HQ: tl.constexpr, HK: tl.constexpr,
                        QLEN, KLEN, DIM: tl.constexpr,
                        QB, QH, QQ, KB, KH, KK,
                        OQB, OQH, OQQ, OKB, OKH, OKK,
                        PB, ROLE_LEN, SINK: tl.constexpr,
                        BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    qblocks = tl.cdiv(BATCH * HQ * QLEN * DIM, BLOCK)
    if pid < qblocks:
        offset = pid * BLOCK + tl.arange(0, BLOCK)
        valid = offset < BATCH * HQ * QLEN * DIM
        d = offset % DIM
        s = offset // DIM % QLEN
        h = offset // (DIM * QLEN) % HQ
        b = offset // (DIM * QLEN * HQ)
        base = b * QB + h * QH + s * QQ
        output = b * OQB + h * OQH + s * OQQ + d
        position = tl.load(POS + b * PB + KLEN - QLEN + s, valid, 0)
        x = tl.load(Q + base + d, valid, 0)
        partner = tl.load(Q + base + (d + DIM // 2) % DIM, valid, 0)
        beacon = tl.full((BLOCK,), False, tl.int1)
        target = OQ
    else:
        offset = (pid - qblocks) * BLOCK + tl.arange(0, BLOCK)
        valid = offset < BATCH * HK * KLEN * DIM
        d = offset % DIM
        s = offset // DIM % KLEN
        h = offset // (DIM * KLEN) % HK
        b = offset // (DIM * KLEN * HK)
        base = b * KB + h * KH + s * KK
        output = b * OKB + h * OKH + s * OKK + d
        position = tl.load(POS + b * PB + s, valid, 0)
        prefix = KLEN - ROLE_LEN
        role = tl.load(ROLES + s - prefix, valid & (s >= prefix), 0)
        beacon = ((s < prefix) & (s >= SINK)) | (role == 1) | (role == 2)
        x = tl.load(K + base + d, valid, 0)
        partner = tl.load(K + base + (d + DIM // 2) % DIM, valid & ~beacon, 0)
        target = OK
    dtype = x.dtype
    cosine = tl.load(COS + position * DIM + d, valid & ~beacon, 0).to(dtype).to(tl.float32)
    sine = tl.load(SIN + position * DIM + d, valid & ~beacon, 0).to(dtype).to(tl.float32)
    half = tl.where(d < DIM // 2, -partner.to(tl.float32), partner.to(tl.float32))
    first = (x.to(tl.float32) * cosine).to(dtype).to(tl.float32)
    second = (half * sine).to(dtype).to(tl.float32)
    rotated = (first + second).to(dtype)
    tl.store(target + output, tl.where(beacon, x, rotated), valid)
