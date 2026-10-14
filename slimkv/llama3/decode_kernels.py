"""CUDA kernels loaded lazily by decode_ops when Triton is available."""
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def rope_kernel(Q, K, COS, SIN, POS, RAW_IDX, OQ, OK,
                HQ: tl.constexpr, HK: tl.constexpr, QLEN: tl.constexpr,
                KLEN, DIM: tl.constexpr, BATCH: tl.constexpr,
                QB: tl.constexpr, QH: tl.constexpr, QQ: tl.constexpr,
                KB, KH, KK: tl.constexpr,
                PB, SINK: tl.constexpr, HISTORY,
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
        position = tl.load(POS + b * PB + s, valid, 0)
        x = tl.load(Q + base + d, valid, 0)
        partner = tl.load(Q + base + (d + DIM // 2) % DIM, valid, 0)
        target = OQ
    else:
        offset = (pid - qblocks) * BLOCK + tl.arange(0, BLOCK)
        valid = offset < BATCH * HK * KLEN * DIM
        d = offset % DIM
        s = offset // DIM % KLEN
        h = offset // (DIM * KLEN) % HK
        b = offset // (DIM * KLEN * HK)
        base = b * KB + h * KH + s * KK
        old_raw = tl.load(RAW_IDX + s - SINK, valid & (s >= SINK) & (s < KLEN - 1), 0)
        current = tl.load(POS + b * PB, valid, 0)
        position = tl.where(s < SINK, s,
                            tl.where(s == KLEN - 1, current, old_raw + SINK + HISTORY))
        x = tl.load(K + base + d, valid, 0)
        partner = tl.load(K + base + (d + DIM // 2) % DIM, valid, 0)
        target = OK
    dtype = x.dtype
    cosine = tl.load(COS + position * DIM + d, valid, 0).to(dtype).to(tl.float32)
    sine = tl.load(SIN + position * DIM + d, valid, 0).to(dtype).to(tl.float32)
    half = tl.where(d < DIM // 2, -partner.to(tl.float32), partner.to(tl.float32))
    # Preserve Torch's BF16 rounding after each multiply and after the add.
    first = (x.to(tl.float32) * cosine).to(dtype).to(tl.float32)
    second = (half * sine).to(dtype).to(tl.float32)
    tl.store(target + offset, first + second, valid)

@triton.jit
def merge_kernel(RAW, LATENT, RLSE, LLSE, OUT,
                 HEADS: tl.constexpr, LENGTH: tl.constexpr, DIM: tl.constexpr,
                 BATCH: tl.constexpr, RB: tl.constexpr, RH: tl.constexpr, RQ: tl.constexpr,
                 LB: tl.constexpr, LH: tl.constexpr, LQ: tl.constexpr,
                 RLB: tl.constexpr, RLH: tl.constexpr, RLQ: tl.constexpr,
                 LLB: tl.constexpr, LLH: tl.constexpr, LLQ: tl.constexpr,
                 BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offset < BATCH * HEADS * LENGTH * DIM
    d = offset % DIM
    q = offset // DIM % LENGTH
    h = offset // (DIM * LENGTH) % HEADS
    b = offset // (DIM * LENGTH * HEADS)
    r = tl.load(RLSE + b * RLB + h * RLH + q * RLQ, valid, 0)
    l = tl.load(LLSE + b * LLB + h * LLH + q * LLQ, valid, 0)
    m = tl.maximum(r, l)
    total = m + libdevice.log1p(libdevice.exp(-tl.abs(r - l)))
    rw = libdevice.exp(r - total)
    lw = libdevice.exp(l - total)
    rv = tl.load(RAW + b * RB + h * RH + q * RQ + d, valid, 0).to(tl.float32)
    lv = tl.load(LATENT + b * LB + h * LH + q * LQ + d, valid, 0).to(tl.float32)
    tl.store(OUT + offset, rw * rv + lw * lv, valid)

