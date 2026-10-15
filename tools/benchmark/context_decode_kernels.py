"""BF16-exact rotary fusion for the shared fixed-context benchmark runtime."""
import triton
import triton.language as tl


@triton.jit
def rotary_kernel(Q, K, COS, SIN, QPOS, KPOS, ROLES, OQ, OK,
                  HQ: tl.constexpr, HK: tl.constexpr, QLEN: tl.constexpr,
                  KLEN: tl.constexpr, DIM: tl.constexpr,
                  QH: tl.constexpr, QQ: tl.constexpr,
                  KH: tl.constexpr, KK: tl.constexpr,
                  OQH: tl.constexpr, OQQ: tl.constexpr,
                  OKH: tl.constexpr, OKK: tl.constexpr,
                  SKIP_BEACONS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    qblocks = tl.cdiv(HQ*QLEN*DIM, BLOCK)
    if pid < qblocks:
        offset = pid*BLOCK+tl.arange(0, BLOCK)
        valid = offset < HQ*QLEN*DIM
        d = offset % DIM
        s = offset//DIM % QLEN
        h = offset//(DIM*QLEN)
        base = h*QH+s*QQ
        target = OQ
        destination = h*OQH+s*OQQ+d
        position = tl.load(QPOS+s, valid, 0)
        x = tl.load(Q+base+d, valid, 0)
        partner = tl.load(Q+base+(d+DIM//2)%DIM, valid, 0)
        beacon = tl.full((BLOCK,), False, tl.int1)
    else:
        offset = (pid-qblocks)*BLOCK+tl.arange(0, BLOCK)
        valid = offset < HK*KLEN*DIM
        d = offset % DIM
        s = offset//DIM % KLEN
        h = offset//(DIM*KLEN)
        base = h*KH+s*KK
        target = OK
        destination = h*OKH+s*OKK+d
        position = tl.load(KPOS+s, valid, 0)
        x = tl.load(K+base+d, valid, 0)
        partner = tl.load(K+base+(d+DIM//2)%DIM, valid, 0)
        if SKIP_BEACONS:
            beacon = tl.load(ROLES+s, valid, False)
        else:
            beacon = tl.full((BLOCK,), False, tl.int1)
    dtype = x.dtype
    cosine = tl.load(COS+position*DIM+d, valid & ~beacon, 1).to(dtype).to(tl.float32)
    sine = tl.load(SIN+position*DIM+d, valid & ~beacon, 0).to(dtype).to(tl.float32)
    half = tl.where(d < DIM//2, -partner.to(tl.float32), partner.to(tl.float32))
    first = (x.to(tl.float32)*cosine).to(dtype).to(tl.float32)
    second = (half*sine).to(dtype).to(tl.float32)
    result = tl.where(beacon, x.to(tl.float32), first+second)
    tl.store(target+destination, result, valid)
