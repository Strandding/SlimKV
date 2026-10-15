"""Reuse shared KV across the raw/Beacon queries without changing split reductions."""
import math
from functools import lru_cache


@lru_cache(None)
def original_splits(width, key_length, query_heads, sm_count):
    """Match the unpacked Q2 launch, including its original reduction boundaries."""
    blocks = (key_length + 63) // 64
    if width == 320:
        splits = min(32, blocks, (sm_count + query_heads - 1) // query_heads)
        while splits > 1 and (splits - 1) * ((blocks + splits - 1) // splits) >= blocks:
            splits -= 1
        return splits
    sms = 2 * sm_count
    if query_heads >= 0.8 * sms:
        return 1
    block_n = 128 if width <= 128 else 64
    blocks = (key_length + block_n - 1) // block_n
    scores = {}
    for splits in range(1, min(128, sms, blocks) + 1):
        if splits > 1 and math.ceil(blocks / splits) == math.ceil(blocks / (splits - 1)):
            continue
        waves = query_heads * splits / sms
        scores[splits] = waves / math.ceil(waves)
    best = max(scores.values())
    return next(splits for splits, score in scores.items() if score >= 0.85 * best)


def paired_attention(forward, q, k, v, scale, paired_lengths, sm_count):
    """Represent Q2 causal attention as two Q1 sequences sharing the same KV view.

    The raw query excludes the newly inserted Beacon. The Beacon query includes
    it. Keeping the original split count preserves the per-row softmax reduction
    order while FA2 packs each query's GQA heads into its query dimension.
    """
    if q.shape[:2] != (1, 2) or k.shape[0] != 1:
        raise ValueError('Paired decode requires B1 and exactly two queries.')
    splits = original_splits(q.shape[-1], k.shape[1], q.shape[2], sm_count)
    output, lse = forward(q.transpose(0, 1), k.expand(2, -1, -1, -1),
                          v.expand(2, -1, -1, -1), scale, False, splits, True,
                          paired_lengths)
    return output.transpose(0, 1), lse.squeeze(-1).transpose(0, 1).unsqueeze(0)
