"""Lossless BF16 cache IO over engine-owned physical pages.

Block tables count raw tokens. Beacon pages contain fewer rows, with each
layer's exact K/V ranks. The first Beacon page also owns the raw sink token.
"""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BeaconPageLayout:
    raw_block_size: int
    token_ratio: int
    key_rank: int
    value_rank: int
    raw_width: int

    def __post_init__(self):
        if min(self.raw_block_size, self.token_ratio, self.key_rank,
               self.value_rank, self.raw_width) <= 0:
            raise ValueError('Cache dimensions must be positive.')
        if self.raw_block_size % self.token_ratio:
            raise ValueError('Raw block size must be divisible by the token ratio.')

    @property
    def slots(self):
        return self.raw_block_size // self.token_ratio

    @property
    def width(self):
        return self.key_rank + self.value_rank

    @property
    def sink_elements(self):
        return 2 * self.raw_width

    @property
    def page_elements(self):
        return self.sink_elements + self.slots * self.width

    @property
    def storage_rows(self):
        return (self.page_elements + self.width - 1) // self.width


class BeaconPages:
    """Read and append compact rows without owning or allocating KV pages."""

    def __init__(self, pages, layout):
        if pages.dtype != torch.bfloat16:
            raise ValueError('SlimKV pages must retain BF16 checkpoint precision.')
        self.pages = pages.flatten(1)
        if self.pages.data_ptr() != pages.data_ptr():
            raise ValueError('Each physical page must have contiguous feature storage.')
        if self.pages.shape[1] < layout.page_elements:
            raise ValueError('Physical page is too small for the declared ranks.')
        self.layout = layout

    def _indices(self, block_table, start, count):
        logical = torch.arange(start, start + count, device=self.pages.device)
        table = block_table.to(device=self.pages.device, dtype=torch.long)
        physical = table[logical // self.layout.slots]
        columns = (self.layout.sink_elements +
                   (logical % self.layout.slots)[:, None] * self.layout.width +
                   torch.arange(self.layout.width, device=self.pages.device)[None])
        return physical[:, None], columns

    def read(self, block_table, count):
        if count == 0:
            return None, None
        page, column = self._indices(block_table, 0, count)
        rows = self.pages[page, column]
        k = rows[:, :self.layout.key_rank][None, None]
        v = rows[:, self.layout.key_rank:][None, None]
        return k, v

    def append(self, block_table, start, key, value):
        if key.dtype != self.pages.dtype or value.dtype != self.pages.dtype:
            raise ValueError('Cache writes must not change dtype.')
        if (key.ndim != 4 or value.ndim != 4 or key.shape[:2] != (1, 1)
                or value.shape[:3] != key.shape[:3]
                or key.shape[-1] != self.layout.key_rank
                or value.shape[-1] != self.layout.value_rank):
            raise ValueError('Expected independent B1 global-rank K/V tensors.')
        page, column = self._indices(block_table, start, key.shape[2])
        self.pages[page, column] = torch.cat([key[0, 0], value[0, 0]], dim=-1)

    def write_sink(self, root_block, key, value):
        rows = torch.cat([key.flatten(), value.flatten()])
        if rows.numel() != self.layout.sink_elements or rows.dtype != self.pages.dtype:
            raise ValueError('Sink cache must retain the original raw K/V dimensions.')
        self.pages[root_block, :self.layout.sink_elements].copy_(rows)

    def read_sink(self, root_block, heads, head_dim):
        if heads * head_dim != self.layout.raw_width:
            raise ValueError('Sink head dimensions differ from the page layout.')
        rows = self.pages[root_block, :self.layout.sink_elements]
        return tuple(x.reshape(1, heads, 1, head_dim) for x in rows.split(self.layout.raw_width))


class RawPages:
    """Read current-window raw KV using absolute raw-token positions."""

    def __init__(self, pages):
        if pages.dtype != torch.bfloat16 or pages.ndim != 5 or pages.shape[2] != 2:
            raise ValueError('Expected BF16 pages in block/token/KV/head/dim order.')
        self.pages = pages
        self.block_size = pages.shape[1]

    def read(self, block_table, start, count):
        if count == 0:
            return None, None
        positions = torch.arange(start, start + count, device=self.pages.device)
        table = block_table.to(device=self.pages.device, dtype=torch.long)
        rows = self.pages[table[positions // self.block_size], positions % self.block_size]
        return tuple(rows[:, kind].transpose(0, 1)[None] for kind in (0, 1))

    def write(self, block_table, start, key, value):
        if key.dtype != self.pages.dtype or value.dtype != self.pages.dtype:
            raise ValueError('Raw cache writes must retain BF16 values.')
        rows = torch.stack([key[0].transpose(0, 1), value[0].transpose(0, 1)], dim=1)
        if rows.shape[1:] != self.pages.shape[2:]:
            raise ValueError('Raw cache dimensions differ from the engine allocation.')
        positions = torch.arange(start, start + rows.shape[0], device=self.pages.device)
        table = block_table.to(device=self.pages.device, dtype=torch.long)
        self.pages[table[positions // self.block_size], positions % self.block_size] = rows
