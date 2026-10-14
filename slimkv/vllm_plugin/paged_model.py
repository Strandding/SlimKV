"""Restore transient attention workspaces from canonical engine-owned KV pages."""
import torch

from .paged_cache import BeaconPages, BeaconPageLayout, RawPages


def restore_memory(model, processed, raw_layers, beacon_layers, raw_table, beacon_table):
    memory = model.memory
    memory.reset()
    if processed == 0:
        return
    window, ratio = model.config.beacon_window, model.config.beacon_ratio[0]
    raw_count = processed % window
    history_count = processed // window * (window // ratio)
    beacon_count = processed // ratio
    # Only the suffix at _end_idx is ever embedded. Prior token IDs serve as
    # length bookkeeping; all prior attention values come from the KV pages.
    memory.all_input_ids = torch.zeros(1, processed, dtype=torch.long, device=memory._cpu)
    memory.all_attention_mask = torch.ones_like(memory.all_input_ids)
    memory._start_idx = processed - raw_count
    memory._end_idx = processed
    memory._step_idx = max(1, (processed + window - 1) // window)
    memory._is_full_window = raw_count == 0
    memory._interleave_remainder = raw_count % ratio
    memory._interleave_compression_ratio = ratio
    memory._compression_ratio = ratio
    memory._select_cache_mode()
    device = model.model.embed_tokens.weight.device
    if raw_count:
        count = raw_count + raw_count // ratio
        roles = torch.zeros(count, dtype=torch.long, device=device)
        roles[ratio::ratio + 1] = 1
        memory._beacon_indices = roles
        memory._window_roles = roles.bool()
        memory._window_raw_indices = (roles == 0).nonzero().flatten()
    for index, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        raw_blocks = raw_table[index] if isinstance(raw_table, list) else raw_table
        beacon_blocks = beacon_table[index] if isinstance(beacon_table, list) else beacon_table
        layout = BeaconPageLayout(beacon_layers[index].spec.block_size, ratio,
            attn.beacon_k_up_proj.in_features, attn.beacon_v_up_proj.in_features,
            attn.num_key_value_heads * attn.head_dim)
        beacon = BeaconPages(beacon_layers[index].kv_cache, layout)
        key, value = beacon.read(beacon_blocks, beacon_count)
        memory.beacon_activations[index] = tuple(x[:, :, :history_count] if history_count else None
                                               for x in (key, value))
        memory.window_beacon_activations[index] = tuple(x[:, :, history_count:] if beacon_count > history_count else None
                                                      for x in (key, value))
        if model.config.beacon_sink_size:
            memory.sink_activations[index] = beacon.read_sink(beacon_blocks[0], attn.num_key_value_heads, attn.head_dim)
        memory.raw_activations[index] = RawPages(raw_layers[index].kv_cache).read(
            raw_blocks, processed - raw_count, raw_count)


def flush_memory(model, previous, current, raw_layers, beacon_layers, raw_table, beacon_table):
    memory = model.memory
    ratio, window = model.config.beacon_ratio[0], model.config.beacon_window
    old_beacons = previous // ratio
    raw_start = max(previous, current - current % window)
    for index, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        raw_blocks = raw_table[index] if isinstance(raw_table, list) else raw_table
        beacon_blocks = beacon_table[index] if isinstance(beacon_table, list) else beacon_table
        layout = BeaconPageLayout(beacon_layers[index].spec.block_size, ratio,
            attn.beacon_k_up_proj.in_features, attn.beacon_v_up_proj.in_features,
            attn.num_key_value_heads * attn.head_dim)
        pages = BeaconPages(beacon_layers[index].kv_cache, layout)
        if current // ratio > old_beacons:
            tensors = []
            for kind in (0, 1):
                parts = [group[index][kind] for group in (memory.beacon_activations, memory.window_beacon_activations)
                         if group[index][kind] is not None]
                all_beacons = parts[0] if len(parts) == 1 else torch.cat(parts, dim=2)
                if all_beacons.shape[2] != current // ratio:
                    raise RuntimeError('Beacon count does not match the raw-token cursor.')
                tensors.append(all_beacons[:, :, old_beacons:])
            pages.append(beacon_blocks, old_beacons, *tensors)
        if model.config.beacon_sink_size:
            pages.write_sink(beacon_blocks[0], *memory.sink_activations[index])
        if current > raw_start:
            key, value = memory.raw_activations[index]
            RawPages(raw_layers[index].kv_cache).write(raw_blocks, raw_start,
                key[:, :, -(current - raw_start):], value[:, :, -(current - raw_start):])


def release_workspace_views(memory):
    """Retain no request KV in the model after it is committed to pages."""
    for name in ('sink_activations', 'raw_activations', 'beacon_activations', 'window_beacon_activations'):
        setattr(memory, name, [(None, None) for _ in range(memory.config.num_hidden_layers)])
