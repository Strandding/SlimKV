"""Measured runtime policies for Llama-3.1-8B-r4 inference."""

SHORT_ANSWER_TASKS = frozenset({
    'narrativeqa', 'qasper', 'multifieldqa_en', 'hotpotqa', '2wikimqa', 'musique',
})


def resolve_preset(requested='auto', capability=None, device=None):
    """Select a policy; unsupported devices retain explicit/manual settings."""
    if requested not in ('auto', 'manual', 'sm90', 'sm120'):
        raise ValueError('performance_preset must be auto, manual, sm90 or sm120.')
    if requested != 'auto':
        return requested
    if capability is None:
        import torch
        if (device is not None and torch.device(device).type != 'cuda') or not torch.cuda.is_available():
            return 'manual'
        capability = torch.cuda.get_device_capability(device)
    return {(9, 0): 'sm90', (12, 0): 'sm120'}.get(tuple(capability), 'manual')


def longbench_settings(preset, task=None):
    """Avoid per-prompt Graph setup for short answers on the SM120 preset."""
    if preset == 'manual':
        return {}
    if preset not in ('sm90', 'sm120'):
        raise ValueError('Resolve the hardware preset before selecting task settings.')
    return {
        'beacon_decode_graph': not (preset == 'sm120' and task in SHORT_ANSWER_TASKS),
        'beacon_decode_compile': True,
    }


def configure_longbench(config, preset, task=None, explicit_options=()):
    """Apply defaults while preserving every explicitly supplied runtime flag."""
    for name, value in longbench_settings(preset, task).items():
        if name not in explicit_options:
            setattr(config, name, value)
