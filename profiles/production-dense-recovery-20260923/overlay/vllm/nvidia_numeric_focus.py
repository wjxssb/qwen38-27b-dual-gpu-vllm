"""Bounded GDN stage observations, with per-layer chunk tracking."""
import hashlib
import os
from pathlib import Path
import re

import torch
from vllm import stability_telemetry as telemetry

_contexts = {}
_written_bytes = 0
MAX_DUMP_BYTES = 2 * 1024**3
DUMP_GENERATIONS = frozenset({9, 11})
DUMP_CHUNKS = frozenset({0, 1})


WIDE_STAGES = frozenset({'after_input_norm', 'gdn_after_qkvz_projection',
    'conv_after', 'chunk_rule_before', 'chunk_rule_after',
    'gdn_after_gated_norm', 'gdn_projection_local_and_reduced', 'gdn_after_output_projection'})
MAX_WIDE_CHUNKS = 12


def layer_index(layer):
    index = getattr(layer, 'layer_idx', None)
    if index is None:
        match = re.search(r'(?:^|\.)layers\.(\d+)(?:\.|$)', getattr(layer, 'prefix', ''))
        if not match: return None
        index = int(match[1])
    return index


def enabled(layer):
    mode = os.environ.get('NVIDIA_NUMERIC_GDN_FOCUS')
    index = layer_index(layer)
    selected = index == 18 if mode == '18' else (type(index) is int and 0 <= index < 64 and index % 4 != 3) if mode == 'gdn-all' else False
    return selected and (telemetry._generation.get() or 0) > 0


def checkpoint(layer, stage, **tensors):
    global _written_bytes
    if not enabled(layer): return
    if torch.cuda.is_current_stream_capturing(): return
    generation, rpc = telemetry._generation.get(), telemetry._rpc_id.get()
    index = layer_index(layer)
    context = _contexts.get(index)
    if context is None or context['generation'] != generation:
        context = _contexts[index] = {'generation': generation, 'rpc': None, 'chunk': -1}
    if stage == 'decoder_input' and rpc != context['rpc']:
        context.update(rpc=rpc, chunk=context['chunk']+1)
    _chunk = context['chunk']
    if _chunk < 0 or rpc != context['rpc']: return
    wide = os.environ.get('NVIDIA_NUMERIC_GDN_FOCUS') == 'gdn-all'
    if wide and (stage not in WIDE_STAGES or _chunk >= MAX_WIDE_CHUNKS): return
    assert re.fullmatch('[a-z0-9_]+', stage)
    rows = {}
    for name, value in tensors.items():
        assert re.fullmatch('[a-z0-9_]+', name)
        if value is None:
            rows[name] = None
            continue
        host = value.detach().to(device='cpu').contiguous()
        raw = host.reshape(-1).view(torch.uint8).numpy().reshape(-1)
        dtype = str(value.dtype)
        if dtype == 'torch.bfloat16':
            finite = not bool(((raw.view('uint16') & 0x7F80) == 0x7F80).any())
        elif dtype == 'torch.float32':
            finite = not bool(((raw.view('uint32') & 0x7F800000) == 0x7F800000).any())
        elif dtype == 'torch.float8_e4m3fn':
            finite = not bool(((raw & 0x7F) == 0x7F).any())
        elif dtype in ['torch.int32', 'torch.int64', 'torch.bool']:
            finite = True
        else:
            raise ValueError('GDN_FOCUS_UNDECLARED_DTYPE:' + dtype)
        row = {'shape': list(value.shape), 'stride': list(value.stride()), 'dtype': dtype,
               'source_ptr': value.data_ptr(), 'bytes': raw.nbytes, 'finite': finite,
               'sha256': hashlib.sha256(memoryview(raw)).hexdigest()}
        if not wide and generation in DUMP_GENERATIONS and _chunk in DUMP_CHUNKS:
            if _written_bytes + raw.nbytes > MAX_DUMP_BYTES:
                row['dump_status'] = 'BUDGET_EXHAUSTED_NOT_SAVED'
            else:
                root = Path(os.environ['VLLM_STABILITY_TELEMETRY_DIR']).parent/'numeric-focus'
                root.mkdir(exist_ok=True)
                path = root/f'pid{os.getpid()}-g{generation}-rpc{rpc}-c{_chunk}-{stage}-{name}.bin'
                with path.open('xb') as f:
                    f.write(memoryview(raw))
                    f.flush()
                    os.fsync(f.fileno())
                _written_bytes += raw.nbytes
                row.update(dump_status='SAVED', dump_file=path.name)
        rows[name] = row
    telemetry.emit('gdn_focus_fingerprint', layer=index, stage=stage, chunk=_chunk,
                   tensors=rows, saved_bytes_this_rank=_written_bytes,
                   scope='CPU_D2H_FOCUS_OBSERVER_NO_OPERATOR_CHANGE')


def state_checkpoint(layer, stage, cache, indices, **tensors):
    if not enabled(layer): return
    if torch.cuda.is_current_stream_capturing(): return
    if os.environ.get('NVIDIA_NUMERIC_GDN_FOCUS') == 'gdn-all' and stage not in WIDE_STAGES: return
    # CPU index observation followed by view selection; no GPU gather/allocation.
    slots = indices.detach().to(device='cpu').reshape(-1).tolist()
    if not 0 < len(slots) <= 4: raise ValueError('GDN_FOCUS_UNDECLARED_SLOT_COUNT')
    for position, slot in enumerate(slots):
        if type(slot) is not int or not 0 <= slot < cache.shape[0]:
            raise ValueError('GDN_FOCUS_INVALID_CACHE_SLOT')
        tensors['cache_slot_' + str(position)] = cache.select(0, slot)
    checkpoint(layer, stage, cache_indices=indices, **tensors)


def projection_checkpoint(layer, input_value, local, reduced):
    if not getattr(layer, 'prefix', '').endswith('.linear_attn.out_proj'): return
    if not enabled(layer): return
    checkpoint(layer, 'gdn_projection_local_and_reduced', input=input_value,
               local=local, reduced=reduced, weight=layer.weight,
               input_scale=getattr(layer,'input_scale',None),
               weight_scale=getattr(layer,'weight_scale',None))
