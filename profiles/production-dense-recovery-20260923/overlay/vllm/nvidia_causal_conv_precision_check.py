"""Bounded startup regression on private tensors; never reads/writes model state."""
import hashlib
import json
import os
from pathlib import Path
import time

import torch
from vllm import stability_telemetry as telemetry


def reference(x, weight, state, lengths, bias=None, activation=None):
    """CPU FP32 products and accumulation, one independent history per sequence."""
    assert x.device.type == weight.device.type == state.device.type == 'cpu'
    output, final, start = [], [], 0
    width = weight.shape[-1]
    for index, length in enumerate(lengths):
        history = torch.cat((state[index], x[:, start:start+length]), -1)
        acc = torch.zeros(x.shape[0], length, dtype=torch.float32)
        if bias is not None: acc += bias.float()[:, None]
        for tap in range(width):
            acc += history[:, tap:tap+length].float() * weight[:, tap:tap+1].float()
        if activation == 'silu': acc = torch.nn.functional.silu(acc)
        output.append(acc.to(x.dtype)); final.append(history[:, -(width-1):])
        start += length
    assert start == x.shape[-1]
    return torch.cat(output, -1), torch.stack(final)


def fixtures():
    # All tensors originate on CPU; no change to the worker's RNG state.
    plans = [('prefill', 8, [5], 4, True, False, None),
             ('decode', 8, [5], 4, True, False, None),
             ('prefill', 8, [2, 3], 4, True, False, None),
             ('decode', 8, [2, 1], 4, True, False, None),
             ('prefill', 257, [3, 17], 2, True, True, 'silu'),
             ('decode', 257, [3, 1], 3, True, True, 'silu'),
             ('prefill', 5120, [2048], 4, False, True, 'silu'),
             ('prefill', 5120, [2048], 4, True, True, 'silu')]
    for mode, dim, lengths, width, initial, bias, activation in plans:
        length = sum(lengths)
        # First four cases reproduce the upstream exact deterministic fixture.
        x = (torch.arange(dim*length).reshape(dim, length) % 113 / 7).bfloat16()
        weight = (torch.arange(dim*width).reshape(dim, width) % 89 / 11).bfloat16()
        state = (torch.arange(dim*(width-1)).reshape(1,dim,width-1) % 61 / 13).bfloat16()
        state = torch.cat([state+i for i in range(len(lengths))])
        effective = state if initial else torch.zeros_like(state)
        bias = (torch.arange(dim) % 13 / 17).bfloat16() if bias else None
        yield dict(mode=mode, dim=dim, lengths=lengths, width=width, initial=initial,
                   activation=activation, x=x, weight=weight, state=state,
                   effective_state=effective, bias=bias)


def execute(module, case, device):
    n = len(case['lengths'])
    x = case['x'].T.contiguous().T.to(device)
    weight = case['weight'].to(device)
    bias = case['bias'].to(device) if case['bias'] is not None else None
    # Reverse slot order exercises indirect addressing; slot 0 stays a canary.
    indices_cpu = torch.arange(n, 0, -1, dtype=torch.int32)
    states_cpu = torch.full((n+1, case['dim'], case['width']-1), -31, dtype=x.dtype)
    states_cpu[indices_cpu.long()] = case['state']
    states = states_cpu.to(device)
    indices = indices_cpu.to(device)
    query = torch.tensor([0, *torch.tensor(case['lengths']).cumsum(0).tolist()],
                         dtype=torch.int32, device=device)
    if case['mode'] == 'prefill':
        actual = module.causal_conv1d_fn(x, weight, bias, states, query,
            cache_indices=indices,
            has_initial_state=torch.full((n,), case['initial'], dtype=torch.bool, device=device),
            activation=case['activation'])
    else:
        actual = module.causal_conv1d_update(x.T, states, weight, bias,
            activation=case['activation'], conv_state_indices=indices,
            query_start_loc=query, max_query_len=max(case['lengths']),
            out=torch.empty_like(x.T)).T
    torch.cuda.synchronize(device)
    actual_cpu, states_after = actual.cpu(), states.cpu()
    assert torch.equal(x.cpu(), case['x']), 'PRIVATE_INPUT_MUTATED'
    assert torch.equal(weight.cpu(), case['weight']), 'PRIVATE_WEIGHT_MUTATED'
    assert torch.equal(states_after[0], states_cpu[0]), 'NULL_STATE_SLOT_MODIFIED'
    return actual_cpu, states_after[indices_cpu.long()]


def run(device, rank):
    from vllm import nvidia_causal_conv_original as original
    from vllm.model_executor.layers.mamba.ops import causal_conv1d as repaired
    root = Path(os.environ['VLLM_STABILITY_TELEMETRY_DIR']).parent
    receipt = {'scope': 'PRIVATE_TENSOR_ORIGINAL_VS_PRODUCT_AND_VARLEN_STATE_REPAIR',
        'rank': rank, 'pid': os.getpid(), 'started_epoch': time.time(), 'cases': [],
        'state': 'RUNNING', 'model_state_accessed': False, 'speculative_decoding': False}
    path = root / f'causal-conv-precision-rank-{rank}.json'
    assert not path.exists(), 'KERNEL_REGRESSION_MUST_NOT_REPLAY'
    def save():
        temporary = path.with_suffix('.tmp')
        with temporary.open('w') as f:
            json.dump(receipt, f, indent=2); f.flush(); os.fsync(f.fileno())
        os.replace(temporary, path)
    save()
    try:
        for index, case in enumerate(fixtures()):
            telemetry.emit('causal_conv_precision_case_begin', index=index, rank=rank)
            expected, final = reference(case['x'], case['weight'], case['effective_state'],
                case['lengths'], case['bias'], case['activation'])
            row = {k:case[k] for k in ['mode','dim','lengths','width','initial','activation']}
            row.update(index=index, state='CASE_STARTED')
            receipt['cases'].append(row); save()
            for name, module in [('original', original), ('repaired', repaired)]:
                actual, state = execute(module, case, device)
                delta = (actual.float()-expected.float()).abs()
                row[name] = {'different_elements': int((actual != expected).sum()),
                    'max_abs_error': float(delta.max()), 'finite': bool(actual.isfinite().all()),
                    'state_equal': bool(torch.equal(state, final)),
                    'state_different_elements': int((state != final).sum()),
                    'state_first16_actual': state.float().flatten().tolist()[:16],
                    'state_first16_expected': final.float().flatten().tolist()[:16],
                    'output_sha256': hashlib.sha256(actual.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}
                save()  # Preserve the current path's actual result before any assertion.
                assert row[name]['finite'], 'NONFINITE_PRIVATE_KERNEL_OUTPUT'
                if name == 'repaired' or case['mode'] != 'decode' or len(set(case['lengths'])) == 1:
                    assert row[name]['state_equal'], 'PRIVATE_STATE_ERROR'
                else:
                    assert not row[name]['state_equal'], 'ORIGINAL_VARLEN_STATE_DEFECT_NOT_REPRODUCED'
                if name == 'repaired':
                    # The rational, no-activation BF16 reference must match exactly.
                    # SiLU uses different CPU/GPU exp implementations: one BF16 ULP.
                    torch.testing.assert_close(actual, expected,
                        rtol=0 if case['activation'] is None else 2**-7, atol=0)
            row['state'] = 'CASE_COMPLETE'; save()
            telemetry.emit('causal_conv_precision_case_end', index=index, rank=rank, result=row)
        assert all(c['original']['different_elements'] > 0 for c in receipt['cases'][:4]), 'ORIGINAL_REGRESSION_NOT_REPRODUCED'
        receipt['state'] = 'PRIVATE_KERNEL_REGRESSION_PASS_MODEL_BASELINE_NOT_PROVEN'
    except BaseException as error:
        receipt.update(state='FAILED_NO_RETRY', error=type(error).__name__+':'+str(error))
        raise
    finally:
        receipt['ended_epoch'] = time.time(); save()
    telemetry.emit('causal_conv_precision_completed', rank=rank, receipt=str(path), state=receipt['state'])
