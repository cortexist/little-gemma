#!/usr/bin/env python3
"""Export an existing NumPy decision-head package to a deterministic GGUF v3.

NumPy is needed only for conversion. The C runtime has no Python dependency.
The original NPZ and contract.json are never modified.
"""
import argparse
import hashlib
import json
import re
import struct
from pathlib import Path

import numpy as np

U32, I32, F32, BOOL, STRING, ARRAY, F64 = 4, 5, 6, 7, 8, 9, 12


def string(value):
    raw = value.encode('utf-8')
    if b'\0' in raw:
        raise ValueError('Embedded NUL in string')
    return struct.pack('<Q', len(raw)) + raw


def write_gguf(path, metadata, tensors):
    """Minimal FP32 writer; tensor dimensions in GGUF are fastest-axis first."""
    prefix = bytearray(struct.pack('<IIQQ', 0x46554747, 3, len(tensors), len(metadata)))
    for name, (kind, value) in sorted(metadata.items()):
        prefix += string(name) + struct.pack('<I', kind)
        if kind == STRING:
            prefix += string(value)
        elif kind == ARRAY:
            prefix += struct.pack('<IQ', STRING, len(value))
            for item in value:
                prefix += string(item)
        else:
            prefix += struct.pack({U32: '<I', I32: '<i', F32: '<f', F64: '<d', BOOL: '<?'}[kind], value)
    payload = bytearray()
    for name, value in sorted(tensors.items()):
        a = np.ascontiguousarray(value, dtype='<f4')
        payload += b'\0' * (-len(payload) % 32)
        prefix += string(name) + struct.pack('<I', a.ndim)
        prefix += struct.pack('<' + 'Q' * a.ndim, *reversed(a.shape))
        prefix += struct.pack('<IQ', 0, len(payload))  # F32 tensor type
        payload += a.tobytes(order='C')
    prefix += b'\0' * (-len(prefix) % 32)
    # Exclusive creation prevents accidentally replacing a historical package.
    with Path(path).open('xb') as output:
        output.write(prefix)
        output.write(payload)


def package(source):
    source = Path(source)
    raw = (source / 'head.npz').read_bytes()
    contract = json.loads((source / 'contract.json').read_text(encoding='utf-8'))
    if hashlib.sha256(raw).hexdigest() != contract.get('head_sha256'):
        raise ValueError('Source NPZ hash does not match contract')
    with np.load(source / 'head.npz', allow_pickle=False) as archive:
        a = {k: archive[k] for k in archive.files}
    architecture = str(a['architecture'])
    if architecture not in ('linear', 'mlp64') or a['architecture'].shape != ():
        raise ValueError('Unsupported source architecture')
    if a['classes'].ndim != 1 or a['classes'].dtype.kind != 'U':
        raise ValueError('Classes must be an ordered Unicode vector')
    labels = a['classes'].tolist()
    if not labels or len(set(labels)) != len(labels) or any(not x or '\0' in x or len(x.encode()) > 4096 for x in labels):
        raise ValueError('Invalid or duplicate class labels')
    width = contract['hidden_dim']
    hidden, classes = (64 if architecture == 'mlp64' else 0), len(labels)
    ids = sorted({int(m.group(1)) for k in a if (m := re.fullmatch(r'model(\d+)\.net\..+', k))})
    if not ids or ids != list(range(len(ids))) or len(ids) > 32:
        raise ValueError('Ensemble member IDs must be contiguous from zero')
    count = len(ids)
    if type(width) is not int or not 1 <= width <= 65536 or classes > 4096:
        raise ValueError('Invalid dimensions')
    if count * (width * hidden + hidden + classes * (hidden or width) + classes) > 64 * 1024 * 1024:
        raise ValueError('Head exceeds v1 parameter limit')
    if a['temperature'].shape != ():
        raise ValueError('Temperature must be scalar')
    temperature = float(a['temperature'])
    if not np.isfinite(temperature) or not np.finfo(np.float32).tiny <= temperature <= np.finfo(np.float32).max:
        raise ValueError('Temperature must be positive and representable in FP32')
    used = {'architecture', 'classes', 'temperature'}
    tensors = {}

    def take(key, name, shape):
        x = a[key]
        if x.shape != shape or x.dtype.kind != 'f' or x.dtype.itemsize != 4 or not np.isfinite(x).all():
            raise ValueError(f'Invalid FP32 tensor: {key}')
        used.add(key)
        tensors[name] = np.asarray(x, dtype='<f4')

    take('mean', 'input.mean', (width,))
    take('scale', 'input.scale', (width,))
    if not (a['scale'] > 0).all():
        raise ValueError('Normalization scales must be positive')
    for i in ids:
        prefix = f'model{i}.net.'
        if hidden:
            take(prefix + '0.weight', f'member.{i}.hidden.weight', (hidden, width))
            take(prefix + '0.bias', f'member.{i}.hidden.bias', (hidden,))
            prefix += '3.'
        take(prefix + 'weight', f'member.{i}.output.weight', (classes, hidden or width))
        take(prefix + 'bias', f'member.{i}.output.bias', (classes,))
    if used != set(a):
        raise ValueError(f'Unrecognized source arrays: {sorted(set(a) - used)}')
    for key in ('task', 'model_sha256', 'input_format', 'capture_boundary'):
        if not isinstance(contract.get(key), str) or not contract[key] or '\0' in contract[key] or len(contract[key].encode()) > 4096:
            raise ValueError(f'Missing/invalid contract field: {key}')
    if not re.fullmatch('[0-9a-f]{64}', contract['model_sha256']):
        raise ValueError('Invalid base-model SHA256')
    if type(contract.get('layer')) is not int or not -(2**31) <= contract['layer'] < 2**31:
        raise ValueError('Invalid feature layer')
    if type(contract.get('answer_channel_seed')) is not bool:
        raise ValueError('Missing feature answer-channel boundary')
    metadata = {
        'general.architecture': (STRING, 'decision_head'),
        'general.alignment': (U32, 32),
        'decision_head.version': (U32, 1),
        'decision_head.architecture': (STRING, 'mlp' if hidden else 'linear'),
        'decision_head.activation': (STRING, 'tanh' if hidden else 'identity'),
        'decision_head.normalization': (STRING, 'standardize_clip'),
        'decision_head.ensemble_reduction': (STRING, 'mean_logits'),
        'decision_head.output_transform': (STRING, 'temperature_softmax'),
        'decision_head.input_dim': (U32, width),
        'decision_head.hidden_dim': (U32, hidden),
        'decision_head.output_dim': (U32, classes),
        'decision_head.ensemble_size': (U32, count),
        'decision_head.clip': (F32, 10.0),
        'decision_head.temperature': (F64, temperature),
        'decision_head.labels': (ARRAY, labels),
        'decision_head.task': (STRING, contract['task']),
        'decision_head.feature.model_sha256': (STRING, contract['model_sha256']),
        'decision_head.feature.input_format': (STRING, contract['input_format']),
        'decision_head.feature.capture_boundary': (STRING, contract['capture_boundary']),
        'decision_head.feature.layer': (I32, contract['layer']),
        'decision_head.feature.answer_channel_seed': (BOOL, contract['answer_channel_seed']),
        'decision_head.source_contract': (STRING, json.dumps(contract, sort_keys=True, separators=(',', ':'), allow_nan=False)),
    }
    return metadata, tensors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('package', type=Path, help='directory containing head.npz and contract.json')
    parser.add_argument('output', type=Path, help='new GGUF path; refuses to overwrite')
    args = parser.parse_args()
    try:
        metadata, tensors = package(args.package)
        write_gguf(args.output, metadata, tensors)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f'Export failed: {exc}\n')
    print(f'{args.output}: {len(tensors)} FP32 tensors, SHA256 {hashlib.sha256(args.output.read_bytes()).hexdigest()}')


if __name__ == '__main__':
    main()
