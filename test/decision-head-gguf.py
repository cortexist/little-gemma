"""Public synthetic differential and rejection tests; no research models needed."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

BINARY = Path(sys.argv.pop(1)).resolve()
SOURCE = Path(__file__).resolve().parents[1] / 'tools' / 'export_decision_head.py'
module = importlib.util.spec_from_file_location('export_decision_head', SOURCE)
export = importlib.util.module_from_spec(module)
module.loader.exec_module(export)


class DecisionHeadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def source(self, arch='mlp64', members=3):
        root = self.root / f'{arch}-{members}'
        root.mkdir()
        rng = np.random.default_rng(471)
        a = dict(mean=rng.normal(size=7).astype('f4'), scale=rng.uniform(.5, 2, 7).astype('f4'),
                 architecture=np.array(arch), temperature=np.array(1.7),
                 classes=np.array(['route', 'think', 'hold', 'name"quoted', 'écouter']))
        for i in range(members):
            p = f'model{i}.net.'
            if arch == 'mlp64':
                a[p+'0.weight'] = rng.normal(0, .2, (64, 7)).astype('f4')
                a[p+'0.bias'] = rng.normal(0, .2, 64).astype('f4')
                p += '3.'
            a[p+'weight'] = rng.normal(0, .2, (5, 64 if arch == 'mlp64' else 7)).astype('f4')
            a[p+'bias'] = rng.normal(0, .2, 5).astype('f4')
        np.savez(root/'head.npz', **a)
        contract = dict(hidden_dim=7, model_sha256='a'*64, task='generic-routing',
                        input_format='synthetic_v1', capture_boundary='fixture_only', layer=-1,
                        answer_channel_seed=False,
                        head_sha256=hashlib.sha256((root/'head.npz').read_bytes()).hexdigest())
        (root/'contract.json').write_text(json.dumps(contract))
        return root, a

    def run_cli(self, *args, ok=True, data=None):
        r = subprocess.run([str(BINARY), *map(str, args)], input=data, capture_output=True)
        if ok:
            self.assertEqual(r.returncode, 0, r.stderr.decode(errors='replace'))
        else:
            self.assertEqual(r.returncode, 1, r.stderr.decode(errors='replace'))
            self.assertNotIn(b'AddressSanitizer', r.stderr)
            self.assertNotIn(b'LeakSanitizer', r.stderr)
            self.assertNotIn(b'runtime error:', r.stderr)
        return r

    def test_export_and_numeric_parity(self):
        rng = np.random.default_rng(415)
        for arch, members in [('linear', 1), ('linear', 2), ('mlp64', 3)]:
            with self.subTest(arch=arch, members=members):
                source, a = self.source(arch, members)
                meta, tensors = export.package(source)
                gguf = source/'head.gguf'
                export.write_gguf(gguf, meta, tensors)
                export.write_gguf(source/'same.gguf', meta, tensors)
                self.assertEqual(gguf.read_bytes(), (source/'same.gguf').read_bytes())
                with self.assertRaises(FileExistsError):
                    export.write_gguf(gguf, meta, tensors)
                info = json.loads(self.run_cli('inspect', gguf).stdout)
                self.assertEqual(info['labels'], a['classes'].tolist())
                self.assertEqual(info['ensemble_size'], members)
                x = rng.normal(size=(40, 7)).astype('f4')
                x[:2] *= 100  # exercise input clipping
                rows = [json.loads(s) for s in self.run_cli('predict', gguf, '-', data=x.tobytes()).stdout.splitlines()]
                for raw, row in zip(x, rows, strict=True):
                    normalized = np.clip((raw-a['mean'])/a['scale'], -10, 10)
                    logits = []
                    for i in range(members):
                        p = f'model{i}.net.'
                        if arch == 'mlp64':
                            h = np.tanh(np.einsum('ij,j->i', a[p+'0.weight'], normalized)+a[p+'0.bias'])
                            logits.append(np.einsum('ij,j->i', a[p+'3.weight'], h)+a[p+'3.bias'])
                        else:
                            logits.append(np.einsum('ij,j->i', a[p+'weight'], normalized)+a[p+'bias'])
                    want = np.mean(logits, axis=0)
                    z = want/float(a['temperature']); z -= z.max()
                    probs = np.exp(z); probs /= probs.sum()
                    np.testing.assert_allclose(row['logits'], want, atol=1e-5, rtol=1e-5)
                    np.testing.assert_allclose(row['probabilities'], probs, atol=1e-6, rtol=1e-5)
                    self.assertEqual(row['choice'], int(probs.argmax()))
                self.run_cli('predict', gguf, '-', ok=False, data=b'')
                self.run_cli('predict', gguf, '-', ok=False, data=x[0].tobytes()[:-1])
                x[0, 0] = np.nan
                self.run_cli('predict', gguf, '-', ok=False, data=x[0].tobytes())

    def test_reject_malformed_packages(self):
        source, _ = self.source()
        meta, tensors = export.package(source)
        variants = []
        for key, value in [
            ('decision_head.version', (export.U32, 2)),
            ('decision_head.ensemble_reduction', (export.STRING, 'mean_probabilities')),
            ('decision_head.temperature', (export.F64, 0.0)),
            ('decision_head.temperature', (export.F64, float('nan'))),
            ('decision_head.input_dim', (export.U32, 1000000)),
            ('decision_head.output_dim', (export.STRING, '5')),
            ('decision_head.labels', (export.ARRAY, ['same']*5)),
            ('general.architecture', (export.STRING, 'gemma4')),
        ]:
            m = copy.deepcopy(meta); m[key] = value; variants.append((m, tensors))
        m = copy.deepcopy(meta); del m['decision_head.feature.capture_boundary']; variants.append((m, tensors))
        for name, array in [('input.scale', np.zeros(7)), ('input.mean', np.full(7, np.nan)),
                            ('member.0.hidden.weight', np.zeros((7,64))), ('unexpected', np.zeros(1))]:
            t = dict(tensors); t[name] = array; variants.append((meta, t))
        for i, (m, t) in enumerate(variants):
            p = self.root/f'bad-{i}.gguf'; export.write_gguf(p, m, t)
            self.run_cli('inspect', p, ok=False)
        good = self.root/'good.gguf'; export.write_gguf(good, meta, tensors)
        for suffix, data in [('truncated', good.read_bytes()[:-4]), ('header', good.read_bytes()[:20])]:
            p = self.root/f'{suffix}.gguf'; p.write_bytes(data); self.run_cli('inspect', p, ok=False)
        m = dict(meta); m['decision_head.versioN'] = (export.U32, 1)
        dup = self.root/'duplicate.gguf'; export.write_gguf(dup, m, tensors)
        dup.write_bytes(dup.read_bytes().replace(b'decision_head.versioN', b'decision_head.version'))
        self.run_cli('inspect', dup, ok=False)
        t = dict(tensors); t['input.meam'] = tensors['input.mean']
        dup = self.root/'duplicate-tensor.gguf'; export.write_gguf(dup, meta, t)
        dup.write_bytes(dup.read_bytes().replace(b'input.meam', b'input.mean'))
        self.run_cli('inspect', dup, ok=False)

    def test_source_hash_and_dtype(self):
        source, a = self.source('linear', 1)
        with (source/'head.npz').open('ab') as f: f.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'hash'):
            export.package(source)
        a['mean'] = a['mean'].astype('f8')
        np.savez(source/'head.npz', **a)
        contract = json.loads((source/'contract.json').read_text())
        contract['head_sha256'] = hashlib.sha256((source/'head.npz').read_bytes()).hexdigest()
        (source/'contract.json').write_text(json.dumps(contract))
        with self.assertRaisesRegex(ValueError, 'FP32'):
            export.package(source)


if __name__ == '__main__':
    unittest.main()
