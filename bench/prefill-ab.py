#!/usr/bin/env python3
"""Warm serve prefill A/B; fresh connection per turn, first turn discarded.

Example (pin Orin clocks first):
  python3 bench/prefill-ab.py --model MODEL --out /tmp/prefill-ab \
      --config LG_Q4K_SWAPXY=0 --config LG_Q4K_SWAPXY=1 --config LG_Q4K_SWAPXY=0

Each config starts a fresh server. Raw logs/replies and JSON summaries are saved;
the caller must assess correctness on varied prompts, not just line929s.txt.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import tempfile
import time


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bin', type=Path, default=repo / 'build/run-cuda-i8')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--prompt', type=Path, default=repo / 'bench/line929s.txt')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--config', action='append', required=True,
                        help='comma-separated NAME=value assignments; repeat for A/B/A')
    parser.add_argument('--turns', type=int, default=6, help='including discarded first turn')
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--server-arg', action='append', default=[],
                        help='extra server argument (use --server-arg=-mtp for flags)')
    args = parser.parse_args()
    if args.turns < 2:
        parser.error('--turns must be at least 2')
    configs = []
    for config in args.config:
        assignments = {}
        for item in config.split(','):
            name, sep, value = item.partition('=')
            if not sep or not name.startswith('LG_'):
                parser.error('--config requires LG_NAME=value assignments')
            assignments[name] = value
        configs.append(assignments)
    binary = str(args.bin.resolve())
    prompt = args.prompt.read_bytes().rstrip(b'\n') + b'\n'
    if b'\n' in prompt[:-1]:
        parser.error('--prompt must contain one serve turn on one line')
    args.out.mkdir(parents=True, exist_ok=False)
    inherited = {k: v for k, v in os.environ.items() if k.startswith('LG_')}
    (args.out / 'manifest.json').write_text(json.dumps({
        'binary': binary, 'binary_sha256': hashlib.sha256(Path(binary).read_bytes()).hexdigest(),
        'model': str(args.model.resolve()), 'prompt': str(args.prompt.resolve()),
        'prompt_sha256': hashlib.sha256(prompt).hexdigest(), 'configs': configs,
        'inherited_lg_env': inherited, 'turns': args.turns,
        'server_args': args.server_arg,
    }, indent=2) + '\n')
    pattern = re.compile(r'turn: (\d+) in [\d.]+s \(([\d.]+) tok/s\), (\d+) out [\d.]+s \(([\d.]+) tok/s\)')
    for index, config in enumerate(configs):
        tag = str(index)
        log = args.out / (tag + '.err')
        replies = []
        with tempfile.TemporaryDirectory(prefix='lg-pf-') as tmp, \
                log.open('wb') as err, (args.out / (tag + '.load')).open('wb') as load:
            sock = Path(tmp) / 'serve.sock'
            server = subprocess.Popen([binary, '-m', str(args.model.resolve()), '-s', str(sock)] + args.server_arg,
                                      env=dict(os.environ, **config), stdout=load, stderr=err)
            try:
                deadline = time.monotonic() + args.timeout
                while not sock.exists():
                    if server.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError('server failed or timed out; see ' + str(log))
                    time.sleep(.1)
                for turn in range(args.turns):
                    result = subprocess.run([binary, '-c', str(sock)], input=prompt,
                                            capture_output=True, timeout=args.timeout, check=True)
                    (args.out / f'{tag}-{turn}.out').write_bytes(result.stdout)
                    replies.append(hashlib.sha256(result.stdout).hexdigest())
            finally:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
        rows = pattern.findall(log.read_text())
        if len(rows) != args.turns:
            raise RuntimeError(f'expected {args.turns} turn stats, got {len(rows)}; see {log}')
        rates = [float(row[1]) for row in rows[1:]]
        summary = {'config': config, 'input_tokens': [int(row[0]) for row in rows],
                   'warm_prefill': rates, 'median_prefill': statistics.median(rates),
                   'warm_decode': [float(row[3]) for row in rows[1:]],
                   'reply_sha256': replies}
        (args.out / (tag + '.json')).write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
