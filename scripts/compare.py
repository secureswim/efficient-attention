import csv
import re
import sys
from pathlib import Path

results = Path(sys.argv[1])
runs = {}
timings = []
for name in ('batched_run', 'batched_reverse', 'batched_repeat'):
    text = (results / f'{name}.log').read_text(encoding='utf-8', errors='replace')
    if 'ERROR ' in text or 'model loading failed' in text:
        raise SystemExit(f'{name}: runtime or model loading error')
    streams = {}
    current = None
    in_text = False
    load_match = re.search(r'^MODEL_LOAD_MS\s+([0-9.]+)', text, re.MULTILINE)
    if not load_match:
        raise SystemExit(f'{name}: missing model load time')
    timing_index = 0
    for line in text.splitlines():
        if in_text and not line.startswith(('RUN\t', 'TIMING\t', 'COMPARE\t')):
            streams[current]['text'] += '\n' + line
            continue
        in_text = False
        if line.startswith('RUN\t'):
            fields = line.split('\t')
            current = (fields[1], fields[2])
            streams[current] = {'threads': int(fields[3].split('=')[1]), 'prompt_tokens': int(fields[4].split('=')[1]), 'generated_tokens': int(fields[5].split('=')[1])}
        elif line.startswith('IDS '):
            if current is None:
                raise SystemExit(f'{name}: token IDs lack a run header')
            streams[current]['ids'] = [int(v) for v in line[4:].split()]
        elif line.startswith('TEXT '):
            streams[current]['text'] = line[5:]
            in_text = True
        elif line.startswith('TIMING\t'):
            fields = line.split('\t')
            prompt = ('A', 'B', 'both')[timing_index]
            timing_index += 1
            key = (fields[1], prompt)
            timing = {'file': name, 'mode': fields[1], 'prompt': prompt,
                      'threads': streams[(fields[1], 'A')]['threads'] if prompt == 'both' else streams[key]['threads'],
                      'prompt_tokens': sum(streams[(fields[1], p)]['prompt_tokens'] for p in ('A', 'B')) if prompt == 'both' else streams[key]['prompt_tokens'],
                      'generated_tokens': sum(streams[(fields[1], p)]['generated_tokens'] for p in ('A', 'B')) if prompt == 'both' else streams[key]['generated_tokens'],
                      'model_load_ms': float(load_match.group(1))}
            timing.update({k: float(v) for k, v in (field.split('=', 1) for field in fields[2:])})
            timings.append(timing)
    for mode in ('single', 'batched_reverse' if name == 'batched_reverse' else 'batched'):
        for prompt in ('A', 'B'):
            entry = streams.get((mode, prompt))
            if not entry or not entry.get('ids') or not entry.get('text'):
                raise SystemExit(f'{name}: missing output for {mode} {prompt}')
            if len(entry['ids']) != entry['generated_tokens']:
                raise SystemExit(f'{name}: count mismatch for {mode} {prompt}')
    runs[name] = streams

lines = ['Criterion: identical greedy token IDs and decoded text, all 24 generated tokens.']
ok = True
for name, streams in runs.items():
    mode = 'batched_reverse' if name == 'batched_reverse' else 'batched'
    for prompt in ('A', 'B'):
        single = streams[('single', prompt)]
        batch = streams[(mode, prompt)]
        first = next((i for i, (a, b) in enumerate(zip(single['ids'], batch['ids'])) if a != b), None)
        if first is None and len(single['ids']) != len(batch['ids']):
            first = min(len(single['ids']), len(batch['ids']))
        match = first is None and single['text'] == batch['text']
        ok &= match
        lines.append(f'{name} {prompt}: {"MATCH" if match else "MISMATCH"}; first differing token index: {first if first is not None else "none"}')

if runs['batched_run'][('batched', 'B')]['ids'] != runs['batched_reverse'][('batched_reverse', 'B')]['ids']:
    lines.append('B differs when batch order is reversed.')
    ok = False
else:
    lines.append('B batched IDs are identical in both batch orders.')
if runs['batched_run'][('batched', 'B')]['ids'] != runs['batched_repeat'][('batched', 'B')]['ids']:
    lines.append('B differs between repeated batched runs.')
    ok = False
else:
    lines.append('B batched IDs are identical on repetition.')

(results / 'comparison.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
with (results / 'timings.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['file', 'mode', 'prompt', 'threads', 'prompt_tokens', 'generated_tokens', 'model_load_ms', 'tokenize_ms', 'prefill_ms', 'generation_ms', 'total_ms'])
    writer.writeheader()
    writer.writerows(timings)
print('\n'.join(lines))
raise SystemExit(0 if ok else 1)
