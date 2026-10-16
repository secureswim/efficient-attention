"""Parse five Python modules, run the CPU driver, and summarize strict comparisons."""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'models' / 'Llama-3.2-1B-Instruct-Q4_K_M.gguf'


def default_driver():
    name = 'structured-attention-driver.exe' if os.name == 'nt' else 'structured-attention-driver'
    candidates = [ROOT / 'build' / 'bin' / name, ROOT / 'build' / 'bin' / 'Release' / name]
    return next((path for path in candidates if path.is_file()), candidates[0])


DRIVER = default_driver()
DEFAULT_OUT = ROOT / 'results' / 'local_source_code_domain'


def run(args, log):
    result = subprocess.run(args, capture_output=True, text=True, encoding='utf-8', errors='replace')
    log.write_text(result.stdout + '\n--- STDERR ---\n' + result.stderr, encoding='utf-8')
    if result.returncode:
        raise RuntimeError(f'command failed ({result.returncode}): {args[0]} — see {log}')
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--driver', type=Path, default=DRIVER)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--max-tokens', type=int, default=16)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUT,
                        help='result folder; default is ignored by Git')
    parser.add_argument('--with-tests', action='store_true', help='run parser and driver boundary tests after experiments')
    args = parser.parse_args()
    model = args.model.expanduser().resolve()
    driver = args.driver.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    parsed_dir = out / 'parsed'
    if args.threads < 1 or args.max_tokens < 1:
        parser.error('--threads and --max-tokens must be positive')
    if not model.is_file():
        raise RuntimeError(f'model missing: {model}')
    if not driver.is_file():
        raise RuntimeError(f'compiled structured-attention-driver missing: {driver}')
    out.mkdir(parents=True, exist_ok=True)
    parsed_dir.mkdir(parents=True, exist_ok=True)
    cases = json.loads((ROOT / 'samples' / 'source_code' / 'cases.json').read_text(encoding='utf-8'))
    stats_rows, time_rows, comparison_rows = [], [], []
    for case in cases:
        case_id = case['id']
        base_path = parsed_dir / f'{case_id}.json'
        modified_path = parsed_dir / f'{case_id}_modified.json'
        for suffix, output in (('', base_path), ('_modified', modified_path)):
            fixture = ROOT / 'samples' / 'source_code' / f'{case_id}{suffix}.py'
            parse_start = time.perf_counter()
            print(run([sys.executable, str(ROOT / 'parsers' / 'python_source.py'), '--input', str(fixture),
                       '--output', str(output), '--model', str(model), '--driver', str(driver),
                       '--document-id', case_id], out / f'{case_id}{suffix}_parse.log').strip(), flush=True)
            if not suffix:
                parse_tokenize_ms = (time.perf_counter() - parse_start) * 1000
        parsed = json.loads(base_path.read_text(encoding='utf-8'))
        changed = json.loads(modified_path.read_text(encoding='utf-8'))
        if parsed['stats']['model_token_count'] >= 2000 or changed['stats']['model_token_count'] >= 2000:
            raise RuntimeError(f'{case_id}: document exceeds 2000 model tokens')
        try:
            import jsonschema
        except ImportError:
            jsonschema = None
        if jsonschema:
            spec = json.loads((ROOT / 'schema' / 'source_code_document.schema.json').read_text(encoding='utf-8'))
            jsonschema.validate(parsed, spec)
            jsonschema.validate(changed, spec)
        result_path = out / f'{case_id}_result.json'
        print(run([str(driver), 'experiment', str(model), str(base_path), str(modified_path),
                   case['focus'], case['unrelated'], case['question'], str(args.max_tokens), str(args.threads), str(result_path)],
                  out / f'{case_id}_experiment.log').strip(), flush=True)
        result = json.loads(result_path.read_text(encoding='utf-8'))
        answer = result['scenarios']['baseline']['text'].strip()
        answer_contains_expected = case['expected'].lower() in answer.lower()
        if not answer_contains_expected:
            raise RuntimeError(f'{case_id}: baseline answer {answer!r} lacks expected value {case["expected"]!r}')
        for label, comp in result['comparisons'].items():
            comparison_rows.append({'document': case_id, 'comparison': label,
                                    'match': comp['match'], 'first_different_token': comp['first_different_token'],
                                    'answer': answer})
            if not comp['match']:
                raise RuntimeError(f'{case_id}: {label} token/text comparison failed')
        evicted = result['scenarios']['evicted_B']
        if not (evicted['kv_after_bytes'] < evicted['kv_before_bytes'] and
                evicted['kv_after_pos'] < evicted['kv_before_pos']):
            raise RuntimeError(f'{case_id}: no live KV state reduction was measured')
        s = parsed['stats']
        stats_rows.append({'document': case_id, 'max_depth': s['max_depth'], 'nodes': s['node_count'],
                           'groups': s['group_count'], 'group_token_sizes': ';'.join(map(str, s['group_token_sizes'])),
                           'model_tokens': s['model_token_count'], 'syntax_tokens': s['pure_syntax_tokens'],
                           'source_tokens': s['source_tokens'], 'syntax_share': s['pure_syntax_share'],
                           'kv_before_bytes': evicted['kv_before_bytes'], 'kv_after_bytes': evicted['kv_after_bytes']})
        for label, scenario in result['scenarios'].items():
            time_rows.append({'document': case_id, 'scenario': label, 'model_load_ms': result['model_load_ms'],
                              'parse_tokenize_ms': round(parse_tokenize_ms, 4),
                              'document_tokens': result['document_tokens'], 'generated_tokens': scenario['generated_tokens'],
                              'threads': result['threads'], 'context_ms': scenario['context_ms'],
                              'document_prefill_ms': scenario['document_prefill_ms'],
                              'eviction_ms': scenario['eviction_ms'],
                              'question_prefill_ms': scenario['question_prefill_ms'],
                              'generation_ms': scenario['generation_ms'], 'total_ms': scenario['total_ms']})
    for name, rows in (('tree_stats.csv', stats_rows), ('cpu_timings.csv', time_rows), ('comparisons.csv', comparison_rows)):
        with (out / name).open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f'PASS: {len(cases)} documents, {len(comparison_rows)} exact comparisons, all five KV reductions', flush=True)
    if args.with_tests:
        env = os.environ.copy()
        env['EAT_MODEL'] = str(model)
        env['EAT_DRIVER'] = str(driver)
        env['EAT_SOURCE_PARSED_DIR'] = str(parsed_dir)
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(ROOT / 'tests'), '-v'],
                       check=True, cwd=ROOT, env=env)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'FAIL: {exc}', file=sys.stderr)
        raise SystemExit(1)
