"""Run the original two-prompt CPU regression on any supported host."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / 'models' / 'Llama-3.2-1B-Instruct-Q4_K_M.gguf'


def default_driver():
    name = 'two-prompt.exe' if os.name == 'nt' else 'two-prompt'
    candidates = [ROOT / 'build' / 'bin' / name, ROOT / 'build' / 'bin' / 'Release' / name]
    return next((path for path in candidates if path.is_file()), candidates[0])


DEFAULT_DRIVER = default_driver()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--driver', type=Path, default=DEFAULT_DRIVER)
    parser.add_argument('--max-tokens', type=int, default=24)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results' / 'local_part1',
                        help='result folder; default is ignored by Git')
    args = parser.parse_args()
    model = args.model.expanduser().resolve()
    driver = args.driver.expanduser().resolve()
    if not model.is_file():
        parser.error(f'model missing: {model}')
    if not driver.is_file():
        parser.error(f'compiled two-prompt driver missing: {driver}')
    if args.max_tokens < 1 or args.threads < 1:
        parser.error('--max-tokens and --threads must be positive')
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    for label, extra in [('batched_run', []), ('batched_reverse', ['--reverse']), ('batched_repeat', [])]:
        log = output / f'{label}.log'
        command = [str(driver), str(model), str(args.max_tokens), str(args.threads), *extra]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        log.write_bytes(result.stdout)
        if result.returncode > 1:
            raise RuntimeError(f'{label} runtime failed ({result.returncode}); see {log}')
        print(f'{label}: completed; see {log}', flush=True)
    subprocess.run([sys.executable, str(ROOT / 'scripts' / 'compare.py'), str(output)], check=True)


if __name__ == '__main__':
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f'FAIL: {exc}', file=sys.stderr)
        raise SystemExit(1)
