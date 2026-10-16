"""Prepare the pinned llama.cpp checkout and build the CPU-only drivers."""

import argparse
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LLAMA = ROOT / 'llama.cpp'
COMMIT = '8bb3cf0d216a5ae48360179f2931daf284be4d01'
REMOTE = 'https://github.com/ggml-org/llama.cpp.git'


def run(*args, **kwargs):
    print('+', subprocess.list2cmdline([str(arg) for arg in args]), flush=True)
    return subprocess.run([str(arg) for arg in args], check=True, **kwargs)


def apply_patch(path):
    command = ['git', '-C', LLAMA, 'apply']
    if subprocess.run([*map(str, command), '--reverse', '--check', str(path)],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        print(f'already applied: {path.name}', flush=True)
        return
    run(*command, '--check', path)
    run(*command, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler', choices=('auto', 'gcc', 'system'), default='auto',
                        help='auto uses GCC on Windows if available; system uses the active toolchain')
    parser.add_argument('--generator', help='CMake generator; defaults to Ninja when available')
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    for tool in ('git', 'cmake'):
        if shutil.which(tool) is None:
            parser.error(f'{tool} is required on PATH')

    if not (LLAMA / '.git').exists():
        if LLAMA.exists():
            parser.error(f'{LLAMA} exists but is not a Git checkout')
        run('git', 'clone', REMOTE, LLAMA)
        run('git', '-C', LLAMA, 'checkout', COMMIT)
    head = subprocess.check_output(['git', '-C', str(LLAMA), 'rev-parse', 'HEAD'], text=True).strip()
    if head != COMMIT:
        parser.error(f'llama.cpp must be at {COMMIT}; found {head}')

    apply_patch(ROOT / 'scripts' / 'llama_targets.patch')
    if os.name == 'nt':
        apply_patch(ROOT / 'scripts' / 'llama_windows.patch')

    generator = args.generator or ('Ninja' if shutil.which('ninja') else None)
    compiler = args.compiler
    if compiler == 'auto':
        compiler = 'gcc' if os.name == 'nt' and shutil.which('gcc') and shutil.which('g++') else 'system'
    if compiler == 'gcc' and not (shutil.which('gcc') and shutil.which('g++')):
        parser.error('GCC and G++ are required for --compiler gcc')
    if os.name == 'nt' and compiler == 'gcc' and generator != 'Ninja':
        parser.error('the Windows GCC build requires Ninja; install it or use --compiler system')

    command = ['cmake', '-S', LLAMA, '-B', ROOT / 'build']
    if generator:
        command.extend(['-G', generator])
    command.extend(['-DCMAKE_BUILD_TYPE=Release', '-DGGML_CUDA=OFF', '-DGGML_VULKAN=OFF',
                    '-DGGML_METAL=OFF', '-DLLAMA_BUILD_EXAMPLES=ON', '-DLLAMA_CURL=OFF'])
    if compiler == 'gcc':
        command.extend(['-DCMAKE_C_COMPILER=gcc', '-DCMAKE_CXX_COMPILER=g++'])
        if os.name == 'nt':
            command.extend(['-DCMAKE_C_FLAGS=-D_WIN32_WINNT=0x0A00',
                            '-DCMAKE_CXX_FLAGS=-D_WIN32_WINNT=0x0A00'])
    run(*command)
    run('cmake', '--build', ROOT / 'build', '--config', 'Release', '--target', 'llama-batched',
        'two-prompt', 'structured-attention-driver', '-j', args.jobs)


if __name__ == '__main__':
    main()
