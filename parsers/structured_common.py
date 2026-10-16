"""Shared tokenizer and structural checks for document-domain parsers."""

import json
import subprocess
import tempfile
from pathlib import Path

MIN_GROUP_TOKENS = 16
MAX_GROUPS = 256  # LLAMA_MAX_SEQ in the pinned llama.cpp checkout.


def call_tokenizer(driver: Path, model: Path, fragments):
    with tempfile.TemporaryDirectory() as scratch:
        request = Path(scratch) / 'request.json'
        output = Path(scratch) / 'tokens.json'
        request.write_text(json.dumps({'fragments': fragments}, ensure_ascii=False), encoding='utf-8')
        run = subprocess.run([str(driver), 'tokenize', str(model), str(request), str(output)],
                             capture_output=True, text=True, encoding='utf-8', errors='replace')
        if run.returncode:
            raise RuntimeError(f'tokenizer failed ({run.returncode}): {run.stderr[-2000:]}')
        return json.loads(output.read_text(encoding='utf-8'))


def validate_local(schema):
    nodes = schema['nodes']
    by_id = {node['id']: node for node in nodes}
    paths = {node['path'] for node in nodes}
    if len(by_id) != len(nodes):
        raise ValueError('duplicate node ID')
    if len(paths) != len(nodes):
        raise ValueError('duplicate node path')
    for node in nodes:
        n = len(node['tokens'])
        if not (n == len(node['positions']) == len(node['seq_ids'])):
            raise ValueError(f'unaligned token metadata at {node["path"]}')
        if node['parent'] is not None and node['parent'] not in by_id:
            raise ValueError(f'missing parent at {node["path"]}')
        if node['merged_into'] is not None and node['merged_into'] not in by_id:
            raise ValueError(f'missing merge target at {node["path"]}')
        seen = {node['id']}
        parent = node['parent']
        while parent is not None:
            if parent in seen:
                raise ValueError(f'parent cycle at {node["path"]}')
            seen.add(parent)
            parent = by_id[parent]['parent']
    if len(schema['groups']) < 2 or len(schema['groups']) > MAX_GROUPS:
        raise ValueError('sequence group count out of range')
