"""Checks for Python tree boundaries and safe A/B substitutions."""

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARSED_DIR = Path(os.environ.get('EAT_SOURCE_PARSED_DIR',
                                 ROOT / 'results' / 'source_code_domain' / 'parsed'))
sys.path.insert(0, str(ROOT / 'parsers'))
import python_source  # noqa: E402
from structured_common import validate_local  # noqa: E402


class PythonSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = json.loads((ROOT / 'samples' / 'source_code' / 'cases.json').read_text(encoding='utf-8'))
        cls.model = Path(os.environ.get('EAT_MODEL',
                                        ROOT / 'models' / 'Llama-3.2-1B-Instruct-Q4_K_M.gguf'))
        name = 'structured-attention-driver.exe' if os.name == 'nt' else 'structured-attention-driver'
        cls.driver = Path(os.environ.get('EAT_DRIVER', ROOT / 'build' / 'bin' / name))

    def test_invalid_python_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'syntax error'):
            python_source.build(b'def broken(:\n    pass\n', 'broken', self.driver, self.model)

    def test_duplicate_definition_path_is_rejected(self):
        source = b'def same():\n    return 1\n\ndef same():\n    return 2\n'
        with self.assertRaisesRegex(ValueError, 'duplicate definition path'):
            python_source.build(source, 'duplicate', self.driver, self.model)

    def test_five_pairs_change_only_the_selected_group(self):
        for case in self.cases:
            with self.subTest(case=case['id']):
                base = json.loads((PARSED_DIR / f'{case["id"]}.json').read_text(encoding='utf-8'))
                changed = json.loads((PARSED_DIR / f'{case["id"]}_modified.json').read_text(encoding='utf-8'))
                validate_local(base)
                validate_local(changed)
                before = {node['path']: node for node in base['nodes']}
                after = {node['path']: node for node in changed['nodes']}
                self.assertEqual(before.keys(), after.keys())
                self.assertEqual(before['']['tokens'], after['']['tokens'])
                self.assertEqual([(g['path'], g['seq_id']) for g in base['groups']],
                                 [(g['path'], g['seq_id']) for g in changed['groups']])
                for path, node in before.items():
                    if path == case['unrelated']:
                        self.assertNotEqual(node['tokens'], after[path]['tokens'])
                    else:
                        self.assertEqual(node['tokens'], after[path]['tokens'])
                self.assertNotEqual(next(g['seq_id'] for g in base['groups'] if g['path'] == case['focus']),
                                    next(g['seq_id'] for g in base['groups'] if g['path'] == case['unrelated']))
                self.assertLess(base['stats']['model_token_count'], 2000)

    def test_class_method_has_class_parent(self):
        doc = json.loads((PARSED_DIR / 'accounts.json').read_text(encoding='utf-8'))
        nodes = {node['path']: node for node in doc['nodes']}
        self.assertEqual(nodes['/classes/Account']['node_type'], 'class')
        self.assertEqual(nodes['/classes/Account/methods/tier']['parent'], nodes['/classes/Account']['id'])
        self.assertEqual(nodes['/classes/Account/methods/tier']['node_type'], 'method')

    def test_tiny_function_merges_into_shared_root(self):
        source = (b'def tiny(): return 1\n\n'
                  b'def alpha(value):\n    """Classify the first independent value."""\n'
                  b'    if value > 10:\n        return "LARGE"\n    return "SMALL"\n\n'
                  b'def beta(value):\n    """Classify the second independent value."""\n'
                  b'    if value < 0:\n        return "NEGATIVE"\n    return "POSITIVE"\n')
        doc = python_source.build(source, 'merge_case', self.driver, self.model)
        paths = {group['path'] for group in doc['groups']}
        nodes = {node['path']: node for node in doc['nodes']}
        self.assertEqual(paths, {'/functions/alpha', '/functions/beta'})
        self.assertEqual(nodes['/functions/tiny']['merged_into'], nodes['']['id'])
        self.assertIn('def tiny(): return 1', doc['root_prefix'])

    def test_driver_rejects_an_extra_changed_function(self):
        case = self.cases[0]
        base = json.loads((PARSED_DIR / 'pricing.json').read_text(encoding='utf-8'))
        changed = json.loads((PARSED_DIR / 'pricing_modified.json').read_text(encoding='utf-8'))
        extra = copy.deepcopy(changed)
        focus = next(node for node in extra['nodes'] if node['path'] == case['focus'])
        focus['tokens'][0] += 1
        with tempfile.TemporaryDirectory() as scratch:
            base_path = Path(scratch) / 'base.json'
            bad_path = Path(scratch) / 'bad.json'
            output_path = Path(scratch) / 'result.json'
            base_path.write_text(json.dumps(base), encoding='utf-8')
            bad_path.write_text(json.dumps(extra), encoding='utf-8')
            run = subprocess.run([str(self.driver), 'experiment', str(self.model), str(base_path),
                                  str(bad_path), case['focus'], case['unrelated'], case['question'],
                                  '3', '4', str(output_path)], capture_output=True, text=True)
            self.assertEqual(run.returncode, 2)
            self.assertIn('focus subtree changed', run.stderr)
            self.assertFalse(output_path.exists())


if __name__ == '__main__':
    unittest.main()
