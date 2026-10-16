"""Checks for Markdown section boundaries and safe A/B substitutions."""

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARSED_DIR = Path(os.environ.get('EAT_MARKDOWN_PARSED_DIR',
                                 ROOT / 'results' / 'markdown_domain' / 'parsed'))
sys.path.insert(0, str(ROOT / 'parsers'))
import markdown_document  # noqa: E402
from structured_common import validate_local  # noqa: E402


class MarkdownTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = json.loads((ROOT / 'samples' / 'markdown' / 'cases.json').read_text(encoding='utf-8'))
        cls.model = Path(os.environ.get('EAT_MODEL',
                                        ROOT / 'models' / 'Llama-3.2-1B-Instruct-Q4_K_M.gguf'))
        name = 'structured-attention-driver.exe' if os.name == 'nt' else 'structured-attention-driver'
        cls.driver = Path(os.environ.get('EAT_DRIVER', ROOT / 'build' / 'bin' / name))

    def test_requires_peer_sections(self):
        with self.assertRaisesRegex(ValueError, 'two peer sections'):
            markdown_document.build('# One title\n\n## Only section\n\nContent.\n',
                                    'one', self.driver, self.model)

    def test_duplicate_section_title_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unique'):
            markdown_document.build('# Title\n\n## Same\n\nA.\n\n## Same\n\nB.\n',
                                    'duplicate', self.driver, self.model)

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

    def test_nested_heading_list_and_code_have_parents(self):
        doc = json.loads((PARSED_DIR / 'release.json').read_text(encoding='utf-8'))
        nodes = {node['id']: node for node in doc['nodes']}
        types = {node['node_type'] for node in doc['nodes']}
        self.assertTrue({'heading', 'list', 'list_item', 'code'} <= types)
        for node in doc['nodes']:
            if node['node_type'] in ('heading', 'list', 'list_item', 'code'):
                self.assertIsNotNone(node['parent'])
                self.assertEqual(node['merged_into'], next(
                    section['id'] for section in doc['nodes']
                    if section['node_type'] == 'section' and node['path'].startswith(section['path'] + '/')))
                self.assertIn(node['parent'], nodes)

    def test_tiny_section_merges_into_shared_root(self):
        source = ('# Guide\n\n## Alpha\n\nAlpha owns the BLUE recovery target for the first '
                  'service. Keep it available during release.\n\n## Tiny\n\nOK.\n\n'
                  '## Beta\n\nBeta owns the GREEN recovery target for the second '
                  'service. Keep it available during release.\n')
        doc = markdown_document.build(source, 'merge', self.driver, self.model)
        nodes = {node['path']: node for node in doc['nodes']}
        self.assertEqual(nodes['/sections/Tiny']['merged_into'], nodes['']['id'])
        self.assertIn('## Tiny', doc['root_prefix'])
        self.assertEqual(len(doc['groups']), 2)

    def test_driver_rejects_changed_focus_section(self):
        case = self.cases[0]
        base = json.loads((PARSED_DIR / 'release.json').read_text(encoding='utf-8'))
        changed = json.loads((PARSED_DIR / 'release_modified.json').read_text(encoding='utf-8'))
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
