#!/usr/bin/python3
"""Bounded stdlib checks that the public v1.0.0 package stays self-consistent.

No Docker, network or GPU access. Run: /usr/bin/python3 -I tests/public_release_contract_cpu.py
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load_launcher():
    spec = importlib.util.spec_from_file_location('public_launcher', ROOT / 'launcher.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PublicRelease(unittest.TestCase):
    def setUp(self):
        self.release = json.loads((ROOT / 'release.json').read_text())
        self.launcher = load_launcher()

    def test_launcher_accepts_default_release(self):
        obj, mode, _ = self.launcher.release()
        self.assertEqual(mode, 'mtp3-graph')
        self.assertEqual(obj['image'], self.launcher.IMAGE)
        self.assertEqual(obj['model'], self.launcher.MODEL)

    def test_image_contract_source_matches_pinned_sha(self):
        built = (ROOT / 'runtime-build/build/release.json').read_bytes()
        self.assertEqual(hashlib.sha256(built).hexdigest(), self.release['image_contract_sha256'])
        contract = json.loads(built)
        self.assertEqual(contract['model'], self.release['model'])
        validated = {k for k, v in self.release['modes'].items() if v.get('validated')}
        self.assertTrue(validated <= set(contract['supported_modes']))

    def test_image_is_digest_pinned(self):
        self.assertRegex(self.release['image_digest'], r'^sha256:[0-9a-f]{64}$')
        self.assertNotIn('latest', self.release['image'])

    def test_manual_env_file_matches_default_mode(self):
        lines = (ROOT / 'mtp3-graph.env').read_text().splitlines()
        env = dict(l.split('=', 1) for l in lines if l and not l.startswith('#'))
        self.assertEqual(env, self.release['modes']['mtp3-graph']['environment'])

    def test_download_defaults_match_release_model(self):
        text = (ROOT / 'download_model.sh').read_text()
        self.assertIn('MODEL_NAME:-' + self.release['model']['repo_id'] + '}', text)
        self.assertIn('MODEL_REVISION:-' + self.release['model']['revision'] + '}', text)
        code = self.launcher.DOWNLOAD_CODE
        self.assertIn('"%s", "%s"' % (self.release['model']['repo_id'], self.release['model']['revision']), code)

    def test_production_image_record_is_consistent(self):
        record = json.loads((ROOT / 'image/prod-dense-20260923/image.json').read_text())
        active = json.loads((ROOT / 'ACTIVE_PROFILE.json').read_text())
        public = active['distribution']['public_image']
        self.assertRegex(record['image_digest'], r'^sha256:[0-9a-f]{64}$')
        self.assertEqual(public['image_digest'], record['image_digest'])
        self.assertEqual(active['distribution']['public_release']['image_digest'], self.release['image_digest'])
        self.assertEqual(record['runtime_manifest_sha256'], active['manifest_sha256'])
        self.assertEqual(record['model']['revision'], active['model']['revision'])
        self.assertIn('qwen38-27b-vllm@' + record['image_digest'], (ROOT / 'README.md').read_text())

    def test_production_dockerfile_is_generated_from_pinned_profile(self):
        here = ROOT / 'image/prod-dense-20260923'
        committed = (here / 'Dockerfile').read_text()
        spec = importlib.util.spec_from_file_location('gen_dockerfile', here / 'gen_dockerfile.py')
        gen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen)
        with tempfile.TemporaryDirectory() as tmp:
            gen.HERE = Path(tmp)
            gen.PROFILE = ROOT / 'profiles/production-dense-recovery-20260923'
            gen.main()
            self.assertEqual((Path(tmp) / 'Dockerfile').read_text(), committed)
        argv = json.loads((gen.PROFILE / 'graph-prefix.json').read_text())['argv']
        self.assertIn('CMD ' + json.dumps(argv), committed)


if __name__ == '__main__':
    unittest.main(verbosity=2)
