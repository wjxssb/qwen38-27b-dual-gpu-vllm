#!/usr/bin/python3
"""Regression and fault-injection checks; no CUDA import or GPU allocation."""
import ast
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import queue
import resource
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(os.environ.get('GRAPH006_TEST_ROOT',
    str(Path(__file__).resolve().parents[1] / 'runtime/graph006')))


def load(path):
    spec = importlib.util.spec_from_file_location('target_' + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TelemetryChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {'VLLM_STABILITY_TELEMETRY_DIR': self.tmp.name})
        self.env.start()
        self.m = load(ROOT / 'overlay/vllm/stability_telemetry.py')

    def tearDown(self):
        if self.m._writer_queue is not None:
            self.assertTrue(self.m.flush_for_test(10))
            self.m._writer_queue.put(self.m._STOP)
            self.m._writer_thread.join(5)
        self.env.stop()
        self.tmp.cleanup()

    def test_repeated_failure_windows_do_not_nest_or_grow(self):
        m = self.m
        m.configure_process('engine-core')
        for i in range(300):
            m.emit('progress', queues={'state': [1] * 32}, rpc_id=i)
            m.emit_failure_window('tp_lifecycle_watchdog', generation=1)
            if i % 10 == 0:
                self.assertTrue(m.flush_for_test())
        self.assertTrue(m.flush_for_test())
        count = 0
        for line in (Path(self.tmp.name) / 'engine-core.jsonl').read_bytes().splitlines():
            self.assertLessEqual(len(line) + 1, m._MAX_FAILURE_BYTES)
            record = json.loads(line)
            if record['event'] == 'tp_lifecycle_watchdog':
                count += 1
                for old in record['recent_event_window']:
                    self.assertNotIn('recent_event_window', old)
        self.assertGreater(count, 100)
        self.assertEqual(len(m._recent_events), 256)
        self.assertLess(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, 256 * 1024)

    def test_huge_cyclic_fields_and_mutating_caller_are_bounded(self):
        m = self.m
        value = {'state': [1, 2]}
        m.emit('before', queues=value)
        value['state'].append(3)
        snapshot = m.recent_event_window()
        self.assertEqual(snapshot[-1]['queues']['state'], [1, 2])
        snapshot[-1]['queues']['state'].append(4)
        self.assertEqual(m.recent_event_window()[-1]['queues']['state'], [1, 2])
        cycle = {}; cycle['self'] = cycle
        m.emit('huge', huge=['中' * 50000] * 10000, cycle=cycle,
               nested={'recent_event_window': snapshot})
        self.assertTrue(m.flush_for_test())
        for line in (Path(self.tmp.name) / 'unknown.jsonl').read_bytes().splitlines():
            self.assertLessEqual(len(line) + 1, m._MAX_RECORD_BYTES)
        self.assertEqual(m.recent_event_window('invalid'), [])

    def test_backpressure_has_byte_and_count_limits(self):
        m = self.m
        blocked = queue.Queue(maxsize=m._WRITER_QUEUE_SIZE)
        with patch.object(m, '_ensure_writer', return_value=blocked):
            for i in range(2000):
                m.emit('large', payload='x' * 1800, i=i)
        self.assertLessEqual(m._queued_bytes, m._MAX_QUEUED_BYTES)
        self.assertLessEqual(blocked.qsize(), m._WRITER_QUEUE_SIZE)
        self.assertGreater(m._dropped_records, 0)

    def test_rotation_bounds_disk(self):
        m = self.m
        m._MAX_FILE_BYTES = 8192
        for i in range(200):
            m.emit('rotate', payload='x' * 1024)
        self.assertTrue(m.flush_for_test())
        files = list(Path(self.tmp.name).iterdir())
        self.assertEqual(len(files), 2)
        self.assertTrue(all(p.stat().st_size <= 8192 for p in files))

    def test_disabled_telemetry_does_not_build_failure_window(self):
        with patch.dict(os.environ, {'VLLM_STABILITY_TELEMETRY_DIR': ''}), \
             patch.object(self.m, 'recent_event_window', side_effect=AssertionError):
            self.m.emit_failure_window('disabled')
        self.assertIsNone(self.m._writer_queue)


class RecoveryChecks(unittest.TestCase):
    def setUp(self):
        self.m = load(ROOT / 'launcher.py')
        self.profile = json.loads((ROOT / 'profile.json').read_text())
        self.tmp = tempfile.TemporaryDirectory()
        self.run = Path(self.tmp.name)
        (self.run / 'artifacts').mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def sample(self, journal='', log='', state=None, ready=True, failures=0):
        def command(args, **kw):
            return journal if args[0].endswith('journalctl') else log
        row = {'State': state or {'Running': True, 'OOMKilled': False}}
        with patch.object(self.m, 'command', side_effect=command), \
             patch.object(self.m, 'inspect_owned', return_value=row):
            return self.m.monitor_sample(self.profile, 'a' * 64, 'b' * 32,
                'c' * 64, self.run, {'kernel_faults': []}, time.monotonic(), ready, failures)

    def test_hardware_and_oom_are_not_restartable(self):
        self.assertEqual(self.sample(journal='NVRM: Xid 13')[:2], ('NEW_KERNEL_FAULT', True))
        self.assertEqual(self.sample(state={'Running': True, 'OOMKilled': True})[:2], ('CONTAINER_OOM', True))
        self.assertEqual(self.sample(log='CUDA error: an illegal memory access was encountered')[:2], ('CUDA_HARD_FAILURE', True))

    def test_poisoned_core_detected_even_if_health_would_return_200(self):
        (self.run / 'artifacts/engine-failure.json').write_text(json.dumps({
            'reason': 'TP_LIFECYCLE_FAIL_CLOSED', 'hard_stop': False}))
        with patch.object(self.m.urllib.request, 'urlopen', side_effect=AssertionError('must not reach health')):
            self.assertEqual(self.sample()[:2], ('TP_LIFECYCLE_FAIL_CLOSED', False))

    def test_misaligned_fault_is_hard_even_after_container_exit(self):
        logs = [
            'CUDA error: misaligned address',
            'CUDA_ERROR_MISALIGNED_ADDRESS',
            'cudaErrorMisalignedAddress',
            'Error: Failed to initialize the TMA descriptor 716',
            'CUDA out of memory. Tried to allocate 1 GiB',
        ]
        with patch.object(self.m.urllib.request, 'urlopen', side_effect=AssertionError):
            for log in logs:
                for running in (True, False):
                    with self.subTest(log=log, running=running):
                        self.assertEqual(self.sample(log=log, state={
                            'Running': running, 'OOMKilled': False})[:2],
                            ('CUDA_HARD_FAILURE', True))
        self.assertEqual(self.sample(log='ordinary worker exit', state={
            'Running': False, 'OOMKilled': False})[:2], ('CONTAINER_EXITED', False))

    def test_old_marker_cannot_downgrade_misaligned_failure(self):
        (self.run / 'artifacts/engine-failure.json').write_text(json.dumps({
            'reason': 'TP_LIFECYCLE_FAIL_CLOSED', 'hard_stop': False,
            'message': 'CUDA error: misaligned address'}))
        self.assertEqual(self.sample(state={'Running': False, 'OOMKilled': False})[:2],
                         ('TP_LIFECYCLE_FAIL_CLOSED', True))

    def test_health_failure_is_debounced_and_startup_has_grace(self):
        with patch.object(self.m.urllib.request, 'urlopen', side_effect=OSError('HTTP 503')):
            self.assertIsNone(self.sample(failures=1)[0])
            self.assertEqual(self.sample(failures=2)[0], 'HEALTH_FAILED')
            self.assertIsNone(self.sample(ready=False, failures=10)[0])

    def test_clean_health_sets_ready_without_inference(self):
        response = Mock(status=200)
        with patch.object(self.m.urllib.request, 'urlopen', return_value=contextlib.nullcontext(response)) as url:
            self.assertEqual(self.sample(ready=False), (None, False, True, 0))
        self.assertTrue(url.call_args.args[0].endswith('/health'))

    def test_bounded_retry_and_hard_stop(self):
        m = self.m
        with patch('sys.argv', ['launcher.py', '--start']), \
             patch.object(m.time, 'sleep'), \
             patch.object(m, 'launch', side_effect=[75, 75, 75]) as launch:
            with self.assertRaisesRegex(m.Refused, 'Two recovery attempts'):
                m.main()
            self.assertEqual(launch.call_count, 3)
        with patch('sys.argv', ['launcher.py', '--start']), \
             patch.object(m, 'launch', return_value=1) as launch:
            self.assertEqual(m.main(), 1)
            launch.assert_called_once()

    def test_mounts_and_in_container_integrity(self):
        m = self.m
        profile, env, argv, sha = m.configuration()
        cmd = m.create_command(profile, env, argv, self.run, 'test', sha)
        mounts = [cmd[i + 1] for i, x in enumerate(cmd) if x == '--mount']
        for item in profile['overlays']:
            self.assertIn('type=bind,src=' + str(ROOT / item['path']) + ',dst=' +
                item['target'] + ',bind-propagation=rprivate,readonly', mounts)
        self.assertEqual(cmd[cmd.index('--memory-swap') + 1], str(profile['memory_bytes']))
        entry = load(ROOT / 'runtime_entry.py')
        source = self.run / 'source.py'; source.write_text('original')
        items = [{'target': str(source), 'bytes': 8, 'sha256': hashlib.sha256(b'original').hexdigest()}]
        entry.verify(items)
        source.write_text('modified')
        with self.assertRaisesRegex(RuntimeError, 'hash mismatch'):
            entry.verify(items)

    def test_kernel_fault_override_no_longer_bypasses_gate(self):
        m = self.m
        with patch.dict(os.environ, {'ALLOW_KERNEL_FAULTS': '1'}), \
             patch.object(m, 'command', return_value='NVRM: Xid unexpected'):
            with self.assertRaises(m.Refused):
                m.startup_checks(self.profile)

    def test_fail_closed_raises_to_existing_engine_dead_handler(self):
        source = (ROOT / 'overlay/vllm/v1/engine/core.py').read_text()
        tree = ast.parse(source)
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                      and n.name == '_handle_tp_lifecycle_failure')
        ns = {'GenerationLifecycleError': RuntimeError, 'logger': Mock()}
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<handler>', 'exec'), ns)
        stopped = threading.Event()
        obj = types.SimpleNamespace(model_executor=types.SimpleNamespace(_stability_watchdog_stop=stopped))
        error = RuntimeError('RPC sample_tokens timed out')
        with patch.object(Path, 'open', side_effect=OSError('read-only')):
            with self.assertRaises(RuntimeError) as caught:
                ns['_handle_tp_lifecycle_failure'](obj, error)
        self.assertIs(caught.exception, error)
        self.assertTrue(stopped.is_set())
        # Existing outer handler sends EngineDead before shutdown can block.
        self.assertIn('engine_core._send_engine_dead()', source)


class ApiChecks(unittest.TestCase):
    def test_reasoning_only_and_http_error_bodies_fail(self):
        m = load(Path(__file__).resolve().parents[1] / 'runtime/graph006/api_smoke.py')
        for body in [
            {'error': {'message': 'EngineCore encountered an issue'}},
            {'choices': [{'message': {'content': None, 'reasoning': 'thinking'}, 'finish_reason': 'length'}]},
            {'choices': [{'message': {'content': 'partial'}, 'finish_reason': 'length'}]},
        ]:
            with self.assertRaises(ValueError):
                m.validate(body)
        self.assertEqual(m.validate({'choices': [{'message': {'content': '运行正常'},
                         'finish_reason': 'stop'}]}), '运行正常')


if __name__ == '__main__':
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    unittest.main(verbosity=2)
