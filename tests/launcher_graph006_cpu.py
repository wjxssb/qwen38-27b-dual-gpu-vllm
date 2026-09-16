#!/usr/bin/python3
"""Bounded stdlib checks. All journal, NVML and Docker calls are fixtures.

The cross-process check uses only a unique temporary flock and its own child.
Run after the formal GPU window: /usr/bin/python3 -I tests/launcher_graph006_cpu.py
"""
import contextlib
import copy
import fcntl
import importlib.util
import io
import json
import multiprocessing
import os
from pathlib import Path
import resource
import signal
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

PACKAGE = Path(os.environ.get('GRAPH006_TEST_ROOT',
    str(Path(__file__).resolve().parents[1] / 'runtime/graph006')))
CID = 'a' * 64
RUN = 'b' * 32
MANIFEST = 'c' * 64


def load_launcher():
    spec = importlib.util.spec_from_file_location('launcher_graph006_test_target', PACKAGE / 'launcher.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def owned_row(module, profile, run_id, stopped=False):
    return {'Id': CID, 'Image': profile['image'], 'Name': '/dense-graph006-' + run_id,
            'Config': {'Image': profile['image'], 'Labels': module.labels(run_id, MANIFEST)},
            'State': {'Running': not stopped, 'Pid': 0 if stopped else 777,
                      'Status': 'exited' if stopped else 'running', 'ExitCode': 0, 'OOMKilled': False}}


def quarantine_child(root, profile, failed, stopped, errors):
    module = load_launcher()
    module.ROOT = Path(root)
    module.startup_checks = lambda _: {'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    selected = {}

    def fixture_command(argv, **kwargs):
        operation = argv[2]
        if operation == 'create':
            selected['run_id'] = argv[argv.index('--name') + 1].removeprefix('dense-graph006-')
            return CID + '\n'
        if operation == 'inspect':
            return json.dumps([owned_row(module, profile, selected['run_id'], stopped.is_set())])
        if operation == 'stop':
            failed.set()
            raise module.Refused('Injected initial cleanup error')
        if argv[2:4] == ['container', 'ls']:
            return CID + '\n'
        raise AssertionError('Unexpected control call: ' + repr(argv))

    class Attachment:
        def __init__(self, *args, **kwargs):
            pass

        def poll(self):
            return 0

        def wait(self, **kwargs):
            return 0

    module.command = fixture_command
    real_sleep = time.sleep
    module.time.sleep = lambda seconds: real_sleep(min(seconds, 0.02))
    module.subprocess.Popen = Attachment
    try:
        module.launch(profile, {}, ['/usr/bin/python3', '-m', 'fixture'], MANIFEST)
        errors.put('Unexpected successful launcher result')
    except module.Refused as exc:
        errors.put(str(exc))
    except BaseException as exc:
        errors.put(type(exc).__name__ + ': ' + str(exc))


class LauncherChecks(unittest.TestCase):
    def setUp(self):
        self.module = load_launcher()
        self.profile = json.loads((PACKAGE / 'profile.json').read_text())
        self.temporary = tempfile.TemporaryDirectory(prefix='graph006-cpu-')
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def fixture_startup(self, baseline_boot, journal):
        module = self.module
        module.ROOT = self.root
        profile = copy.deepcopy(self.profile)
        profile['model_mount']['source'] = str(self.root)
        (self.root / 'kernel-baseline.json').write_text(json.dumps({
            'boot_id': baseline_boot, 'acknowledged_kernel_faults': ['fixture NVRM: Xid old']}))
        calls = []

        def control(argv, **kwargs):
            calls.append(argv)
            if argv[0] == '/usr/bin/journalctl':
                return journal
            if argv[0] == '/usr/bin/nvidia-smi':
                if '--query-gpu=index,uuid,pci.bus_id' in argv:
                    return '\n'.join(f'{i}, {u}, 00000000:0{i + 1}:00.0' for i, u in enumerate(profile['gpu_uuids']))
                return ''
            if argv[2:4] == ['image', 'inspect']:
                return json.dumps([{'Id': profile['image']}])
            if argv[2] == 'ps' and argv[-1] == '--quiet':
                return ''
            raise AssertionError('Unexpected control call: ' + repr(argv))

        read_text = Path.read_text

        def proc_fixture(path, *args, **kwargs):
            if str(path) == '/proc/meminfo':
                return 'MemAvailable: 67108864 kB\n'
            return read_text(path, *args, **kwargs)

        with patch.object(module, 'command', side_effect=control), patch.object(Path, 'read_text', proc_fixture):
            result = module.startup_checks(profile)
        return result, calls

    def test_clean_new_boot_reidentifies_gpu_and_is_allowed(self):
        result, calls = self.fixture_startup('different-previous-boot', '')
        self.assertEqual(result['kernel_gate'], 'CLEAN_NEW_BOOT')
        self.assertEqual([g['index'] for g in result['gpus']], [0, 1])
        self.assertEqual({g['uuid'] for g in result['gpus']}, set(self.profile['gpu_uuids']))
        self.assertTrue(all(g['pci_bus_id'] for g in result['gpus']))
        self.assertTrue(any(c[2:4] == ['image', 'inspect'] for c in calls))

    def test_new_boot_with_any_fault_is_refused(self):
        with self.assertRaisesRegex(self.module.Refused, 'zero kernel fault'):
            self.fixture_startup('different-previous-boot', 'fixture NVRM: Xid new\n')

    def test_same_boot_retains_exact_multiset(self):
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        result, _ = self.fixture_startup(boot, 'fixture NVRM: Xid old\n')
        self.assertEqual(result['kernel_gate'], 'EXACT_ACKNOWLEDGED_CURRENT_BOOT')
        with self.assertRaisesRegex(self.module.Refused, 'rows differ'):
            self.fixture_startup(boot, 'fixture NVRM: Xid old\nfixture NVRM: Xid old\n')

    def test_print_command_has_no_subprocess_lock_or_pidfd_call(self):
        module = self.module
        output = io.StringIO()
        with patch.object(sys, 'argv', ['launcher.py', '--print-command']), \
             patch.object(module, 'command', side_effect=AssertionError('control operation forbidden')), \
             patch.object(module.subprocess, 'Popen', side_effect=AssertionError('Docker forbidden')), \
             patch.object(module.subprocess, 'run', side_effect=AssertionError('subprocess forbidden')), \
             patch.object(module.fcntl, 'flock', side_effect=AssertionError('lock forbidden')), \
             patch.object(module.os, 'pidfd_open', side_effect=AssertionError('pidfd forbidden')), \
             contextlib.redirect_stdout(output):
            self.assertEqual(module.main(), 0)
        self.assertIn('VLLM_SM120_MTP_DECODE_GRAPH=1', output.getvalue())
        self.assertIn('first pass remains eager', output.getvalue())

    def write_active(self, module, fd, start):
        active = {'run_id': RUN, 'pid': os.getpid(), 'start_ticks': start,
            'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'manifest_sha256': MANIFEST, 'container_id': CID, 'lock_fd': fd}
        directory = self.root / 'runs' / RUN
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'launcher.json').write_text(json.dumps(active))
        (self.root / 'active.json').write_text(json.dumps(active))

    def test_stop_rejects_reused_pid_and_foreign_program_without_signal(self):
        module = self.module
        module.ROOT = self.root
        lock = self.root / 'group.lock'
        lock.touch()
        profile = {**self.profile, 'gpu_group_lock': str(lock)}
        with lock.open('r+') as fd, patch.object(module.signal, 'pidfd_send_signal') as sent:
            fcntl.flock(fd, fcntl.LOCK_EX)
            self.write_active(module, fd.fileno(), module.proc_start(os.getpid()) + 1)
            with self.assertRaisesRegex(module.Refused, 'PID has been reused'):
                module.request_stop(profile, MANIFEST)
            self.write_active(module, fd.fileno(), module.proc_start(os.getpid()))
            with self.assertRaisesRegex(module.Refused, 'not this isolated launcher'):
                module.request_stop(profile, MANIFEST)
            sent.assert_not_called()

    def test_stop_rejects_foreign_cid_without_signal(self):
        module = self.module
        module.ROOT = self.root
        lock = self.root / 'group.lock'
        lock.touch()
        profile = {**self.profile, 'gpu_group_lock': str(lock)}
        read_bytes = Path.read_bytes

        def commandline(path):
            if str(path) == f'/proc/{os.getpid()}/cmdline':
                return b'\0'.join([b'/usr/bin/python3', b'-I', os.fsencode(self.root / 'launcher.py'), b''])
            return read_bytes(path)

        row = owned_row(module, profile, RUN)
        row['Config']['Labels'][module.LABEL_PREFIX + 'run'] = 'foreign'
        with lock.open('r+') as fd, patch.object(module.signal, 'pidfd_send_signal') as sent, \
             patch.object(Path, 'read_bytes', commandline), \
             patch.object(module, 'command', return_value=json.dumps([row])):
            fcntl.flock(fd, fcntl.LOCK_EX)
            self.write_active(module, fd.fileno(), module.proc_start(os.getpid()))
            with self.assertRaisesRegex(module.Refused, 'ownership changed'):
                module.request_stop(profile, MANIFEST)
            sent.assert_not_called()

    def test_cleanup_failure_retains_real_cross_process_flock_until_stopped(self):
        lock = self.root / 'group.lock'
        lock.touch()
        profile = {**self.profile, 'gpu_group_lock': str(lock)}
        context = multiprocessing.get_context('fork')
        failed, stopped = context.Event(), context.Event()
        errors = context.Queue()
        child = context.Process(target=quarantine_child, args=(str(self.root), profile, failed, stopped, errors))
        child.start()
        try:
            self.assertTrue(failed.wait(5), 'Fixture did not reach cleanup failure')
            with lock.open('r+') as probe:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.kill(child.pid, signal.SIGTERM)  # Only this test's captured child.
            time.sleep(0.1)
            self.assertTrue(child.is_alive(), 'TERM incorrectly escaped quarantine')
            with lock.open('r+') as probe:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            stopped.set()
            child.join(5)
            self.assertFalse(child.is_alive(), 'Stopped container did not release quarantine')
            self.assertEqual(child.exitcode, 0)
            self.assertIn('Initial cleanup failed', errors.get(timeout=2))
            with lock.open('r+') as probe:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            runs = list((self.root / 'runs').iterdir())
            self.assertEqual(len(runs), 1)
            receipt = json.loads((runs[0] / 'receipt.json').read_text())
            recovery = json.loads((runs[0] / 'quarantine-release.json').read_text())
            self.assertFalse(receipt['cleanup_verified'])
            self.assertFalse(recovery['cleanup_verified'])
            self.assertTrue(recovery['resource_release_verified'])
            self.assertFalse(recovery['container_absent'])
        finally:
            stopped.set()
            child.join(1)
            if child.is_alive():
                child.kill()  # Only the child created above, after a failed test.
                child.join(2)
            errors.close()


if __name__ == '__main__':
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    unittest.main(verbosity=2)
