#!/usr/bin/env python3
"""Portable foreground Docker launcher. No host Python ML packages are needed."""
import argparse
import contextlib
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parent
IMAGE = 'ghcr.io/wjxssb/qwen38-27b-vllm:sm120-nvfp4-k3'
MODEL = {'repo_id': 'unsloth/Qwen3.8-27B-NVFP4',
         'revision': '57926baca9a82b4d6906b43f2750d55315f5b10f'}
LABEL = 'io.github.wjxssb.qwen38.'
# Independent of checkout location, XDG variables, TMPDIR, and selected order.
GPU_LOCK_DIR = Path('/tmp/qwen38-runtime-gpu-locks-v1')
STOP = False


class Refused(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise Refused(message)


def read_json(path):
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as f:
        s = os.fstat(f.fileno())
        require(stat.S_ISREG(s.st_mode) and s.st_size <= 1024**2, 'Invalid JSON file: ' + str(path))
        return json.loads(f.read())


def save(path, value):
    """Exclusive, atomic publication within the owned state directory."""
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with tmp.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def release(mode=None):
    obj = read_json(ROOT / 'release.json')
    require(obj.get('schema') == 1 and obj.get('image') == IMAGE and obj.get('model') == MODEL,
            'Unknown release contract, image, or model revision')
    require(obj.get('status') == 'RELEASED',
            'This package is a draft; its public image/profile has not been GPU validated')
    require(re.fullmatch(r'sha256:[0-9a-f]{64}', obj.get('image_digest') or ''), 'Release image digest is missing')
    require(re.fullmatch(r'[0-9a-f]{64}', obj.get('image_contract_sha256') or ''), 'Image contract SHA is missing')
    require(obj.get('image_contract_path') == '/opt/qwen38-runtime/release.json', 'Unexpected image contract path')
    selected = mode or obj['default_mode']
    require(selected in obj['modes'], 'Unknown mode; no automatic fallback is supported')
    profile = obj['modes'][selected]
    require(profile.get('validated') is True and bool(profile.get('validation_reference')),
            'This exact mode has no recorded GPU validation')
    req = obj['requirements']
    require(req.get('compute_capability') == '12.0' and req.get('gpu_count') == 2, 'Only SM120 TP2 is supported')
    require(re.fullmatch(r'\d+(?:\.\d+)+', req.get('min_driver_version') or ''), 'Minimum tested driver is missing')
    for key in ('min_gpu_memory_bytes', 'min_available_gpu_bytes', 'min_available_host_bytes'):
        require(type(req.get(key)) is int and req[key] > 0, 'Missing physical capacity requirement: ' + key)
    for key in ('cpus', 'host_memory_bytes', 'pinned_memory_bytes', 'shm_bytes', 'startup_timeout_s'):
        require(type(profile.get(key)) is int and profile[key] > 0, 'Missing mode bound: ' + key)
    args = profile['argv']
    require(args[:4] == ['/usr/bin/python3', '-m', 'vllm.entrypoints.openai.api_server', '--model'], 'Unexpected entrypoint')
    require(args[4] == model_path(), 'Model path must name the fixed downloaded revision')
    require(args[args.index('--tensor-parallel-size') + 1] == '2', 'TP2 is required')
    require(args[args.index('--port') + 1] == '8000', 'Internal port must be 8000')
    return obj, selected, profile


def model_path():
    return '/models/hub/models--' + MODEL['repo_id'].replace('/', '--') + '/snapshots/' + MODEL['revision']


def command(args, timeout=30, limit=1024**2):
    with tempfile.TemporaryFile() as out:
        result = subprocess.run(args, stdout=out, stderr=out, timeout=timeout, check=False)
        require(out.tell() <= limit, 'Control output exceeded its bound')
        out.seek(0)
        value = out.read().decode('utf-8', 'replace')
    require(result.returncode == 0, f'{args[0]} failed ({result.returncode}): {value[-1500:]}')
    return value


def docker(*args, timeout=30):
    # A remote Docker daemon would see different paths and GPUs. Do not use it.
    return command(['docker', '--host=unix:///var/run/docker.sock', *args], timeout=timeout)


def version(value):
    require(re.fullmatch(r'\d+(?:\.\d+)+', value) is not None, 'Invalid driver version')
    return tuple(int(x) for x in value.split('.'))


def select_gpus(csv_text, requirements, requested=None):
    rows = []
    for row in csv.reader(csv_text.splitlines()):
        require(len(row) == 7, 'GPU discovery returned an unsupported schema')
        index, uid, pci, cc, total, free, driver = (x.strip() for x in row)
        require(re.fullmatch(r'GPU-[0-9a-fA-F-]{36}', uid) is not None, 'GPU UUID is invalid')
        rows.append({'index': int(index), 'uuid': uid, 'pci': pci, 'cc': cc,
                     'total': int(total) * 1024**2, 'free': int(free) * 1024**2, 'driver': driver})
    require(len({r['uuid'] for r in rows}) == len(rows), 'Duplicate GPU UUIDs')
    if requested:
        ids = requested.split(',')
        require(len(ids) == 2 and len(set(ids)) == 2, '--gpus needs two distinct complete UUIDs')
        selected = [r for r in rows if r['uuid'] in ids]
    else:
        selected = [r for r in rows if r['cc'] == '12.0']
    require(len(selected) == 2, 'Need exactly two SM120 GPUs; with more GPUs specify --gpus UUID,UUID')
    for row in selected:
        require(row['cc'] == '12.0', 'Selected GPU is not SM120')
        require(row['total'] >= requirements['min_gpu_memory_bytes'], 'Selected GPU has insufficient capacity')
        require(row['free'] >= requirements['min_available_gpu_bytes'], 'Selected GPU has insufficient currently free memory')
        require(version(row['driver']) >= version(requirements['min_driver_version']), 'NVIDIA driver is older than the tested minimum')
    return sorted(selected, key=lambda r: (r['pci'], r['uuid']))


def discover(req, requested):
    return select_gpus(command(['nvidia-smi', '--query-gpu=index,uuid,pci.bus_id,compute_cap,memory.total,memory.free,driver_version',
                               '--format=csv,noheader,nounits']), req, requested)


@contextlib.contextmanager
def locks(uuids, directory=GPU_LOCK_DIR):
    # Cooperative inter-process locks, never unlinked, including after release.
    # This does not reserve GPUs against applications that do not use this protocol.
    try:
        directory.mkdir(mode=0o1777)
        directory.chmod(0o1777)
    except FileExistsError:
        pass
    require(not directory.is_symlink() and directory.is_dir(), 'GPU lock directory is not a real directory')
    handles = []
    try:
        for uid in sorted(uuids):
            require(re.fullmatch(r'GPU-[0-9a-fA-F-]{36}', uid) is not None, 'Invalid lock UUID')
            path = directory / (uid + '.lock')
            previous = os.umask(0)
            try:
                fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o666)
            finally:
                os.umask(previous)
            handles.append(fd)
            require(stat.S_ISREG(os.fstat(fd).st_mode), 'GPU lock is not a regular file')
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise Refused('Selected GPU is locked by another launcher; nothing was stopped') from exc
        yield
    finally:
        for fd in reversed(handles):
            os.close(fd)


def transient(image_id, flags, code, data, timeout):
    """CPU-only control container, including owned cleanup when a client times out."""
    run_id = uuid.uuid4().hex
    run_dir = data / 'control-runs' / run_id
    run_dir.mkdir(parents=True, mode=0o700)
    active = {'run_id': run_id, 'image_id': image_id, 'gpu_uuids': [],
              'release_sha256': hashlib.sha256((ROOT / 'release.json').read_bytes()).hexdigest()}
    labels = []
    for key, value in own_labels(run_id, active['release_sha256'], []).items():
        labels += ['--label', key + '=' + value]
    cid = None
    try:
        cid = docker('create', '--name', 'qwen38-' + run_id, '--pull=never', '--restart=no',
                     '--cap-drop=ALL', '--security-opt=no-new-privileges', '--pids-limit=256',
                     '--env=NVIDIA_VISIBLE_DEVICES=void', *labels, *flags,
                     '--entrypoint=/usr/bin/python3', image_id, '-I', '-c', code).strip()
        inspect_owned(cid, active)
        value = docker('start', '--attach', cid, timeout=timeout)
        final = inspect_owned(cid, active)['State']
        require(final['Running'] is False and final['ExitCode'] == 0 and not final['OOMKilled'], 'CPU control/download container failed')
        return value
    finally:
        if cid is None:
            found = docker('ps', '-aq', '--no-trunc', '--filter', 'name=^/qwen38-' + run_id + '$').splitlines()
            require(len(found) <= 1, 'Ambiguous control container create')
            cid = found[0] if found else None
        if cid:
            cleanup(cid, active, run_dir)


def pull_and_verify(obj, data):
    ref = obj['image'] + '@' + obj['image_digest']
    print('Pulling pinned public runtime: ' + ref, flush=True)
    docker('pull', '--quiet', ref, timeout=3600)
    rows = json.loads(docker('image', 'inspect', ref))
    require(len(rows) == 1 and re.fullmatch(r'sha256:[0-9a-f]{64}', rows[0]['Id']), 'Cannot resolve immutable local image ID')
    require(rows[0].get('Os') == 'linux' and rows[0].get('Architecture') == 'amd64', 'Image must be linux/amd64')
    require(not rows[0].get('Config', {}).get('Volumes'), 'Release image must not declare anonymous volumes')
    image_id = rows[0]['Id']
    probe = "import pathlib,sys;sys.stdout.buffer.write(pathlib.Path('/opt/qwen38-runtime/release.json').read_bytes())"
    raw = transient(image_id, ['--network=none', '--read-only', '--memory=256m', '--memory-swap=256m', '--cpus=1'], probe, data, 30)
    require(hashlib.sha256(raw.encode()).hexdigest() == obj['image_contract_sha256'], 'Image runtime contract does not match this release')
    contract = json.loads(raw)
    require(contract.get('schema') == 1 and contract.get('runtime_abi') == 'qwen38-sm120-v1'
            and contract.get('model') == MODEL, 'Image ABI/model contract is incompatible')
    require(all(k in contract.get('supported_modes', []) for k, v in obj['modes'].items() if v.get('validated')), 'Image lacks a validated package mode')
    return image_id


DOWNLOAD_CODE = r'''
import json, os
from pathlib import Path
from huggingface_hub import snapshot_download
repo, rev = "unsloth/Qwen3.8-27B-NVFP4", "57926baca9a82b4d6906b43f2750d55315f5b10f"
p = Path(snapshot_download(repo_id=repo, revision=rev, cache_dir="/models/hub", token=False, max_workers=1))
assert p.name == rev and p.parent.name == "snapshots"
assert (p / "config.json").is_file() and (p / "tokenizer_config.json").is_file()
index = json.loads((p / "model.safetensors.index.json").read_text())
files = set(index["weight_map"].values())
assert files
for name in files:
    assert Path(name).name == name and name.endswith(".safetensors")
    shard = p / name
    assert shard.is_file() and shard.stat().st_size > 8
    assert shard.resolve().is_relative_to(Path("/models/hub").resolve())
marker = Path("/models/download-complete.json")
temporary = marker.with_suffix(".tmp")
temporary.write_text(json.dumps({"repo_id":repo,"revision":rev,"snapshot":str(p),"shards":len(files)}) + "\n")
os.replace(temporary, marker)
print("Fixed model snapshot download completed: " + str(p))
'''


def download(image_id, data):
    model = data / 'models'
    model.mkdir(mode=0o700, exist_ok=True)
    require(',' not in str(model), 'Docker bind paths cannot contain a comma')
    # Shared per-user download lock; no GPU is requested by this container.
    fd = os.open(data / 'download.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        # Retained partial Hub files can resume; a failed command never reaches server create.
        transient(image_id, ['--user', f'{os.getuid()}:{os.getgid()}', '--memory=2g', '--memory-swap=2g', '--cpus=2',
                  '--env=HF_HOME=/models', '--env=HOME=/models', '--env=HF_HUB_DISABLE_PROGRESS_BARS=1',
                  '--mount', f'type=bind,src={model},dst=/models'], DOWNLOAD_CODE, data, 4 * 3600)
        marker = read_json(model / 'download-complete.json')
        require(marker.get('repo_id') == MODEL['repo_id'] and marker.get('revision') == MODEL['revision']
                and marker.get('snapshot') == model_path(), 'Model download completion identity differs')
    finally:
        os.close(fd)


def own_labels(run_id, digest, ids):
    return {LABEL + 'app': 'dense-public-runtime-v1', LABEL + 'run': run_id,
            LABEL + 'uid': str(os.getuid()), LABEL + 'release': digest, LABEL + 'gpus': ','.join(sorted(ids))}


def inspect_owned(cid, active):
    require(re.fullmatch(r'[0-9a-f]{64}', cid) is not None, 'Invalid container ID')
    rows = json.loads(docker('inspect', cid))
    require(len(rows) == 1, 'Container inspection is ambiguous')
    row = rows[0]
    expected = own_labels(active['run_id'], active['release_sha256'], active['gpu_uuids'])
    require(row.get('Id') == cid and row.get('Image') == active['image_id']
            and row.get('Name') == '/qwen38-' + active['run_id']
            and all((row['Config'].get('Labels') or {}).get(k) == v for k, v in expected.items()),
            'Container ownership differs; it will not be signalled or removed')
    return row


def server_command(obj, mode, profile, image_id, ids, run_dir, data, port, run_id, release_sha):
    args = ['create', '--name', 'qwen38-' + run_id, '--pull=never', '--restart=no', '--init',
            '--network=bridge', '--ipc=private', '--read-only', '--cap-drop=ALL', '--security-opt=no-new-privileges',
            '--user', f'{os.getuid()}:{os.getgid()}', '--cpus', str(profile['cpus']),
            '--memory', str(profile['host_memory_bytes']), '--memory-swap', str(profile['host_memory_bytes']),
            '--shm-size', str(profile['shm_bytes']), '--ulimit', 'memlock=' + str(profile['pinned_memory_bytes']),
            '--pids-limit=1024', '--stop-timeout=30', '--tmpfs=/tmp:rw,nosuid,nodev,size=268435456',
            '--gpus', '"device=' + ','.join(ids) + '"', '--publish', f'127.0.0.1:{port}:8000',
            '--log-driver=json-file', '--log-opt=max-size=20m', '--log-opt=max-file=5', '--workdir=/results']
    for key, value in own_labels(run_id, release_sha, ids).items():
        args += ['--label', key + '=' + value]
    for path, target, ro in ((data / 'models', '/models', True),
                             (run_dir / 'cache', '/cache', False), (run_dir / 'artifacts', '/results', False)):
        require(',' not in str(path), 'Docker bind paths cannot contain a comma')
        args += ['--mount', f'type=bind,src={path},dst={target}' + (',readonly' if ro else '')]
    env = {**profile['environment'], 'HOME': '/cache/home', 'XDG_CACHE_HOME': '/cache/xdg',
           'HF_HOME': '/cache/huggingface', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
           'CUDA_CACHE_PATH': '/cache/cuda', 'TRITON_CACHE_DIR': '/cache/triton',
           'TORCH_EXTENSIONS_DIR': '/cache/torch', 'TMPDIR': '/cache',
           'NVIDIA_VISIBLE_DEVICES': ','.join(ids), 'NVIDIA_DRIVER_CAPABILITIES': 'compute,utility'}
    for key, value in sorted(env.items()):
        require(type(value) is str, 'Environment values must be strings')
        args += ['--env', key + '=' + value]
    return args + ['--entrypoint', profile['argv'][0], image_id] + profile['argv'][1:]


def cleanup(cid, active, run_dir):
    row = inspect_owned(cid, active)
    if row['State']['Running']:
        docker('stop', '--time=30', cid, timeout=45)
    row = inspect_owned(cid, active)
    require(row['State']['Running'] is False and row['State']['Pid'] == 0, 'Container has not stopped')
    save(run_dir / 'container-final.json', row)
    docker('rm', cid)
    require(cid not in docker('ps', '-aq', '--no-trunc').splitlines(), 'Container removal is not confirmed')
    return row['State']


def proc_start(pid):
    value = Path(f'/proc/{pid}/stat').read_text()
    fields = value[value.rfind(')') + 2:].split()
    require(fields[0] != 'Z', 'Launcher is a zombie')
    return int(fields[19])


def stop(data):
    active = read_json(data / 'active.json')
    require(active['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip(), 'Active record is from another boot')
    require(type(active['pid']) is int and active['pid'] > 1, 'Invalid active process')
    # pidfd prevents a PID-reuse race between verification and signalling.
    handle = os.pidfd_open(active['pid'])
    try:
        require(proc_start(active['pid']) == active['start_ticks'], 'Launcher PID has been reused')
        require(Path(f'/proc/{active["pid"]}').stat().st_uid == os.getuid(), 'Launcher belongs to another user')
        cmdline = Path(f'/proc/{active["pid"]}/cmdline').read_bytes().split(b'\0')
        require(len(cmdline) >= 3 and cmdline[1:3] == [b'-I', os.fsencode(active['launcher_path'])], 'Active process is not the recorded launcher')
        require(read_json(data / 'runs' / active['run_id'] / 'launcher.json') == active, 'Active record differs from run record')
        inspect_owned(active['container_id'], active)
        signal.pidfd_send_signal(handle, signal.SIGTERM)
    finally:
        os.close(handle)
    print('Stop requested for this owned run; the foreground launcher will confirm cleanup.')


def set_stop(signum, frame):
    global STOP
    STOP = True


def run_server(obj, mode, profile, image_id, ids, data, port, requested):
    global STOP
    with locks(ids):
        require(not (data / 'active.json').exists(), 'An active or unresolved run already exists; inspect its receipt')
        fresh = discover(obj['requirements'], requested)
        require([r['uuid'] for r in fresh] == ids, 'GPU identity/order changed after acquiring locks')
        apps = csv.reader(command(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits']).splitlines())
        require(not any(row and row[0].strip() in ids for row in apps), 'Selected GPU already has a compute process')
        mem = {x.split(':')[0]: int(x.split()[1]) * 1024 for x in Path('/proc/meminfo').read_text().splitlines()}
        require(mem['MemAvailable'] >= obj['requirements']['min_available_host_bytes'], 'Insufficient available host RAM')
        run_id = uuid.uuid4().hex
        run_dir = data / 'runs' / run_id
        run_dir.mkdir(parents=True, mode=0o700)
        for name in ('cache', 'artifacts'):
            (run_dir / name).mkdir(mode=0o700)
        release_sha = hashlib.sha256((ROOT / 'release.json').read_bytes()).hexdigest()
        active = {'run_id': run_id, 'release_sha256': release_sha, 'image_id': image_id,
                  'gpu_uuids': ids, 'pid': os.getpid(), 'start_ticks': proc_start(os.getpid()),
                  'launcher_path': str(ROOT / 'launcher.py'),
                  'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
        receipt = {'run_id': run_id, 'mode': mode, 'cleanup_verified': False, 'ready': False}
        args = server_command(obj, mode, profile, image_id, ids, run_dir, data, port, run_id, release_sha)
        save(run_dir / 'invocation.json', {'docker_argv': args, 'gpus': fresh, 'release': obj})
        cid = None
        attachment = None
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, set_stop)
        try:
            cid = docker(*args).strip()
            active['container_id'] = cid
            inspect_owned(cid, active)
            save(run_dir / 'launcher.json', active)
            save(data / 'active.json', active)
            print(f'Foreground server; local API http://127.0.0.1:{port}/v1 ; evidence {run_dir}', flush=True)
            if not STOP:
                attachment = subprocess.Popen(['docker', '--host=unix:///var/run/docker.sock', 'start', '--attach', cid])
            deadline = time.monotonic() + profile['startup_timeout_s']
            while attachment is not None and attachment.poll() is None and not STOP:
                if not receipt['ready']:
                    require(time.monotonic() < deadline, 'API startup timed out; no automatic retry or fallback')
                    try:
                        with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=1) as response:
                            receipt['ready'] = response.status == 200
                    except OSError:
                        pass
                    if receipt['ready']:
                        print('API is ready. Ctrl-C or ./launcher.py --stop stops this run.', flush=True)
                time.sleep(0.5)
        finally:
            try:
                # Docker create can succeed before its client fails; recover only our exact unique name.
                if not cid:
                    found = docker('ps', '-aq', '--no-trunc', '--filter', 'name=^/qwen38-' + run_id + '$').splitlines()
                    require(len(found) <= 1, 'Ambiguous create outcome')
                    cid = found[0] if found else None
                final = cleanup(cid, active, run_dir) if cid else None
                if attachment is not None:
                    attachment.wait(timeout=5)
                receipt.update(cleanup_verified=True, container_id=cid, container_state=final)
                if (data / 'active.json').exists() and read_json(data / 'active.json') == active:
                    (data / 'active.json').unlink()
            except Exception as exc:
                receipt['cleanup_error'] = str(exc)
                save(run_dir / 'receipt.json', receipt)
                # Retain both GPU locks and the failed receipt. Read-only observations only.
                print('Cleanup unresolved; GPU locks retained. Inspect exact owned CID: ' + str(cid), file=sys.stderr, flush=True)
                while True:
                    try:
                        if cid and cid not in docker('ps', '-aq', '--no-trunc').splitlines():
                            break
                    except Exception:
                        pass
                    time.sleep(5)
                raise Refused('Cleanup failed; external removal later observed. Original FAIL is retained') from exc
            save(run_dir / 'receipt.json', receipt)
        state = receipt.get('container_state') or {}
        return 0 if STOP and not state.get('OOMKilled') else (0 if state.get('ExitCode') == 0 else 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--stop', action='store_true')
    action.add_argument('--download-only', action='store_true')
    action.add_argument('--plan', action='store_true')
    parser.add_argument('--mode', help='Only explicitly validated release modes are accepted')
    parser.add_argument('--gpus', help='Two complete GPU UUIDs; omitted when exactly two SM120 GPUs exist')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--data-dir', type=Path, default=Path(os.environ.get('XDG_DATA_HOME', str(Path.home() / '.local/share'))) / 'qwen38-runtime')
    args = parser.parse_args()
    require(1024 <= args.port <= 65535, 'Host port must be between 1024 and 65535')
    if args.plan:
        print(json.dumps(read_json(ROOT / 'release.json'), indent=2))
        return 0
    require(platform.system() == 'Linux' and platform.machine() == 'x86_64', 'This release supports Linux x86_64 only')
    require(os.getuid() != 0, 'Run as an ordinary user with Docker socket access')
    require(shutil.which('docker') is not None, 'Install Docker Engine and configure access for your user')
    data = args.data_dir.expanduser().absolute()
    data.mkdir(parents=True, mode=0o700, exist_ok=True)
    require(data.resolve() == data and data.stat().st_uid == os.getuid(), 'Data directory must be a real directory owned by the current user')
    if args.stop:
        return stop(data) or 0  # Stop remains available even after a package update.
    obj, mode, profile = release(args.mode)
    require(shutil.which('nvidia-smi') is not None or args.download_only, 'NVIDIA driver/nvidia-smi is missing')
    # No GPU lock or device is acquired while downloading.
    selected = [] if args.download_only else discover(obj['requirements'], args.gpus)
    image_id = pull_and_verify(obj, data)
    download(image_id, data)
    if args.download_only:
        return 0
    return run_server(obj, mode, profile, image_id, [r['uuid'] for r in selected], data, args.port, args.gpus)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (Refused, OSError, ValueError, KeyError, IndexError, subprocess.SubprocessError) as exc:
        print('REFUSED: ' + str(exc), file=sys.stderr)
        sys.exit(2)
