# AeroForge-TRIAD Colab receiver for a pod export served by scripts/runpod/serve_export.py (HTTPS via the RunPod proxy).
# Polls BASE/SHA256SUMS until it appears, downloads each listed file with resume (curl -C -), verifies sha256, installs
# into Drive WITHOUT overwriting (same rules as colab_v2_watcher.py), copies the raw archive, then sends the ack
# (BASE/ack?sha=<sha of SHA256SUMS>) so the pod may stop. Status in _staging/v2_<TAG>_watcher_status.txt.
# Usage: python colab_v2_http_watcher.py BASE_URL TAG     (BASE_URL = https://<podid>-8888.proxy.runpod.net/<TOKEN>)
import datetime, hashlib, json, os, shutil, subprocess, sys, tarfile, time
from pathlib import Path

ROOT = Path(os.environ.get('AF_ROOT', '.'))
BASE, TAG = sys.argv[1].rstrip('/'), sys.argv[2]
X = Path(os.environ.get('AF_XFER', f'/content/xfer_v2_{TAG}'))
LOG = ROOT / 'logs' / f'transfer_v2_{TAG}.log'
STATUS = ROOT / '_staging' / f'v2_{TAG}_watcher_status.txt'
CODE_PREFIXES = ('scripts/', 'src/', 'research/')


def now(): return datetime.datetime.now(datetime.timezone.utc).strftime('%FT%TZ')
def say(m):
    line = f'{now()} {m}'; print(line, flush=True); LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, 'a') as fh: fh.write(line + '\n')
def status(m): STATUS.parent.mkdir(parents=True, exist_ok=True); STATUS.write_text(f'{now()} {m}\n')
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for c in iter(lambda: fh.read(1 << 22), b''): h.update(c)
    return h.hexdigest()
def curl(name, dst, extra=()):
    return subprocess.run(['curl', '-sS', '-f', '--retry', '5', '--retry-delay', '10', *extra, '-o', str(dst), f'{BASE}/{name}'],
                          capture_output=True, text=True, timeout=7200).returncode == 0


def main():
    X.mkdir(parents=True, exist_ok=True)
    say('H_START waiting for the pod export (HTTP)')
    deadline, n = time.time() + 24 * 3600, 0
    while time.time() < deadline:
        n += 1; status(f'WAITING attempt {n}')
        if curl('SHA256SUMS', X / 'SHA256SUMS'):
            break
        time.sleep(120)
    else:
        say('H_GAVE_UP'); status('GAVE_UP'); return 1
    sums = dict(reversed(l.split()) for l in (X / 'SHA256SUMS').read_text().splitlines() if l.strip())
    for name, h in sums.items():
        for attempt in range(1, 6):
            curl(name, X / name, ('-C', '-'))
            if (X / name).exists() and sha(X / name) == h:
                say(f'H_SHA_OK {name} {(X / name).stat().st_size / 1e6:.1f} MB (attempt {attempt})'); break
            say(f'H_SHA_BAD {name} attempt {attempt}; re-downloading from scratch'); (X / name).unlink(missing_ok=True)
        else:
            say(f'H_ABORT {name}'); status(f'ABORT {name}'); return 2
    core = [k for k in sums if k.endswith('_core.tar')][0]
    stage = X.parent / f'v2_{TAG}_stage'; shutil.rmtree(stage, ignore_errors=True); stage.mkdir()
    with tarfile.open(X / core) as tf: tf.extractall(stage, filter='data')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    bak = ROOT / 'env' / 'code_backups'; bak.mkdir(parents=True, exist_ok=True)
    counts = {'new': 0, 'same': 0, 'code_updated': 0, 'pod_copy': 0}
    for p in sorted(stage.rglob('*')):
        if not p.is_file(): continue
        rel = p.relative_to(stage).as_posix(); tgt = ROOT / rel; tgt.parent.mkdir(parents=True, exist_ok=True)
        if not tgt.exists(): shutil.copyfile(p, tgt); counts['new'] += 1
        elif sha(tgt) == sha(p): counts['same'] += 1
        elif rel.startswith(CODE_PREFIXES): shutil.copy2(tgt, bak / f"{rel.replace('/', '__')}.{stamp}"); shutil.copyfile(p, tgt); counts['code_updated'] += 1
        else: shutil.copyfile(p, tgt.with_name(tgt.name + '_pod')); counts['pod_copy'] += 1
    say('H_INSTALLED ' + json.dumps(counts))
    for name in sums:
        if name.endswith('_raw.tar.gz'):
            dst = ROOT / 'datasets' / 'archives' / name; dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists(): shutil.copyfile(X / name, dst)
            ok = sha(dst) == sums[name]; say(f'H_RAW_ARCHIVE {name} sha {"OK" if ok else "MISMATCH"}')
            if not ok: status('ABORT raw archive copy'); return 3
    (ROOT / 'datasets' / 'archives' / f'SHA256SUMS_v2_{TAG}').write_text((X / 'SHA256SUMS').read_text())
    ack = sha(X / 'SHA256SUMS')
    for _ in range(30):
        r = subprocess.run(['curl', '-sS', f'{BASE}/ack?sha={ack}'], capture_output=True, text=True)
        if 'ACK_OK' in r.stdout: say('H_ACK_SENT'); break
        time.sleep(30)
    else:
        say('H_ACK_FAILED (pod keeps running; stop it by hand after checking Drive)')
    status('DONE ' + json.dumps(counts)); return 0


if __name__ == '__main__':
    sys.exit(main())
