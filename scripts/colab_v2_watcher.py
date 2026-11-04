# AeroForge-TRIAD: unattended Colab receiver for one 5 GHz worker pod (counterpart of scripts/runpod/finish_v2_worker.sh);
# same logic as colab_v2_s1_watcher.py with the pod TAG in every file name.
# Waits (up to 16 h) for the pod to send its export with the pre-agreed runpodctl code, verifies SHA256SUMS, and installs
# into Drive WITHOUT overwriting: new files are copied; identical files skipped; differing code files are backed up to
# env/code_backups/ then replaced; differing data/log/report files are kept and the pod copy is written with "_pod".
# Heartbeat and final status: _staging/v2_<TAG>_watcher_status.txt and logs/transfer_v2_<TAG>.log on Drive.
# Usage (Colab, Drive mounted): python /content/colab_v2_watcher.py CODE TAG
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path('.')
TAG = sys.argv[2] if len(sys.argv) > 2 else 'w0'
LOG = ROOT / 'logs' / f'transfer_v2_{TAG}.log'
STATUS = ROOT / '_staging' / f'v2_{TAG}_watcher_status.txt'
X = Path(f'/content/xfer_v2_{TAG}')
B = '/content/runpodctl114'  # same version as the pod (v1.14.15)
CODE_PREFIXES = ('scripts/', 'src/', 'research/')


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime('%FT%TZ')


def say(m):
    line = f'{now()} {m}'
    print(line, flush=True)
    with open(LOG, 'a') as fh:
        fh.write(line + '\n')


def status(m):
    STATUS.write_text(f'{now()} {m}\n')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for c in iter(lambda: fh.read(1 << 22), b''):
            h.update(c)
    return h.hexdigest()


def main(code):
    X.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + 16 * 3600
    say('W_START waiting for the pod export')
    n = 0
    while time.time() < deadline:
        n += 1
        status(f'WAITING attempt {n}')
        # runpodctl appends a relay index it chooses itself ("<code>-<k>"), so every suffix is tried in turn;
        # a wrong suffix fails within seconds, the right one transfers.
        tail = ''
        for k in range(64):
            try:
                r = subprocess.run([B, 'receive', f'{code}-{k}'], cwd=X, capture_output=True, text=True, timeout=7200,
                                   stdin=subprocess.DEVNULL)
                tail = (r.stdout + r.stderr)[-160:].replace('\n', ' | ')
            except subprocess.TimeoutExpired:
                tail = f'timeout on suffix {k}'
            if (X / 'SHA256SUMS').exists():
                break
        if (X / 'SHA256SUMS').exists() and all((X / f).exists() for f in (f'af_v2_{TAG}_core.tar', f'af_v2_{TAG}_raw.tar.gz')):
            say(f'W_RECEIVED attempt {n}: {tail}')
            break
        if n % 10 == 1:
            say(f'W_RETRY attempt {n}: {tail}')
        time.sleep(60)
    else:
        say('W_GAVE_UP no export received'); status('GAVE_UP'); return 1
    sums = dict(reversed(l.split()) for l in (X / 'SHA256SUMS').read_text().split('\n') if l.strip())
    for name, h in sums.items():
        if sha(X / name) != h:
            say(f'W_ABORT checksum mismatch {name}'); status(f'ABORT checksum {name}'); return 2
        say(f'W_SHA_OK {name} {os.path.getsize(X / name) / 1e6:.1f} MB')
    stage = Path(f'/content/v2_{TAG}_stage'); shutil.rmtree(stage, ignore_errors=True); stage.mkdir()
    with tarfile.open(X / f'af_v2_{TAG}_core.tar') as tf:
        tf.extractall(stage, filter='data')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    bak = ROOT / 'env' / 'code_backups'; bak.mkdir(parents=True, exist_ok=True)
    counts = {'new': 0, 'same': 0, 'code_updated': 0, 'pod_copy': 0}
    for p in sorted(stage.rglob('*')):
        if not p.is_file():
            continue
        rel = p.relative_to(stage).as_posix(); tgt = ROOT / rel; tgt.parent.mkdir(parents=True, exist_ok=True)
        if not tgt.exists():
            shutil.copyfile(p, tgt); counts['new'] += 1
        elif sha(tgt) == sha(p):
            counts['same'] += 1
        elif rel.startswith(CODE_PREFIXES):
            shutil.copy2(tgt, bak / f'{tgt.name}.{stamp}'); shutil.copyfile(p, tgt); counts['code_updated'] += 1
        else:
            shutil.copyfile(p, tgt.with_name(tgt.name + '_pod')); counts['pod_copy'] += 1
    say('W_INSTALLED ' + json.dumps(counts))
    raw = X / f'af_v2_{TAG}_raw.tar.gz'; dst = ROOT / 'datasets' / 'archives' / raw.name
    du = shutil.disk_usage('.')
    if du.free > raw.stat().st_size * 1.2 + 1e9 and not dst.exists():
        shutil.copyfile(raw, dst)
        ok = sha(dst) == sums[raw.name]
        say(f'W_RAW_ARCHIVE copied to Drive, sha {"OK" if ok else "MISMATCH"} ({raw.stat().st_size / 1e9:.2f} GB)')
        (ROOT / 'datasets' / 'archives' / f'SHA256SUMS_v2_{TAG}').write_text((X / 'SHA256SUMS').read_text())
    else:
        say(f'W_RAW_ARCHIVE not copied (exists={dst.exists()}, Drive free {du.free / 1e9:.1f} GB); kept at {raw}')
    summ = {}
    for f in ('datasets/designs/dataset_v2_design.json',):
        q = ROOT / f
        summ[f] = 'present' if q.exists() else 'MISSING'
    say('W_DONE ' + json.dumps(summ))
    status('DONE ' + json.dumps({**counts, **summ}))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
