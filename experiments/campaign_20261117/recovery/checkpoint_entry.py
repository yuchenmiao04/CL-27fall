"""Show a prebound formal checkpoint; optionally hash one explicit final backup."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--job', required=True)
p.add_argument('--index', type=Path, default=Path(__file__).with_name('formal_checkpoint_index_partial_v1.json'))
p.add_argument('--full', action='store_true')
p.add_argument('--verify-file', type=Path, help='Explicit immutable completed backup only; never an active checkpoint')
a = p.parse_args()
index = json.loads(a.index.read_text())
assert len(index['entries']) == 12 and len({r['job'] for r in index['entries']}) == 12
rows = [r for r in index['entries'] if r['job'] == a.job]
if len(rows) != 1:
    p.error('Job is not one of the twelve prebound formal cells')
r = rows[0]
if a.verify_file:
    if r['status'] != 'obtained_verified_checkpoint':
        p.error('Pending cell has no final checkpoint SHA; do not read its active checkpoint')
    h = hashlib.sha256()
    with a.verify_file.open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    if a.verify_file.stat().st_size != r['checkpoint_bytes'] or h.hexdigest() != r['checkpoint_sha256']:
        raise SystemExit('Checkpoint size/SHA mismatch')
    print(json.dumps({'job': a.job, 'status': 'explicit_completed_file_size_and_sha_match',
                      'bytes': r['checkpoint_bytes'], 'sha256': h.hexdigest()}))
else:
    fields = ('job', 'status', 'checkpoint_bytes', 'checkpoint_sha256', 'backup',
              'runtime_snapshot', 'source_config_implementation_commit', 'audit')
    print(json.dumps(r if a.full else {k: r.get(k) for k in fields}, ensure_ascii=False, indent=2))
