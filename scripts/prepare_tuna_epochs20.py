"""Prepare the user-requested TUNA 20-epoch overlay; never launch training.

The hash-bound original four-job campaign and a006 scientific source remain
unchanged. Only the two TUNA image-training budgets change; the current20
entries are unused copies retained for the existing four-job controller API.
Execute this campaign only with ``run_followup_partition.py --partition tuna``.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil

if __package__:
    from .run_followup_partition import load_guarded_queue
else:
    from run_followup_partition import load_guarded_queue


BASE_SHA256 = '8a70c7e5a6011bb293c833387d71046173503d4ba94efca3c3ee2587b82e4c43'


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def validate_base(path):
    """Apply the original cache/source/config/recipe guards before overriding."""
    path = Path(path).resolve()
    need(sha(path) == BASE_SHA256, 'Original campaign manifest SHA differs')
    api = load_guarded_queue(path)
    base = api.FollowupQueue(path)
    original = importlib.import_module(api.__package__ + '.prepare_followup_campaign')
    need(original.source_inventory(base.repo) == base.manifest['source_files_sha256'],
         'Original scientific source inventory differs')
    for row in base.manifest['jobs']:
        cfg = json.loads(Path(row['config']['path']).read_text())
        original.check_template(base.repo, row['job'], cfg, row['dataset'], row['method'])
    return base


def assert_config_delta(original, modified, method, output):
    expected = copy.deepcopy(original)
    expected['output_dir'] = str(output)
    if method == 'tuna':
        need(original['tuned_epoch'] in (10, 15)
             and original['crct_epochs'] == 30
             and original['comparison_variant'] == 'pilot_native_same_protocol',
             'Unexpected original native TUNA recipe')
        expected.update(tuned_epoch=20, training_budget_variant='epochs20',
                        prefix='followup_tuna_epochs20')
    need(modified == expected,
         'Scientific delta exceeds the requested TUNA image epochs and budget tag')


def prepare(base_manifest_path, campaign_root):
    base_manifest_path = Path(base_manifest_path).resolve()
    campaign_root = Path(campaign_root).resolve()
    need(not campaign_root.exists(), 'Refusing existing campaign or overwrite')
    base = validate_base(base_manifest_path)
    base_text = base_manifest_path.read_text()
    need(hashlib.sha256(base_text.encode()).hexdigest() == BASE_SHA256,
         'Original manifest text changed during admission')
    staging = campaign_root.with_name(campaign_root.name + '.preparing')
    need(not staging.exists(), 'Unresolved prior overlay preparation directory')
    staging.mkdir(parents=True, exist_ok=False)
    try:
        (staging / 'formal/configs').mkdir(parents=True)
        manifest = copy.deepcopy(base.manifest)
        manifest.update(campaign_root=str(campaign_root), user_requested_image_epochs=20,
                        base_manifest_sha256=BASE_SHA256,
                        base_manifest_path=str(base_manifest_path), base_manifest_text=base_text)
        for row in manifest['jobs']:
            original = json.loads(Path(row['config']['path']).read_text())
            modified = copy.deepcopy(original)
            output = campaign_root / 'formal/runs' / row['job']
            modified['output_dir'] = str(output)
            if row['method'] == 'tuna':
                modified.update(tuned_epoch=20, training_budget_variant='epochs20',
                                prefix='followup_tuna_epochs20')
            assert_config_delta(original, modified, row['method'], output)
            config_name = row['job'] + '.json'
            staged_config = staging / 'formal/configs' / config_name
            final_config = campaign_root / 'formal/configs' / config_name
            write(staged_config, modified)
            row.update(config=dict(path=str(final_config), sha256=sha(staged_config)),
                       output_dir=str(output),
                       command=[manifest['python'], str(base.repo / 'main.py'),
                                '--config', str(final_config)])
        # Recheck every base file immediately before publishing the new overlay.
        base.validate()
        need(sha(base_manifest_path) == BASE_SHA256, 'Original manifest changed during preparation')
        write(staging / 'manifest.json', manifest)
        need(not campaign_root.exists(), 'Campaign appeared during preparation')
        os.rename(staging, campaign_root)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return campaign_root / 'manifest.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-manifest', type=Path, required=True)
    parser.add_argument('--campaign-root', type=Path, required=True)
    args = parser.parse_args()
    path = prepare(args.base_manifest, args.campaign_root)
    print(json.dumps(dict(status='prepared_tuna_epochs20_no_training',
                          manifest=str(path), sha256=sha(path),
                          base_manifest_sha256=BASE_SHA256)))


if __name__ == '__main__':
    main()
