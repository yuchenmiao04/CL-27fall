"""Prepare four fixed follow-up runs; never train or overwrite a campaign."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

NPZ_NAME = 'B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz'
NPZ_SHA = 'ac104c0df8158c46754510e50495536fa6de9647e572e38b5659dc54f260b124'
JOBS = (
    ('formal_cifar_current_epochs20_v1', 0, 'cifar224', 'current_adapted'),
    ('formal_imagenetr_current_epochs20_v1', 1, 'imagenetr', 'current_adapted'),
    ('formal_cifar_tuna_native_v1', 0, 'cifar224', 'tuna'),
    ('formal_imagenetr_tuna_native_v1', 1, 'imagenetr', 'tuna'),
)
CURRENT_SOURCE = {
    'models/current_adapted.py': '8cfbe15d35f4d956a665e6c553670bad296289c0ba457df07af7ab5585c0f1c1',
    'cl27/current/engine.py': '43fc2597aeaaa64f23517d598cbb62a0a18e0af1f82520bdfd58de265aa1e372',
    'cl27/current/episodes.py': '114dbb71f97a954f6bca786dd83599ca82caf2fcb273c93ab21e1500512a062f',
    'cl27/current/readout.py': 'a08cd3c7f036f589081dbb0422bd22114e8e002c2f96b122b9aab234a9c39547',
}
NATIVE_SOURCE = ('models/tuna.py', 'models/base.py', 'backbone/vit_tuna.py',
                 'backbone/linears.py', 'utils/inc_net.py',
                 'exps/tuna_cifar.json', 'exps/tuna_inr.json')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def need(ok, message):
    if not ok:
        raise ValueError(message)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def source_inventory(repo):
    pinned = read(repo / 'UPSTREAM_PIN.json')
    need(pinned['commit'] == '7a6e904c5bc5cb7a4e1823b3434020be27469b63', 'Wrong upstream snapshot')
    upstream = {row['path']: row['sha256'] for row in pinned['files']}
    expected = {**CURRENT_SOURCE, **{name: upstream[name] for name in NATIVE_SOURCE}}
    for name, digest in expected.items():
        need(sha(repo / name) == digest, 'Scientific source changed: ' + name)
    files = {repo / 'main.py', repo / 'trainer.py', repo / 'UPSTREAM_PIN.json'}
    files.update(repo / name for name in expected)
    for directory in ('cl27', 'models', 'utils', 'backbone'):
        files.update((repo / directory).rglob('*.py'))
    files.update(repo / 'scripts' / name for name in
                 ('prepare_followup_campaign.py', 'run_followup_queue.py'))
    return {str(p.relative_to(repo)): sha(p) for p in sorted(files)}


def check_template(repo, job, cfg, dataset, method):
    short = 'cifar' if dataset == 'cifar224' else 'imagenetr'
    if method == 'current_adapted':
        old = read(repo / 'experiments/campaign_20261117/formal/configs' /
                   ('formal_' + short + '_current_selected_v5.json'))
        mutable = {'epochs', 'prefix', 'comparison_variant', 'data_root', 'output_dir',
                   'implementation_commit', 'reference_5epoch_source_commit', 'path_export_note'}
        need({k: v for k, v in cfg.items() if k not in mutable}
             == {k: v for k, v in old.items() if k not in mutable}, 'Current scientific recipe changed')
        need(cfg['epochs'] == 20 and cfg['comparison_variant'] == 'epochs20', 'Wrong current budget')
    else:
        old = read(repo / ('exps/tuna_cifar.json' if short == 'cifar' else 'exps/tuna_inr.json'))
        protocol = {'prefix', 'init_cls', 'increment', 'backbone_type', 'device'}
        need(all(cfg.get(k) == v for k, v in old.items() if k not in protocol), 'Native TUNA hyperparameter changed')
        need(cfg['backbone_type'] == 'vit_base_patch16_224_tuna'
             and cfg['comparison_variant'] == 'pilot_native_same_protocol', 'Wrong TUNA branch')
    width = 10 if dataset == 'cifar224' else 20
    need(cfg['model_name'] == method and cfg['dataset'] == dataset
         and (cfg['init_cls'], cfg['increment']) == (width, width)
         and cfg['seed'] == [1993] and cfg['phase'] == 'final'
         and cfg['evaluation_head'] == 'cnn' and cfg['validation_only'] is False
         and not any(cfg.get(k) is not None for k in ('resume', 'max_tasks', 'validation_max_per_class')),
         'Wrong complete same-protocol configuration: ' + job)


def prepare(runtime_root, repo, python, campaign_root, source_commit=None):
    runtime_root, repo, campaign_root = (Path(p).resolve() for p in
                                         (runtime_root, repo, campaign_root))
    # Resolving a virtualenv's Python symlink would silently select its base
    # interpreter and lose the admitted environment.
    python = Path(os.path.abspath(python))
    need(repo.is_dir() and python.is_file(), 'Executed repository/Python must exist')
    need(not campaign_root.exists(), 'Refusing existing campaign or overwrite')
    need(all((runtime_root / 'data' / d).is_dir() for d in ('cifar100', 'imagenetr')), 'Missing existing datasets')
    npz = runtime_root / 'torch/hub/checkpoints' / NPZ_NAME
    need(npz.is_file() and sha(npz) == NPZ_SHA, 'Missing or wrong existing pretrained NPZ; no download')
    sources = source_inventory(repo)
    templates = repo / 'experiments/extension_20261117/configs'
    staging = campaign_root.with_name(campaign_root.name + '.preparing')
    need(not staging.exists(), 'Unresolved prior preparation directory')
    staging.mkdir(parents=True, exist_ok=False)
    try:
        (staging / 'formal/configs').mkdir(parents=True)
        entries = []
        for job, gpu, dataset, method in JOBS:
            template = templates / (job + '.json')
            cfg = read(template)
            check_template(repo, job, cfg, dataset, method)
            cfg['data_root'] = str(runtime_root / 'data' / ('cifar100' if dataset == 'cifar224' else 'imagenetr'))
            cfg['output_dir'] = str(campaign_root / 'formal/runs' / job)
            actual = campaign_root / 'formal/configs' / template.name
            write(staging / 'formal/configs' / template.name, cfg)
            entries.append(dict(job=job, gpu=gpu, dataset=dataset, method=method,
                                comparison_variant=cfg['comparison_variant'],
                                config=dict(path=str(actual), sha256=sha(staging / 'formal/configs' / template.name)),
                                template_sha256=sha(template), output_dir=cfg['output_dir'],
                                command=[str(python), str(repo / 'main.py'), '--config', str(actual)],
                                environment={'CL27_CURRENT_FROZEN_PREFIX_REUSE': '1'} if method == 'current_adapted' else {}))
        manifest = dict(version=1, runtime_root=str(runtime_root), repo=str(repo), python=str(python),
                        campaign_root=str(campaign_root), source_commit=source_commit,
                        scientific_source_guards='Native TUNA upstream bytes; current five-epoch core bytes unchanged.',
                        source_files_sha256=sources,
                        pretrained_npz=dict(path=str(npz), sha256=NPZ_SHA),
                        environment=dict(TORCH_HOME=str(runtime_root / 'torch'), OMP_NUM_THREADS='4',
                                         OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
                                         PYTHONUNBUFFERED='1', CUBLAS_WORKSPACE_CONFIG=':4096:8'),
                        minimum_initial_free_bytes=13 * 2**30, jobs=entries,
                        queue={str(gpu): [r['job'] for r in entries if r['gpu'] == gpu] for gpu in (0, 1)})
        write(staging / 'manifest.json', manifest)
        need(not campaign_root.exists(), 'Campaign appeared during preparation')
        os.rename(staging, campaign_root)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return campaign_root / 'manifest.json'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('runtime-root', 'repo', 'python', 'campaign-root'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--source-commit', help='Optional actual checkout commit; source-file SHAs remain authoritative')
    a = p.parse_args()
    path = prepare(a.runtime_root, a.repo, a.python, a.campaign_root, a.source_commit)
    print(json.dumps(dict(status='prepared_only_no_training', manifest=str(path), sha256=sha(path))))


if __name__ == '__main__':
    main()
