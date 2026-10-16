"""Copy the two retained model packages without moving or overwriting originals.

Run from any directory with python -m src.package_models. Only explicitly approved checkpoint/PTE pairs are tracked; optional NPZ vectors
stay Git-ignored. --verify-only checks copies against their tracked SHA256 manifests;
original sources are checked too when present. No training or dataset access.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = (
    ('01_multi_subject_tinyeegnet', 'combined-seed42-20261016',
     'ethosu85-export-seed42-20261016-alif-e8'),
    ('02_subject_01_tinyeegnet', 'subject1-seed42-20261016',
     'subject1-ethosu85-seed42-20261016'),
)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify(directory):
    manifest = json.loads((directory / 'provenance.json').read_text())
    verified_count = 0
    for name, record in manifest['files'].items():
        copy = directory / name
        if name == 'test_vectors.npz' and not copy.exists():
            print(f'{directory.name}: optional local test vectors absent; skipped')
            continue
        if not copy.is_file() or digest(copy) != record['sha256']:
            raise ValueError(f'Missing or changed packaged file: {copy}')
        source = ROOT / record['original_path']
        if source.exists() and digest(source) != record['sha256']:
            raise ValueError(f'Original source no longer matches: {source}')
        verified_count += 1
    print(f'{directory.name}: {verified_count} files verified')


def create(name, training, exported):
    train = ROOT / 'artifacts' / training
    export = ROOT / 'artifacts' / exported
    out = ROOT / 'models_ready_for_alif_e8' / name
    sources = {
        'best_checkpoint.pt': train / 'best_checkpoint.pt',
        'tinyeegnet_int8.pte': export / 'tinyeegnet_int8.pte',
        'preprocess_params.json': export / 'preprocess_params.json',
        'normalization_params.json': train / 'preprocess_params.json',
        'model_config.json': train / 'config.json',
        'label_mapping.json': train / 'label_mapping.json',
        'training_metrics.json': train / 'metrics.json',
        'export_report.json': export / 'export_report.json',
        'test_vectors.npz': export / 'test_vectors.npz',
        'ensemble_vela.ini': export / 'ensemble_vela.ini',
    }
    report = json.loads(sources['export_report.json'].read_text())
    if digest(sources['best_checkpoint.pt']) != report['checkpoint_sha256']:
        raise ValueError('Checkpoint does not match the compiled model provenance')
    float_params = json.loads(sources['normalization_params.json'].read_text())
    deploy_params = json.loads(sources['preprocess_params.json'].read_text())
    for key in ('mean', 'std', 'sos', 'channel_order', 'labels'):
        if float_params[key] != deploy_params[key]:
            raise ValueError(f'Float/deployment preprocessing mismatch: {key}')
    manifest = {'package': name, 'schema_version': 1,
                'policy': 'Byte-identical convenience copies; originals remain authoritative',
                'files': {key: {'original_path': str(path.relative_to(ROOT)),
                                'sha256': digest(path), 'bytes': path.stat().st_size}
                          for key, path in sources.items()}}
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / 'provenance.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise FileExistsError(f'Refusing to replace provenance: {manifest_path}')
    # Preflight every destination before copying any files.
    for key, path in sources.items():
        dest = out / key
        if dest.exists() and digest(dest) != manifest['files'][key]['sha256']:
            raise FileExistsError(f'Refusing to overwrite a changed package file: {dest}')
    for key, path in sources.items():
        dest = out / key
        if not dest.exists():
            shutil.copy2(path, dest)
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    verify(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    for name, training, exported in PACKAGES:
        if args.verify_only:
            verify(ROOT / 'models_ready_for_alif_e8' / name)
        else:
            create(name, training, exported)


if __name__ == '__main__':
    main()
