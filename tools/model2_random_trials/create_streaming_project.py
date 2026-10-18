from pathlib import Path
import hashlib,json,tarfile,re,shutil
import argparse
parser=argparse.ArgumentParser(description='Create a checksum-verified local rollback archive and isolated streaming project; source is read-only.')
parser.add_argument('--baseline-repo',type=Path,required=True)
root=parser.parse_args().baseline_repo.resolve()
dev=Path(__file__).resolve().parents[2]
assert root!=dev and root not in dev.parents
p=root/'artifacts/fvp-subject1-20261016-ready/project';out=dev/'artifacts/streaming-rollback'
out.mkdir(parents=True,exist_ok=False)
files=[]
for folder in ('src','board','ai_layer'):
 files.extend(f for f in (p/folder).rglob('*') if f.is_file())
files.extend(f for f in p.iterdir() if f.is_file() and (f.suffix in ('.yml','.json','.txt') or f.name=='README.md'))
files.extend(p/f for f in ('.cmsis/tools-environment.yml','.vscode/fvp.sh'))
files.extend((p/'out/cmsis-executorch/DevKit-E8/Debug').glob('*.axf'))
files.extend((p/'out/cmsis-executorch/DevKit-E8/Debug').glob('*.hex'))
files.extend((root/'models_ready_for_alif_e8/02_subject_01_tinyeegnet').glob('*'))
files.extend((root/'results/e8-model2-uart-validation-20261017-1404').glob('*'))
files.extend(root/'.vscode'/name for name in ('launch.json','tasks.json','cmsis.json'))
files=sorted(set(f for f in files if f.is_file() and f.name!='.DS_Store'))
manifest={str(f.relative_to(root)):{'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size} for f in files}
archive=out/'validated-four-trial.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
 for f in files:tar.add(f,arcname=str(f.relative_to(root)),recursive=False)
with tarfile.open(archive) as tar:
 for member in tar:
  assert hashlib.sha256(tar.extractfile(member).read()).hexdigest()==manifest[member.name]['sha256']
(out/'manifest.json').write_text(json.dumps({'files':manifest,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'archive_bytes':archive.stat().st_size,'verified':True},indent=2)+'\n')
copy=dev/'artifacts/streaming-firmware/project'
copy.mkdir(parents=True,exist_ok=False)
for folder in ('src','board','ai_layer'):
 shutil.copytree(p/folder,copy/folder)
for f in p.iterdir():
 if f.is_file() and f.suffix in ('.yml','.json','.txt'):shutil.copy2(f,copy/f.name)
(copy/'.vscode').mkdir()
shutil.copy2(p/'.vscode/fvp.sh',copy/'.vscode/fvp-base.sh')
shutil.copy2(dev/'firmware/streaming/fvp-capture.sh',copy/'.vscode/fvp.sh')
(copy/'.vscode/fvp.sh').chmod(0o755)
settings=json.loads((root/'.vscode/settings.json').read_text())
configuration={'cmsis-csolution.environmentVariables':settings['cmsis-csolution.environmentVariables']}
(copy/'.vscode/settings.json').write_text(json.dumps(configuration,indent=2)+'\n')
workspace_settings=dev/'.vscode/settings.json';workspace_settings.parent.mkdir(exist_ok=True)
current=json.loads(workspace_settings.read_text()) if workspace_settings.exists() else {}
current.update(configuration);workspace_settings.write_text(json.dumps(current,indent=2)+'\n')
print('Rollback archive verified:',len(files),'files',archive.stat().st_size,'bytes')
print('Isolated firmware copy:',copy)
