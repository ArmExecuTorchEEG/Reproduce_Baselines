import argparse,shutil,json,sys,hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from prepare_streaming_assets import frame,crc
p=argparse.ArgumentParser(description='Create isolated numerical FVP test, leaving streaming implementation intact');p.add_argument('--out',type=Path,required=True);a=p.parse_args();root=Path.cwd();project=a.out.resolve();project.mkdir(parents=True,exist_ok=False);source=root/'artifacts/streaming-firmware/project'
for folder in ['src','board','ai_layer']:shutil.copytree(source/folder,project/folder)
for name in ['cmsis-executorch.csolution.yml','cmsis-executorch.cproject.yml','cmsis-executorch.cbuild-pack.yml','cdefault.yml','vcpkg-configuration.json']:
 if (source/name).exists():shutil.copy2(source/name,project/name)
(project/'.vscode').mkdir()
for name in ['fvp.sh','fvp-base.sh','settings.json']:shutil.copy2(source/'.vscode'/name,project/'.vscode'/name)
p=project/'cmsis-executorch.csolution.yml';p.write_text(p.read_text().replace(str(source.resolve()/'.vscode/fvp.sh'),str(project/'.vscode/fvp.sh')))
phase=root/'artifacts/model2-session2-phase1';meta=json.loads((phase/'manifest.json').read_text());d=np.load(phase/'windows.npz');refs=np.load(root/'artifacts/model2-numerical-investigation/integer-references.npz');sequence=[0,3,2,5,32,64,64,32,5,2,3,0];wire=bytearray(frame(1,91231,0,0xffffffff));bad=bytearray(frame(3,91231,1,0,d['inputs_int8'][0].tobytes()));bad[100]^=1;wire.extend(bad);wire.extend(b'\x05bad\0');expected=[]
for req,i in enumerate(sequence,1):
 raw=d['inputs_int8'][i].tobytes();wire.extend(frame(3,91231,req,i,raw));expected.append(dict(request=req,index=i,trial_id=meta['trials'][i]['trial_id'],input_crc=crc(raw),input_sha256=hashlib.sha256(raw).hexdigest(),tosa_raw=refs['tosa_raw_int8'][i].tolist(),host_raw=refs['desktop_raw_int8'][i].tolist(),ground_truth=meta['trials'][i]['ground_truth']))
(project/'src/test_frames.h').write_text('#pragma once\nstatic const unsigned expected_results='+str(len(expected))+';\nstatic const unsigned char test_frames[]={'+','.join(map(str,wire))+'};\n')
(project.parent/'expected.json').write_text(json.dumps(expected,indent=2)+'\n')
print('Prepared independent FVP numerical test:',sequence,'encoded bytes',len(wire))
