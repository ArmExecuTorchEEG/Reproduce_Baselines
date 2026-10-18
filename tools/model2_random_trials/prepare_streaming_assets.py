"""Create local-only browser assets and FVP framed inputs from verified Phase1 data."""
import argparse,hashlib,json,shutil,sys
from pathlib import Path
import numpy as np
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parent))
from select_trials import TrialSequence

def sha(b):return hashlib.sha256(b).hexdigest()
def crc(b):
 import zlib
 return zlib.crc32(b)
def frame(typ,session,req,trial,payload=b''):
 import struct
 raw=struct.pack('<4sBBHQIIIHH',b'EEG2',1,typ,0,session,req,trial,0,len(payload),0)+payload
 raw+=struct.pack('<I',crc(raw));out=bytearray([0]);at=0;code=1
 for b in raw:
  if not b:out[at]=code;at=len(out);out.append(0);code=1
  else:
   out.append(b);code+=1
   if code==255:out[at]=code;at=len(out);out.append(0);code=1
 out[at]=code;out.append(0);return bytes(out)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline-repo',type=Path,required=True);p.add_argument('--full',action='store_true');p.add_argument('--diagnostic-index',type=int);a=p.parse_args();root=Path.cwd();baseline=a.baseline_repo
 phase=root/'artifacts/model2-session2-phase1';m=json.loads((phase/'manifest.json').read_text());q=json.loads((phase/'qdq-validation.json').read_text());assert sha((phase/'windows.npz').read_bytes())==m['output_sha256']
 with np.load(phase/'windows.npz') as z:inputs=z['inputs_int8'].copy();uv=z['filtered_uv_float32'].copy()
 data=root/'frontend/streaming/data';data.mkdir(exist_ok=True)
 (data/'inputs.bin').write_bytes(inputs.tobytes());(data/'eeg.f32').write_bytes(uv.astype('<f4').tobytes())
 payload=dict(schema=1,model_sha256='644bf8865ed515ed41c5418070a0b147008c4326b9da34d3232884459d9b3298',channel_order=m['preprocessing']['channel_order'],trials=[{k:t[k] for k in ('index','trial_id','ground_truth','source_trial','input_sha256')} for t in m['trials']],assets={name:sha((data/name).read_bytes()) for name in ('inputs.bin','eeg.f32')})
 (data/'metadata.json').write_text(json.dumps(payload,indent=2)+'\n')
 project=root/'artifacts/streaming-firmware/project'
 for name in ('protocol.hpp','app_main.cpp'):shutil.copy2(root/'firmware/streaming'/name,project/'src'/name)
 shutil.copy2(root/'firmware/streaming/uart_stream.c',project/'board/DevKit-E8/retarget_stdio.c')
 # Use original four input references, followed by six seed42 held-out trials plus a flagged trial.
 stage=baseline/'artifacts/fvp-subject1-20261016-ready';ref=json.loads((stage/'reference.json').read_text());original=np.load(stage/'validation_vectors.npz')
 ids=[t['trial_id'] for t in m['trials']];sequence=TrialSequence(ids,42);chosen=[ids.index(next(sequence)['trial_id']) for _ in range(144 if a.full else 6)]
 flagged=next(t['index'] for t in m['trials'] if t['source_trial']['artifact'])
 if flagged not in chosen:chosen.append(flagged)
 if a.full:chosen.extend(chosen[:3])
 if a.diagnostic_index is not None:
  assert 0<=a.diagnostic_index<144
  chosen=[a.diagnostic_index,a.diagnostic_index]
 cases=[]
 for i,x in enumerate(original['inputs']):cases.append(dict(trial=i,trial_id='original-four/'+str(i),input=x.tobytes(),logits=ref.get('logits',[])[i] if ref.get('logits') else None,prediction=ref.get('predictions',[])[i] if ref.get('predictions') else None))
 if a.diagnostic_index is not None:cases=[]
 # Reference.json keys may differ: NPZ carries validated host Q/DQ logits.
 for i,c in enumerate(cases):
  c['logits']=original['expected_logits'][i].tolist() if 'expected_logits' in original.files else original['expected'][i].tolist() if 'expected' in original.files else c['logits']
  assert c['logits'] is not None,(ref.keys(),original.files)
  c['prediction']=int(np.argmax(c['logits']))
 for i in chosen:cases.append(dict(trial=i,trial_id=m['trials'][i]['trial_id'],input=inputs[i].tobytes(),logits=q['trials'][i]['logits'],prediction=q['trials'][i]['prediction']))
 wire=bytearray(frame(1,424242,0,0xffffffff));expected=[]
 # Deliberately corrupted complete request and a truncated frame never invoke inference.
 bad=bytearray(frame(3,424242,1,0,cases[0]['input']));bad[100]^=1;wire.extend(bad);wire.extend(b'\x05bad\0')
 for req,c in enumerate(cases,1):wire.extend(frame(3,424242,req,c['trial'],c['input']));expected.append({k:v for k,v in (c|dict(request=req,input_crc32=crc(c['input']))).items() if k!='input'})
 (project/'src/test_frames.h').write_text('#pragma once\nstatic const unsigned expected_results='+str(len(cases))+';\nstatic const unsigned char test_frames[]={'+','.join(map(str,wire))+'};\n')
 (root/'artifacts/streaming-firmware/expected.json').write_text(json.dumps(expected,indent=2)+'\n')
 # Only edit copies; PTE and model C array remain byte-identical.
 original_project=stage/'project'
 for f in ('ai_layer/model_pte.c','ai_layer/model_pte.h','src/arm_embedded_module.cpp','src/arm_embedded_module.hpp'):
  assert (project/f).read_bytes()==(original_project/f).read_bytes()
 board=project/'board/DevKit-E8';config=board/'RTE/Device/AE822FA0E5597LS0_M55_HP/RTE_Device.h'
 config.write_text(config.read_text().replace('#define RTE_UART4_BLOCKING_MODE_ENABLE 1','#define RTE_UART4_BLOCKING_MODE_ENABLE 0'))
 layer=board/'Board-U85.clayer.yml';text=layer.read_text();text='\n'.join(line for line in text.splitlines() if 'Services:Retarget IO:' not in line)+'\n';layer.write_text(text)
 fvp=project/'board/Corstone-320/Board-U85.clayer.yml';text=fvp.read_text();
 if 'EEG_STREAM_FVP_TEST' not in text:text=text.replace('  define:\n','  define:\n    - EEG_STREAM_FVP_TEST\n')
 fvp.write_text(text)
 solution=project/'cmsis-executorch.csolution.yml';text=solution.read_text();text=text.replace('${workspaceFolder}/.vscode/fvp.sh',str(project/'.vscode/fvp.sh'));text=text.replace('--simlimit 60','--simlimit 300') if a.full else text;solution.write_text(text)
 mainc=board/'main.c';mainc.write_text(mainc.read_text().replace('    stdio_init();','    if (stdio_init() != 0) { for (;;) { __WFE(); } }'))
 print('Prepared local browser assets: 7,920,000 binary bytes; no reference predictions included.')
 print('FVP cases:',len(cases),'(',sum(c['trial_id'].startswith('original-four/') for c in cases),'original +',len(chosen),'Session2), frames',len(wire),'bytes')
if __name__=='__main__':main()
