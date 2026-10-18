"""Read-only numerical audit: saved PT2E Q/DQ, saved TOSA integer graph and FVP frames."""
import argparse,hashlib,json,struct,zlib,csv,os
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import torchao
import tosa_reference_model as reference
from tosa.TosaGraph import TosaGraph
from tosa.Op import Op

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,obj):p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def uncobs(b):
 out=bytearray();i=0
 while i<len(b):
  code=b[i];i+=1;assert code and i+code-1<=len(b)
  out.extend(b[i:i+code-1]);i+=code-1
  if code!=255 and i<len(b):out.append(0)
 return bytes(out)
def read_frames(path):
 frames=[]
 for line in path.read_text().splitlines():
  if not line.startswith('STREAM_FRAME '):continue
  wire=bytes.fromhex(line[13:]);assert wire[-1]==0;b=uncobs(wire[:-1]);assert zlib.crc32(b[:-4])==struct.unpack('<I',b[-4:])[0]
  magic,ver,typ,flags,session,req,trial,res,n,res2=struct.unpack('<4sBBHQIIIHH',b[:32]);assert magic==b'EEG2' and ver==1 and flags==res==res2==0 and len(b)==36+n
  frames.append(dict(type=typ,session=session,request=req,trial=trial,payload=b[32:-4]))
 return frames
class Capture(torch.fx.Interpreter):
 def __init__(self,m):super().__init__(m);self.values={}
 def run_node(self,n):
  v=super().run_node(n)
  if isinstance(v,torch.Tensor):self.values[n.name]=v.detach().clone().numpy()
  return v

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline-repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();root=Path.cwd();base=a.baseline_repo;out=a.out;out.mkdir(parents=True,exist_ok=False)
 export=base/'artifacts/subject1-ethosu85-seed42-20261016';phase=root/'artifacts/model2-session2-phase1';stream=root/'artifacts/streaming-firmware';meta=json.loads((phase/'manifest.json').read_text());expected=json.loads((stream/'expected-151.json').read_text());identity={t['trial_id']:t for t in meta['trials']}
 assert sha(phase/'windows.npz')==meta['output_sha256']
 with np.load(phase/'windows.npz') as d:inputs=d['inputs_int8'].copy();normalized=d['normalized_float32'].copy();labels=d['labels'].copy()
 with np.load(phase/'qdq-reference.npz') as d:host_saved=d['qdq_logits'].copy();assert np.array_equal(labels,d['labels'])
 params=json.loads((export/'preprocess_params.json').read_text());assert params==meta['preprocessing']
 train=base/'artifacts/subject1-seed42-20261016';saved=torch.load(train/'prepared_tensors.pt',weights_only=True)['test'];assert np.array_equal(saved['x'].numpy(),normalized) and np.array_equal(saved['y'].numpy(),labels)
 checkpoint_hash='63e3479baa0b00cadce5ee7906f25eee873ba561f9040cd07505f5d32b7a51d9';assert sha(train/'best_checkpoint.pt')==checkpoint_hash
 assert sha(export/'tinyeegnet_int8.pte')=='644bf8865ed515ed41c5418070a0b147008c4326b9da34d3232884459d9b3298'
 assert (stream/'project/ai_layer/model_pte.c').read_bytes()==(base/'artifacts/fvp-subject1-20261016-ready/project/ai_layer/model_pte.c').read_bytes()
 assert (root/'frontend/streaming/data/inputs.bin').read_bytes()==inputs.tobytes()
 qin=params['quantization']['input'];rebuilt=torch.clamp(torch.round(torch.from_numpy(normalized)/qin['scale'])+qin['zero_point'],-128,127).to(torch.int8).numpy();assert np.array_equal(inputs,rebuilt)
 model=torch.export.load(export/'quantized_qdq.pt2').module();torch.set_num_threads(4)
 quant_nodes=[n for n in model.graph.nodes if n.op=='call_function' and 'quantize_per_tensor.default' in str(n.target) and 'dequantize' not in str(n.target)]
 final=quant_nodes[-1];scale=float(final.args[1]);zp=int(final.args[2]);assert scale==0.052609603852033615 and zp==-5
 graph_path=export/'compiler_debug/output_tag0_TOSA-1.0+INT+int16+int4+cf.tosa';graph=graph_path.read_bytes();b=TosaGraph.GetRootAs(graph).Regions(0).Blocks(0)
 assert b.InputsLength()==b.OutputsLength()==1
 fvp={};acks={};exp={e['request']:e for e in expected};repeated=[]
 for f in read_frames(stream/'fvp-151-trial.log'):
  if f['type'] not in (4,5):continue
  e=exp[f['request']];assert f['trial']==e['trial'] and f['session']==424242
  if not e['trial_id'].startswith('S1/1test/'):continue
  row=identity[e['trial_id']];i=row['index'];assert i==f['trial'];assert sha_bytes(inputs[i].tobytes())==row['input_sha256'];checksum=zlib.crc32(inputs[i].tobytes());assert checksum==e['input_crc32']==struct.unpack('<I',f['payload'][:4])[0]
  if f['type']==4:acks[f['request']]=True;continue
  assert acks.get(f['request']);_,l0,l1,pred=struct.unpack('<IffB3x',f['payload']);v=np.array([l0,l1],np.float32);assert pred==int(l1>l0)
  if i in fvp:assert np.array_equal(fvp[i],v);repeated.append(i)
  fvp[i]=v
 assert set(fvp)==set(range(144)) and len(repeated)==3
 fvp_float=np.stack([fvp[i] for i in range(144)])
 raw_fvp=np.rint(fvp_float.astype(np.float64)/scale+zp).astype(np.int16);assert ((raw_fvp>=-128)&(raw_fvp<=127)).all()
 reconstruct=(raw_fvp.astype(np.float32)-np.float32(zp))*np.float32(scale);assert np.array_equal(reconstruct,fvp_float)
 host_raw=[];host_float=[];tosa_raw=[]
 # These are model evaluation calls only. No training, calibration, export or compiler invocation.
 for i in range(144):
  with torch.no_grad():
   cap=Capture(model);v=cap.run(torch.from_numpy(normalized[i:i+1]));raw=cap.values[final.name];host_raw.append(raw[0]);host_float.append(v[0][0].numpy() if isinstance(v,tuple) else v[0].numpy())
  result,status=reference.run(graph,[inputs[i:i+1].copy()]);assert status==reference.GraphStatus.TOSA_VALID
  assert result[0].dtype==np.int8 and result[0].shape==(1,2);tosa_raw.append(result[0][0])
  if i%24==23:print('Audited',i+1,'/144',flush=True)
 host_raw=np.asarray(host_raw);host_float=np.asarray(host_float);tosa_raw=np.asarray(tosa_raw)
 assert np.array_equal(host_float,host_saved)
 np.savez_compressed(out/'integer-references.npz',desktop_raw_int8=host_raw,tosa_raw_int8=tosa_raw,fvp_reconstructed_int8=raw_fvp.astype(np.int8),fvp_float=fvp_float,desktop_float=host_float)
 delta=raw_fvp-host_raw.astype(np.int16);pertrial=np.max(np.abs(delta),axis=1);diff=fvp_float.astype(np.float64)-host_float.astype(np.float64)
 rows=[]
 for i,t in enumerate(meta['trials']):
  hm=float(host_float[i,1]-host_float[i,0]);fm=float(fvp_float[i,1]-fvp_float[i,0]);rows.append(dict(index=i,trial_id=t['trial_id'],ground_truth=int(labels[i]),artifact=t['source_trial']['artifact'],desktop_raw=host_raw[i].tolist(),fvp_raw_reconstructed=raw_fvp[i].tolist(),tosa_raw=tosa_raw[i].tolist(),raw_delta=delta[i].tolist(),desktop_logits=host_float[i].tolist(),fvp_logits=fvp_float[i].tolist(),logit_delta=diff[i].tolist(),max_abs_error=float(np.abs(diff[i]).max()),max_integer_steps=int(pertrial[i]),legacy_001_pass=bool(np.abs(diff[i]).max()<=.01),desktop_prediction=int(host_float[i,1]>host_float[i,0]),fvp_prediction=int(fvp_float[i,1]>fvp_float[i,0]),desktop_margin=hm,fvp_margin=fm,margin_delta=fm-hm))
 summary=dict(n=144,output_scale=scale,output_zero_point=zp,trial_step_histogram=dict(Counter(map(int,pertrial))),logit_signed_step_histogram=dict(Counter(map(int,delta.ravel()))),logit_absolute_step_histogram=dict(Counter(map(int,np.abs(delta).ravel()))),tosa_fvp_raw_exact=bool(np.array_equal(tosa_raw,raw_fvp)),tosa_fvp_mismatch_elements=int(np.count_nonzero(tosa_raw!=raw_fvp)),legacy_001_failed_trials=sum(not r['legacy_001_pass'] for r in rows),max_abs_error=float(np.abs(diff).max()),predictions_changed=sum(r['desktop_prediction']!=r['fvp_prediction'] for r in rows),min_abs_desktop_margin=min(abs(r['desktop_margin']) for r in rows),min_abs_fvp_margin=min(abs(r['fvp_margin']) for r in rows),max_abs_margin_change=max(abs(r['margin_delta']) for r in rows),repeated_input_indices=repeated,repeated_results_exact=True,input_bytes_and_ids_verified=True,normalized_saved_test_bit_exact=True,desktop_reference_bit_exact=True,fvp_raw_values_source='uniquely reconstructed from transmitted float32 values; dequantization is bit-exact for all288values',tosa_raw_values_source='direct INT8 output of saved pre-Vela TOSA graph',source_files={str(p):sha(p) for p in [graph_path,export/'quantized_qdq.pt2',export/'tinyeegnet_int8.pte',train/'best_checkpoint.pt',export/'preprocess_params.json',stream/'fvp-151-trial.log',stream/'expected-151.json',phase/'windows.npz',phase/'qdq-reference.npz']})
 save(out/'summary.json',summary);save(out/'all-trials.json',rows)
 with (out/'failing-trials.csv').open('w') as f:
  columns=['index','trial_id','ground_truth','artifact','desktop_raw','fvp_raw_reconstructed','tosa_raw','raw_delta','desktop_logits','fvp_logits','logit_delta','max_abs_error','max_integer_steps','desktop_margin','fvp_margin','margin_delta'];w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(r for r in rows if not r['legacy_001_pass'])
 print(json.dumps(summary,indent=2),flush=True)
def sha_bytes(b):return hashlib.sha256(b).hexdigest()
if __name__=='__main__':main()
