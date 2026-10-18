"""Compare focused FVP frames against independent TOSA INT8 and old Q/DQ references."""
import argparse,json,sys,struct
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from investigate_numerics import read_frames,save
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);a=p.parse_args();root=a.directory;expected=json.loads((root/'expected.json').read_text());frames=read_frames(root/'project/fvp-stream.log');results=[f for f in frames if f['type']==5];acks={f['request']:f for f in frames if f['type']==4};assert len(results)==len(expected)==12
scale=0.052609603852033615;zp=-5;seen={};comparisons=[]
for f,e in zip(results,expected):
 assert f['session']==91231 and f['request']==e['request'] and f['trial']==e['index'];crc,l0,l1,pred=struct.unpack('<IffB3x',f['payload']);assert crc==e['input_crc']==struct.unpack('<I',acks[f['request']]['payload'])[0]
 logits=np.array([l0,l1],np.float32);raw=np.rint(logits.astype(np.float64)/scale+zp).astype(np.int16);assert np.array_equal((raw.astype(np.float32)-zp)*np.float32(scale),logits);assert raw.tolist()==e['tosa_raw'];assert pred==int(raw[1]>raw[0]);assert pred==int(e['host_raw'][1]>e['host_raw'][0])
 if e['index'] in seen:assert seen[e['index']]==raw.tolist()
 seen[e['index']]=raw.tolist();comparisons.append(dict(request=f['request'],index=e['index'],trial_id=e['trial_id'],raw=raw.tolist(),tosa_raw=e['tosa_raw'],desktop_raw=e['host_raw'],prediction=pred))
r=dict(n=12,unique_trials=len(seen),tosa_raw_exact=True,reordered_repeats_exact=True,predictions_match_desktop=True,sequence=[e['index'] for e in expected],comparisons=comparisons);save(root/'comparison.json',r);print(json.dumps({k:v for k,v in r.items() if k!='comparisons'},indent=2))
