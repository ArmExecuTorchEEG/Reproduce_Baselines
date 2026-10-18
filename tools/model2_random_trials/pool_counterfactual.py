"""Diagnostic interpreter only: isolate integer AVG_POOL rounding in saved Q/DQ graph."""
import argparse,json,sys
from pathlib import Path
import numpy as np,torch,torchao
sys.path.insert(0,str(Path(__file__).parent))
from investigate_numerics import Capture,save
class IntegerPool(Capture):
 def run_node(self,n):
  mapping={'quantize_per_tensor_4':('quantize_per_tensor_3',4),'quantize_per_tensor_9':('quantize_per_tensor_8',8)}
  if n.name in mapping:
   prev,k=mapping[n.name];x=torch.from_numpy(self.values[prev]).to(torch.int64);width=x.shape[-1]//k
   sums=x[...,:width*k].reshape(*x.shape[:-1],width,k).sum(-1)
   # Round exact integer average to nearest, half-way toward positive infinity.
   # ReLU activation zero-point -128 and kernel4/8 make raw/centered equivalent.
   v=torch.div(sums+k//2,k,rounding_mode='floor').clamp(-128,127).to(torch.int8)
   self.values[n.name]=v.numpy().copy();return v
  return super().run_node(n)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline-repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();root=Path.cwd();out=a.out
 export=a.baseline_repo/'artifacts/subject1-ethosu85-seed42-20261016';m=torch.export.load(export/'quantized_qdq.pt2').module();data=np.load(root/'artifacts/model2-session2-phase1/windows.npz');ref=np.load(out/'integer-references.npz');torch.set_num_threads(4);pred=[];detail=[]
 for i in range(144):
  with torch.no_grad():interpreter=IntegerPool(m);interpreter.run(torch.from_numpy(data['normalized_float32'][i:i+1]))
  raw=interpreter.values['quantize_per_tensor_12'][0];pred.append(raw)
  if not np.array_equal(raw,ref['tosa_raw_int8'][i]):detail.append(dict(index=i,diagnostic_raw=raw.tolist(),tosa_raw=ref['tosa_raw_int8'][i].tolist()))
 pred=np.asarray(pred);report=dict(scope='diagnostic interpreter only; saved Q/DQ/PTE/firmware unchanged',override='integer half-up AVG_POOL outputs only, kernel4 and8, at two existing quantize nodes',n=144,tosa_exact_trials=int(np.all(pred==ref['tosa_raw_int8'],axis=1).sum()),tosa_mismatched_elements=int(np.count_nonzero(pred!=ref['tosa_raw_int8'])),mismatches=detail)
 # Record an actual first-pool mismatch, including the desktop float quantizer input.
 with torch.no_grad():cap=Capture(m);cap.run(torch.from_numpy(data['normalized_float32'][3:4]))
 dump=np.load(out/'trace-trial3/intermediate-integers.npz');host=cap.values['quantize_per_tensor_4'].transpose(0,2,3,1);target=dump['tosa_avg_pool2d_default'];before=dump['tosa_rescale_default_1'];sums=before.reshape(1,1,125,4,16).sum(3);float_pool=cap.values['avg_pool2d'].transpose(0,2,3,1);scale=.02125292643904686
 examples=[]
 for pos in np.argwhere(host!=target)[:12]:
  pos=tuple(map(int,pos));pre=float_pool[pos];scaled=torch.tensor(pre)*float(1/scale)
  examples.append(dict(position=list(pos),input_raw_sum=int(sums[pos]),kernel=4,exact_raw_average=float(sums[pos]/4),desktop_float_pool=float(pre),desktop_pre_round_scaled=float(scaled),desktop_round=float(torch.round(scaled)),desktop_quantized=int(host[pos]),tosa_quantized=int(target[pos])))
 report['actual_first_pool_examples']=examples;save(out/'pool-rounding-counterfactual.json',report);np.save(out/'diagnostic-pool-rounded-outputs.npy',pred);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
