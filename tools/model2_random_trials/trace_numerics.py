"""Trace saved integer graph versus float Q/DQ; no export or parameter changes."""
import argparse,hashlib,json,re,sys,os
from pathlib import Path
import numpy as np,torch,torchao,tosa_reference_model as ref
from tosa.TosaGraph import TosaGraph
from executorch.exir._serialize._program import deserialize_pte_binary
sys.path.insert(0,str(Path(__file__).parent))
from investigate_numerics import Capture,save,sha

def tensors_from_log(path):
 text=path.read_text();result={}
 pattern=r'^Name: (\S+) DType=(\S+) isValid=1 Rank=\d+ Shape=\[([^\]]*)\]\n(.*?)(?=\n\n)'
 for name,dtype,shape,values in re.findall(pattern,text,re.M|re.S):
  shape=tuple(map(int,re.findall(r'\d+',shape)))
  if dtype.startswith('INT'):
   data=np.array(list(map(int,re.findall(r'-?\d+',values))),np.int64)
   assert data.size==np.prod(shape),(name,data.size,shape)
   result[name]=data.reshape(shape)
 return result

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline-repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--indices',type=int,nargs='+',default=[0,3]);a=p.parse_args();root=Path.cwd();base=a.baseline_repo;out=a.out.resolve();export=base/'artifacts/subject1-ethosu85-seed42-20261016';model=torch.export.load(export/'quantized_qdq.pt2').module();torch.set_num_threads(4)
 graph_path=export/'compiler_debug/output_tag0_TOSA-1.0+INT+int16+int4+cf.tosa';graph=graph_path.read_bytes();b=TosaGraph.GetRootAs(graph).Regions(0).Blocks(0);ts={t.Name().decode():t for t in (b.Tensors(i) for i in range(b.TensorsLength()))};buffers=dict(model.named_buffers())
 def tensor(name,dtype):
  t=ts[name];return np.frombuffer(t.DataAsNumpy().tobytes(),dtype=dtype).reshape(tuple(t.ShapeAsNumpy()))
 weights=[]
 for index,name,channels in [(0,'aten_convolution_default_weight_hwicm',1),(2,'aten_convolution_default_1_weight_hwicm',8),(4,'aten_convolution_default_2_weight_hwicm',16),(5,'aten_convolution_default_3_weight_ohwi',None),(7,'aten_convolution_default_4_weight_ohwi',None)]:
  w=buffers[f'_frozen_param{index}'].numpy();target=tensor(name,'i1')
  if channels:expected=w.reshape(channels,w.shape[0]//channels,w.shape[2],w.shape[3]).transpose(2,3,0,1)
  elif w.ndim==4:expected=w.transpose(0,2,3,1)
  else:expected=w.reshape(w.shape[0],1,1,w.shape[1])
  assert np.array_equal(expected,target),name;weights.append(dict(parameter=f'_frozen_param{index}',tosa_tensor=name,shape=list(target.shape),raw_values_exact=True))
 for index in [1,3,6,8]:
  name=f'b__frozen_param{index}';assert np.array_equal(buffers[f'_frozen_param{index}'].numpy(),tensor(name,'<i4'));weights.append(dict(parameter=f'_frozen_param{index}',tosa_tensor=name,raw_values_exact=True))
 program=deserialize_pte_binary((export/'tinyeegnet_int8.pte').read_bytes()).program;plan=program.execution_plan[0];assert len(plan.operators)==1 and plan.operators[0].name=='quantized_decomposed::dequantize_per_tensor'
 assert plan.values[3].val.double_val==0.052609603852033615 and plan.values[4].val.int_val==-5
 report=dict(weights_and_biases=weights,pte_output_dequant=dict(scale=plan.values[3].val.double_val,zero_point=plan.values[4].val.int_val,raw_shape=plan.values[1].val.sizes,raw_dtype=str(plan.values[1].val.scalar_type),output_shape=plan.values[2].val.sizes),trials=[])
 data=np.load(root/'artifacts/model2-session2-phase1/windows.npz')
 mappings=[('quantize_per_tensor_2','tosa_rescale_default'),('quantize_per_tensor_3','tosa_rescale_default_1'),('quantize_per_tensor_4','tosa_avg_pool2d_default'),('quantize_per_tensor_7','tosa_rescale_default_2'),('quantize_per_tensor_8','tosa_rescale_default_3'),('quantize_per_tensor_9','tosa_avg_pool2d_default_1'),('quantize_per_tensor_12','tosa_reshape_default_2')]
 for idx in a.indices:
  folder=out/f'trace-trial{idx}';folder.mkdir(exist_ok=False);os.chdir(folder)
  values,status=ref.run(graph,[data['inputs_int8'][idx:idx+1].copy()],dump_intermediates=1,debug_filename=str(folder/'tosa-intermediates.log'));assert status==ref.GraphStatus.TOSA_VALID;dump=tensors_from_log(folder/'tosa-intermediates.log')
  with torch.no_grad():cap=Capture(model);cap.run(torch.from_numpy(data['normalized_float32'][idx:idx+1]))
  stages=[]
  for qname,tname in mappings:
   hn=next(n for n in cap.values if n==qname or n=='quantized_decomposed_'+qname+'_default');h=cap.values[hn];h=h.transpose(0,2,3,1) if h.ndim==4 else h;t=dump[tname];assert h.shape==t.shape,(hn,h.shape,tname,t.shape)
   delta=t-h.astype(np.int64);stages.append(dict(desktop_node=hn,tosa_tensor=tname,shape=list(h.shape),different_elements=int(np.count_nonzero(delta)),max_integer_delta=int(np.max(np.abs(delta)))))
  # Verify pooling rounding independent of any incoming activation differences.
  pool_examples=[]
  for before,after,kernel in [('tosa_rescale_default_1','tosa_avg_pool2d_default',4),('tosa_rescale_default_3','tosa_avg_pool2d_default_1',8)]:
   x=dump[before];y=dump[after];sums=x[:,:,:y.shape[2]*kernel,:].reshape(x.shape[0],x.shape[1],y.shape[2],kernel,x.shape[3]).sum(axis=3)
   # Zero-point is -128, so averaging raw integers is equivalent to averaging
   # zero-centered integers and adding the zero-point (128 divisible by4/8).
   even=np.rint(sums/kernel).astype(np.int64);up=np.floor(sums/kernel+.5).astype(np.int64)
   examples=[]
   for pos in np.argwhere(y!=even)[:8]:
    pos=tuple(map(int,pos));examples.append(dict(position=list(pos),sum=int(sums[pos]),denominator=kernel,exact_average=float(sums[pos]/kernel),ties_to_even=int(even[pos]),tosa=int(y[pos]),nearest_positive_tie=int(up[pos])))
   pool_examples.append(dict(pool=after,tosa_vs_even_count=int(np.count_nonzero(y!=even)),tosa_vs_up_count=int(np.count_nonzero(y!=up)),examples=examples))
  save(folder/'stages.json',dict(index=idx,stages=stages,pooling=pool_examples));np.savez_compressed(folder/'intermediate-integers.npz',**dump)
  report['trials'].append(dict(index=idx,stages=stages,pooling=pool_examples))
 save(out/'operator-trace.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
