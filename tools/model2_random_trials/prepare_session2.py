"""Read-only reconstruction of Model 2 held-out EEG; never trains or accesses hardware."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def prepare(root, out):
    import numpy as np
    import torch
    # Import original implementation without running download/cache setup or main().
    sys.path.insert(0, str(root))
    from src.data.preprocess import EEG_CHANNELS, Split, extract_session
    from src.models.tinyeegnet import TinyEEGNet
    from src.eval import evaluate
    from moabb.datasets.bnci.base import _convert_mi
    from moabb.datasets.preprocessing import SetRawAnnotations
    from scipy.signal import butter
    train=root/'artifacts/subject1-seed42-20261016'
    export=root/'artifacts/subject1-ethosu85-seed42-20261016'
    params=json.loads((export/'preprocess_params.json').read_text())
    norm=json.loads((train/'preprocess_params.json').read_text())
    for key in ('mean','std','channel_order','sos'):
        assert params[key]==norm[key]
    assert params['channel_order']==list(EEG_CHANNELS)
    assert np.array_equal(butter(4,[7,30],btype='bandpass',fs=250,output='sos'),params['sos'])
    source=list((root/'mne_data').rglob('A01E.mat'))
    assert len(source)==1
    source=source[0]
    records=json.loads((train/'source_recordings.json').read_text())
    assert sha(source)==next(r['sha256'] for r in records if r['session']=='1test')
    assert sha(export/'tinyeegnet_int8.pte')=='644bf8865ed515ed41c5418070a0b147008c4326b9da34d3232884459d9b3298'
    assert sha(train/'best_checkpoint.pt')=='63e3479baa0b00cadce5ee7906f25eee873ba561f9040cd07505f5d32b7a51d9'
    runs,codes=_convert_mi(str(source),list(EEG_CHANNELS)+['EOG1','EOG2','EOG3'],['eeg']*22+['eog']*3,artifact_handling='annotate',artifact_interval=(2,6))
    assert codes=={'left_hand':1,'right_hand':2,'feet':3,'tongue':4}
    marker=SetRawAnnotations(codes,interval=[2,6])
    runs={str(i):marker.transform(raw) for i,raw in enumerate(runs)}
    x,y,trials,audit=extract_session(runs,'1test',np.asarray(params['sos']),subject=1)
    manifest=json.loads((train/'split_manifest.json').read_text())
    rows=[asdict(t) for t in trials]
    assert rows==manifest['test']['trials']
    identities=lambda rows:{(t['subject'],t['session'],t['run'],t['cue_sample']) for t in rows}
    assert not identities(rows)&identities(manifest['train']['trials']+manifest['validation']['trials'])
    assert all(t['session']=='0train' for s in ('train','validation') for t in manifest[s]['trials'])
    assert x.shape==(144,1,22,500) and np.bincount(y).tolist()==[72,72]
    normalized=((x-np.asarray(params['mean'])[None,None,:,None])/np.asarray(params['std'])[None,None,:,None]).astype(np.float32)
    saved=torch.load(train/'prepared_tensors.pt',weights_only=True)['test']
    assert np.array_equal(normalized,saved['x'].numpy()) and np.array_equal(y,saved['y'].numpy())
    # Same torch float32 rounding and clipping as original src.fvp.prepare.
    q=params['quantization']['input']
    assert q['shape']==[1,1,22,500]
    v=torch.from_numpy(normalized)
    unbounded=torch.round(v/q['scale'])+q['zero_point']
    quantized=torch.clamp(unbounded,q['quant_min'],q['quant_max']).to(torch.int8).numpy()
    model=TinyEEGNet(dropout=.5).eval()
    model.load_state_dict(torch.load(train/'best_checkpoint.pt',weights_only=True)['model_state'])
    torch.set_num_threads(4)
    with torch.inference_mode():
        logits=torch.cat([model(z[None]) for z in v])
    prior=torch.load(train/'predictions.pt',weights_only=True)['test']
    torch.testing.assert_close(logits,prior['logits'],rtol=1e-5,atol=1e-5)
    assert torch.equal(logits.argmax(1),prior['logits'].argmax(1))
    metrics=evaluate(model,Split(v,torch.tensor(y),trials))
    expected=json.loads((train/'metrics.json').read_text())['metrics']['test']
    assert metrics['confusion_matrix']==expected['confusion_matrix']
    # Verify training-only normalization from the saved development tensors' source split,
    # and calibration identities from the original export manifest.
    export_manifest=json.loads((export/'data_manifest.json').read_text())
    assert identities(export_manifest['calibration_trials'])<=identities(manifest['train']['trials'])
    assert not identities(rows)&identities(export_manifest['calibration_trials'])
    out.mkdir(parents=True,exist_ok=False)
    display=(x*1e6).astype(np.float32)
    np.savez_compressed(out/'windows.npz',inputs_int8=quantized,normalized_float32=normalized,filtered_uv_float32=display,labels=y.astype(np.int64),float_logits=logits.numpy())
    descriptions=[]
    for i,t in enumerate(rows):
        tid=f"S1/1test/run{t['run']}/cue{t['cue_sample']}"
        descriptions.append(dict(index=i,trial_id=tid,ground_truth=int(y[i]),source_trial=t,input_sha256=hashlib.sha256(quantized[i].tobytes()).hexdigest(),float_logits=logits[i].tolist(),float_prediction=int(logits[i].argmax())))
    files=[source,train/'best_checkpoint.pt',train/'preprocess_params.json',train/'split_manifest.json',train/'source_recordings.json',train/'prepared_tensors.pt',train/'predictions.pt',train/'metrics.json',export/'preprocess_params.json',export/'data_manifest.json',export/'quantized_qdq.pt2',export/'tinyeegnet_int8.pte',root/'src/train_subject.py',root/'src/data/preprocess.py',root/'src/eval.py',root/'src/models/tinyeegnet.py']
    report=dict(schema=1,n=144,class_counts=[72,72],artifact_flagged=sum(t.artifact for t in trials),weight_training_trials=115,checkpoint_selection_trials=29,training_unseen=True,previously_evaluated=True,normalized_saved_test_bit_exact=True,all_float_predictions_match_saved=True,float_metrics=metrics,max_float_logit_error=float((logits-prior['logits']).abs().max()),saturation_count=int(((unbounded<q['quant_min'])|(unbounded>q['quant_max'])).sum()),total_input_samples=int(quantized.size),source_files={str(p.relative_to(root)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in files},versions={name:importlib.metadata.version(name) for name in ('torch','numpy','scipy','mne','moabb')},preprocessing=params,display_units='filtered microvolts, before normalization and int8 quantization; not live acquisition',audit=audit,trials=descriptions,output_sha256=sha(out/'windows.npz'),license_status='Local preparation only. Dataset redistribution/license review required before publishing samples; no dataset committed.')
    write(out/'manifest.json',report)
    print(json.dumps({k:report[k] for k in ('n','class_counts','artifact_flagged','float_metrics','max_float_logit_error','saturation_count')},indent=2))


def validate_qdq(root,out):
    import numpy as np
    import torch
    import torchao  # registers quantized_decomposed operators used by saved export
    sys.path.insert(0,str(root))
    from src.export import metrics
    torch.set_num_threads(4)
    export=root/'artifacts/subject1-ethosu85-seed42-20261016'
    m=json.loads((out/'manifest.json').read_text())
    path=out/'qdq-reference.npz'
    assert not path.exists(),'Refuse to overwrite validation reference'
    assert sha(out/'windows.npz')==m['output_sha256']
    assert sha(export/'quantized_qdq.pt2')==m['source_files']['artifacts/subject1-ethosu85-seed42-20261016/quantized_qdq.pt2']['sha256']
    with np.load(out/'windows.npz') as d:
        x=torch.from_numpy(d['normalized_float32'].copy());q=torch.from_numpy(d['inputs_int8'].copy());labels=torch.from_numpy(d['labels'].copy());fl=torch.from_numpy(d['float_logits'].copy())
    model=torch.export.load(export/'quantized_qdq.pt2').module()
    quant=m['preprocessing']['quantization']['input']
    deq=(q.float()-quant['zero_point'])*quant['scale']
    with torch.no_grad():
        logits=torch.cat([model(t[None]) for t in x])
        transferred=torch.cat([model(t[None]) for t in deq])
        dev=torch.load(export/'export_data.pt',weights_only=True)
        dl=torch.cat([model(t[None]) for t in dev['development_x']])
    with np.load(export/'development_predictions.npz') as saved:
        np.testing.assert_allclose(dl.numpy(),saved['qdq_logits'],rtol=0,atol=1e-6)
    torch.testing.assert_close(transferred,logits,rtol=0,atol=0)
    np.savez_compressed(path,qdq_logits=logits.numpy(),predictions=logits.argmax(1).numpy(),labels=labels.numpy())
    report=dict(scope='saved host Q/DQ graph; not new FVP or physical NPU execution',metrics=metrics(logits,labels),float_prediction_agreement=float((logits.argmax(1)==fl.argmax(1)).float().mean()),int8_dequantized_logits_exact=True,saved_development_reference_matches=True,reference_sha256=sha(path),versions={n:importlib.metadata.version(n) for n in ('torch','torchao','executorch')},trials=[dict(trial_id=t['trial_id'],prediction=int(logits[i].argmax()),logits=logits[i].tolist()) for i,t in enumerate(m['trials'])])
    write(out/'qdq-validation.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='trials'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','validate-qdq'])
    p.add_argument('--baseline-repo',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args()
    os.environ.setdefault('MPLCONFIGDIR','/private/tmp/edgebci-phase1/matplotlib')
    os.environ.setdefault('MNE_DONTWRITE_HOME','true')
    root=a.baseline_repo.resolve();out=a.out_dir.resolve()
    if out==root or root in out.parents:raise ValueError('Outputs must be outside read-only original workspace')
    (prepare if a.action=='prepare' else validate_qdq)(root,out)
