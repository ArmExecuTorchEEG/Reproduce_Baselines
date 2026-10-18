"""Offline integrity checks for prepared Session 2 assets; no original data writes."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from select_trials import TrialSequence


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);a=p.parse_args()
    root=a.directory;m=json.loads((root/'manifest.json').read_text());q=json.loads((root/'qdq-validation.json').read_text())
    assert hashlib.sha256((root/'windows.npz').read_bytes()).hexdigest()==m['output_sha256']
    assert hashlib.sha256((root/'qdq-reference.npz').read_bytes()).hexdigest()==q['reference_sha256']
    with np.load(root/'windows.npz') as d,np.load(root/'qdq-reference.npz') as ref:
        inputs=d['inputs_int8'];x=d['normalized_float32'];uv=d['filtered_uv_float32'];labels=d['labels']
        assert inputs.shape==x.shape==uv.shape==(144,1,22,500)
        assert inputs.dtype==np.int8 and x.dtype==uv.dtype==np.float32
        assert np.isfinite(x).all() and np.isfinite(uv).all()
        quant=m['preprocessing']['quantization']['input']
        independently_quantized=np.clip(np.rint(x/np.float32(quant['scale']))+quant['zero_point'],quant['quant_min'],quant['quant_max']).astype(np.int8)
        assert np.array_equal(inputs,independently_quantized)
        np.testing.assert_allclose(uv,((x*np.array(m['preprocessing']['std'])[None,None,:,None]+np.array(m['preprocessing']['mean'])[None,None,:,None])*1e6),atol=2e-5,rtol=2e-7)
        assert np.bincount(labels).tolist()==[72,72]
        assert np.array_equal(labels,ref['labels'])
        assert len({t['trial_id'] for t in m['trials']})==144
        for i,t in enumerate(m['trials']):
            assert t['index']==i and t['ground_truth']==labels[i]
            assert inputs[i:i+1].shape==(1,1,22,500) and inputs[i].nbytes==11000
            assert hashlib.sha256(inputs[i].tobytes()).hexdigest()==t['input_sha256']
            assert t['trial_id']==q['trials'][i]['trial_id']
            assert ref['predictions'][i]==q['trials'][i]['prediction']
        ids=[t['trial_id'] for t in m['trials']];seq=TrialSequence(ids,42)
        rows=[next(seq) for _ in range(432)]
        for cycle in range(3):assert {r['trial_id'] for r in rows[cycle*144:(cycle+1)*144]}==set(ids)
    print('PASS: 144 genuine assets, channel shapes, finite display data, independent INT8 quantization, all trial checksums/labels, QDQ references and three complete no-replacement cycles.')


if __name__=='__main__':main()
