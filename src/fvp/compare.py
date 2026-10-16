"""Compare real EEG_RESULT FVP console records against staged host references."""
import argparse
import json
from pathlib import Path
import re
import numpy as np


def compare(reference, console, atol=.01):
    expected=np.asarray(reference['expected_logits'],dtype=float)
    records=re.findall(r'^EEG_RESULT (\d+) (\S+) (\S+) ([01])\s*$',console,re.M)
    if len(records)!=len(expected):
        raise ValueError('Missing or duplicate trial outputs; successful FVP execution is required')
    ordered=sorted(records,key=lambda r:int(r[0]))
    if [int(r[0]) for r in ordered]!=list(range(len(expected))): raise ValueError('Invalid trial indices')
    actual=np.array([[float(r[1]),float(r[2])] for r in ordered])
    if not np.isfinite(actual).all(): raise ValueError('Non-finite logits')
    predicted=actual.argmax(1)
    if not np.array_equal(predicted,[int(r[3]) for r in ordered]): raise ValueError('Firmware argmax mismatch')
    errors=np.abs(actual-expected)
    return {'trials':len(expected),'prediction_agreement':float(np.mean(predicted==expected.argmax(1))),
            'maximum_absolute_logit_difference':float(errors.max()),'mean_absolute_logit_difference':float(errors.mean()),
            'atol':atol,'logits_within_tolerance':bool(np.all(errors<=atol)),
            'actual_logits':actual.tolist(),'expected_logits':expected.tolist(),
            'all_predictions_match':bool(np.array_equal(predicted,expected.argmax(1)))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--log',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    report=compare(json.loads(a.reference.read_text()),a.log.read_text())
    a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    if not report['all_predictions_match'] or not report['logits_within_tolerance']:raise SystemExit(1)

if __name__=='__main__':main()
