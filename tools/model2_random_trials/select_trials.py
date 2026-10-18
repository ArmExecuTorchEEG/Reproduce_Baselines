"""Deterministic, unlimited cycling of verified real trial IDs; no transport or inference."""
import argparse
import hashlib
import json
from pathlib import Path


class TrialSequence:
    """A pull iterator: callers advance only when ready; state resumes exactly."""
    def __init__(self, trial_ids, seed, position=0):
        self.ids=tuple(trial_ids)
        if not self.ids or len(set(self.ids))!=len(self.ids):
            raise ValueError('Require a nonempty set of unique trial IDs')
        if not isinstance(seed,int) or not isinstance(position,int) or position<0:
            raise ValueError('Seed and nonnegative position must be integers')
        self.seed,self.position=seed,position
        self.dataset_sha256=hashlib.sha256(json.dumps(self.ids,separators=(',',':')).encode()).hexdigest()

    def __iter__(self):return self

    def __next__(self):
        cycle,offset=divmod(self.position,len(self.ids))
        def key(tid):
            return hashlib.sha256(json.dumps([self.seed,cycle,tid],separators=(',',':')).encode()).digest(),tid
        order=sorted(self.ids,key=key)
        result=dict(trial_id=order[offset],cycle=cycle,position_in_cycle=offset,selection_number=self.position+1,repeat_cycle=cycle>0)
        self.position+=1
        return result

    def state(self):
        return dict(schema=1,algorithm='sha256-sort-v1',seed=self.seed,position=self.position,dataset_sha256=self.dataset_sha256)

    @classmethod
    def restore(cls,ids,state):
        if state.get('schema')!=1 or state.get('algorithm')!='sha256-sort-v1':raise ValueError('Unsupported sequence state')
        result=cls(ids,state['seed'],state['position'])
        if result.dataset_sha256!=state['dataset_sha256']:raise ValueError('Dataset/order differs from saved state')
        return result

    def reset(self,seed):
        """Reset explicitly with a newly recorded seed for a new random sequence."""
        return TrialSequence(self.ids,seed)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--count',type=int,required=True,help='Any positive count; no four/twelve-trial cap')
    p.add_argument('--state-in',type=Path)
    p.add_argument('--state-out',type=Path)
    a=p.parse_args()
    if a.count<=0:p.error('count must be positive')
    rows=json.loads(a.manifest.read_text())['trials']
    by_id={r['trial_id']:r for r in rows}
    ids=[r['trial_id'] for r in rows]
    seq=TrialSequence.restore(ids,json.loads(a.state_in.read_text())) if a.state_in else TrialSequence(ids,a.seed)
    if seq.seed!=a.seed:p.error('seed must match restored state')
    for _ in range(a.count):
        record=next(seq);row=by_id[record['trial_id']]
        print(json.dumps(record|dict(dataset_index=row['index'],ground_truth=row['ground_truth'],input_sha256=row['input_sha256'])))
    if a.state_out:
        with a.state_out.open('x') as f:json.dump(seq.state(),f,indent=2);f.write('\n')


if __name__=='__main__':main()
