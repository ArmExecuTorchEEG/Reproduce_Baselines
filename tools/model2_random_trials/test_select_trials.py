import unittest
from select_trials import TrialSequence


class SequenceTests(unittest.TestCase):
    ids=[f'trial-{i}' for i in range(144)]

    def test_unlimited_cycles_without_replacement(self):
        seq=TrialSequence(self.ids,42)
        records=[next(seq) for _ in range(144*4+17)]
        for cycle in range(4):
            rows=records[cycle*144:(cycle+1)*144]
            self.assertEqual({r['trial_id'] for r in rows},set(self.ids))
            self.assertTrue(all(r['cycle']==cycle for r in rows))
            self.assertEqual([r['position_in_cycle'] for r in rows],list(range(144)))
        self.assertNotEqual([r['trial_id'] for r in records[:144]],[r['trial_id'] for r in records[144:288]])

    def test_reproducible_pause_resume_reset(self):
        a=TrialSequence(self.ids,7);b=TrialSequence(self.ids,7)
        for _ in range(190):self.assertEqual(next(a),next(b))
        state=a.state();restored=TrialSequence.restore(self.ids,state)
        self.assertEqual(a.state(),state)  # pausing does not consume a selection
        for _ in range(500):self.assertEqual(next(a),next(restored))
        reset=a.reset(8);new=TrialSequence(self.ids,8)
        self.assertEqual(next(reset),next(new))
        self.assertEqual(reset.position,1)
        with self.assertRaises(ValueError):TrialSequence.restore(self.ids[::-1],state)

    def test_invalid_input(self):
        for ids in ([],['same','same']):
            with self.assertRaises(ValueError):TrialSequence(ids,42)
        with self.assertRaises(ValueError):TrialSequence(self.ids,42,-1)


if __name__=='__main__':unittest.main()
