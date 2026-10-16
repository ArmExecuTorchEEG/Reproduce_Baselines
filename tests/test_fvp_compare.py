"""Parser verification with synthetic records; these are NOT FVP executions."""
import unittest
from src.fvp.compare import compare

class FVPComparisonTests(unittest.TestCase):
    reference={'expected_logits':[[1.,-1.],[-.5,.5]]}
    def test_reports_agreement_and_numerical_error(self):
        r=compare(self.reference,'EEG_RESULT 1 -0.5 0.51 1\nEEG_RESULT 0 1.005 -1 0\n')
        self.assertEqual(r['prediction_agreement'],1.)
        self.assertAlmostEqual(r['maximum_absolute_logit_difference'],.01)
    def test_rejects_missing_or_duplicate_outputs(self):
        for text in ('Execution_result: PASS','EEG_RESULT 0 1 -1 0\nEEG_RESULT 0 1 -1 0\n'):
            with self.assertRaises(ValueError):compare(self.reference,text)
    def test_flags_changed_prediction(self):
        r=compare(self.reference,'EEG_RESULT 0 -1 1 1\nEEG_RESULT 1 -0.5 0.5 1\n')
        self.assertEqual(r['prediction_agreement'],.5)
        self.assertFalse(r['logits_within_tolerance'])
    def test_rejects_nonfinite_or_inconsistent_prediction(self):
        for line in ('EEG_RESULT 0 nan -1 0','EEG_RESULT 0 1 -1 1'):
            with self.assertRaises(ValueError):compare(self.reference,line+'\nEEG_RESULT 1 -0.5 0.5 1\n')
