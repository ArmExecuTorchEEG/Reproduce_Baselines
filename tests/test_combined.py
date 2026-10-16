import os
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/tinyeegnet-matplotlib')
import unittest
import numpy as np
import torch

from src.data.preprocess import Trial, Split
from src.data.splits import split_for, normalize_groups, validate_splits
from src.models.tinyeegnet import TinyEEGNet
from src.train import seed_everything


class CombinedTests(unittest.TestCase):
    def test_recording_allocations(self):
        counts = dict.fromkeys(('train', 'validation', 'demonstration', 'test'), 0)
        for subject in range(1, 10):
            for session in ('0train', '1test'):
                counts[split_for(subject, session)] += 144
        self.assertEqual(counts, dict(train=1728, validation=288, demonstration=288, test=288))
        self.assertEqual(split_for(6, '1test'), 'demonstration')
        self.assertEqual(split_for(8, '0train'), 'validation')
        self.assertEqual(split_for(9, '1test'), 'test')

    def test_training_only_stats(self):
        rng = np.random.default_rng(42)
        extracted = {name: (rng.normal(size=(4, 1, 22, 500)), np.array([0, 1, 0, 1]), ())
                     for name in ('train', 'validation', 'demonstration', 'test')}
        groups, mean, std = normalize_groups(extracted)
        for name in ('validation', 'demonstration', 'test'):
            extracted[name] = (extracted[name][0] * 100, extracted[name][1], ())
        other, other_mean, other_std = normalize_groups(extracted)
        np.testing.assert_array_equal(mean, other_mean)
        np.testing.assert_array_equal(std, other_std)
        torch.testing.assert_close(groups['train'].x, other['train'].x, rtol=0, atol=0)
        np.testing.assert_allclose(groups['train'].x.numpy().mean(axis=(0, 3)), 0, atol=1e-6)

    def test_leakage_and_duplicates_rejected(self):
        x, y = torch.zeros(2, 1, 22, 500), torch.tensor([0, 1])
        trial = Trial(1, '0train', '0', 500, 750, 1250, False)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            validate_splits({'train': Split(x, y, (trial, trial))})
        with self.assertRaisesRegex(ValueError, 'allocation'):
            validate_splits({'test': Split(x, y, (trial, Trial(1, '0train', '0', 2500, 2750, 3250, False)))})

    def test_forward_export_reload(self):
        seed_everything(42, 2)
        model = TinyEEGNet().eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 1938)
        for n in (1, 5):
            self.assertEqual(tuple(model(torch.zeros(n, 1, 22, 500)).shape), (n, 2))
        x = torch.randn(1, 1, 22, 500)
        exported = torch.export.export(model, (x,), strict=True)
        torch.testing.assert_close(model(x), exported.module()(x))
        other = TinyEEGNet().eval()
        other.load_state_dict(model.state_dict())
        torch.testing.assert_close(model(x), other(x), rtol=0, atol=0)

    def test_seed_reproduces_training_step(self):
        states = []
        for _ in range(2):
            seed_everything(42, 2)
            model = TinyEEGNet()
            opt = torch.optim.Adam(model.parameters(), lr=0.001)
            logits = model(torch.randn(4, 1, 22, 500))
            torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 0, 1])).backward()
            opt.step()
            states.append({k: v.clone() for k, v in model.state_dict().items()})
        for name in states[0]:
            torch.testing.assert_close(states[0][name], states[1][name], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
