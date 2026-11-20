import unittest

import numpy as np

from inference.sbi_flows import flow_sample_posterior, train_flow
from inference.supervised_baseline import predict_inverse, train_inverse_regressor


class InferenceTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.X = np.column_stack(
            (
                rng.uniform(1.1, 1.8, 24),
                rng.uniform(10.0, 15.0, 24),
                rng.uniform(100.0, 1200.0, 24),
            )
        )
        self.TH = np.column_stack(
            (
                rng.uniform(1.2e14, 1.8e14, 24),
                rng.uniform(4.5e14, 6e14, 24),
                rng.uniform(1.3, 1.6, 24),
                rng.uniform(1.8, 2.4, 24),
                rng.uniform(2.8, 3.1, 24),
                10 ** rng.uniform(np.log10(2e33), np.log10(5e33), 24),
            )
        )

    def test_trainers_handle_large_parameter_scales(self):
        inverse = train_inverse_regressor(
            self.X, self.TH, epochs=2, verbose=False, device="cpu"
        )
        prediction = predict_inverse(inverse, self.X[:1], device="cpu")
        flow = train_flow(self.TH, self.X, epochs=2, verbose=False, device="cpu")
        samples = flow_sample_posterior(flow, self.X[0], n_samples=8)
        self.assertEqual(prediction.shape, (1, 6))
        self.assertEqual(samples.shape, (8, 6))
        self.assertTrue(np.all(np.isfinite(prediction)))
        self.assertTrue(np.all(np.isfinite(samples)))

    def test_trainers_reject_non_finite_data(self):
        invalid_x = self.X.copy()
        invalid_x[0, 0] = np.nan
        with self.assertRaises(ValueError):
            train_inverse_regressor(invalid_x, self.TH, epochs=1, verbose=False)
        with self.assertRaises(ValueError):
            train_flow(self.TH, invalid_x, epochs=1, verbose=False)


if __name__ == "__main__":
    unittest.main()
