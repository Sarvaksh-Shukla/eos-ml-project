import unittest

import numpy as np

from eos.eos_param import PiecewisePolytrope
from eos.units import MSUN
from inference.dataset import synth_dataset
from physics.tov import TOVSolver


class PiecewisePolytropeTests(unittest.TestCase):
    def setUp(self):
        self.eos = PiecewisePolytrope(
            rho1=1.69657870e14,
            rho2=5.44749660e14,
            Gamma1=1.52742632,
            Gamma2=2.01271558,
            Gamma3=3.09120941,
            P1=4.53355222e33,
        )

    def test_pressure_and_energy_density_are_continuous_at_breaks(self):
        for density in (self.eos.rho1, self.eos.rho2):
            below = density * (1.0 - 1e-9)
            above = density * (1.0 + 1e-9)
            self.assertAlmostEqual(
                self.eos.pressure(below) / self.eos.pressure(above),
                1.0,
                delta=1e-7,
            )
            self.assertAlmostEqual(
                self.eos.energy_density(below) / self.eos.energy_density(above),
                1.0,
                delta=1e-7,
            )

    def test_pressure_inverse_round_trips(self):
        densities = np.geomspace(1e13, 1e15, 20)
        recovered = np.array(
            [self.eos.rho_from_pressure(self.eos.pressure(rho)) for rho in densities]
        )
        np.testing.assert_allclose(recovered, densities, rtol=1e-12)

    def test_tov_surface_and_tidal_deformability_are_finite(self):
        solver = TOVSolver(self.eos)
        mass, radius, deformability = solver.tidal_deformability(1.2e15)
        self.assertTrue(np.isfinite(mass))
        self.assertTrue(np.isfinite(radius))
        self.assertTrue(np.isfinite(deformability))
        self.assertGreater(mass / MSUN, 1.5)
        self.assertLess(mass / MSUN, 2.8)
        self.assertGreater(radius / 1e5, 10.0)
        self.assertLess(radius / 1e5, 20.0)
        self.assertGreater(deformability, 0.0)

    def test_tov_solution_converges_when_step_is_halved(self):
        solver = TOVSolver(self.eos)
        coarse_mass, coarse_radius = solver.integrate_star(1.2e15, dr=2e4)
        fine_mass, fine_radius = solver.integrate_star(1.2e15, dr=1e4)
        self.assertAlmostEqual(fine_mass / coarse_mass, 1.0, delta=1e-5)
        self.assertAlmostEqual(fine_radius / coarse_radius, 1.0, delta=1e-5)

    def test_synthetic_dataset_has_requested_shapes_and_units_ranges(self):
        X, theta = synth_dataset(N=1, seed=42, max_tries=30)
        self.assertEqual(X.shape, (1, 3))
        self.assertEqual(theta.shape, (1, 6))
        self.assertAlmostEqual(X[0, 0], 1.4, delta=0.15)
        self.assertGreater(X[0, 1], 5.0)
        self.assertLess(X[0, 1], 40.0)
        self.assertGreater(X[0, 2], 0.0)


if __name__ == "__main__":
    unittest.main()
