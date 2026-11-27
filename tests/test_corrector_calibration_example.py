import unittest

from src.corrector.calibration_example import run


class CalibrationExampleTest(unittest.TestCase):
    def test_finds_temperature_two_and_lowers_ece(self):
        r = run(n=20_000, seed=0)
        self.assertEqual(r["T_grid_0.5_to_4_step_0.05"], 2.0)
        self.assertAlmostEqual(r["T_ternary_search"], 2.0, delta=0.05)
        self.assertGreater(r["ece_T1"], 0.08)
        self.assertLess(r["ece_T_grid"], 0.01)
        self.assertLess(r["nll_T_grid"], r["nll_T1"])


if __name__ == "__main__":
    unittest.main()
