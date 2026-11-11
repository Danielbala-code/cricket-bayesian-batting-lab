"""Independent known-answer tests for the evaluation scale and decomposition."""
import pathlib
import sys
import unittest
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from audit_math import share_loss, calibration_bins


class EvaluationMathTests(unittest.TestCase):
    def test_binary_and_multiclass_conventions(self):
        # Binary one-event convention is .25; sum over both outcomes is .5.
        self.assertEqual((.5-1)**2, .25)
        self.assertAlmostEqual(share_loss([[.5,.5]], [[1,0]])[0], .5)
        self.assertAlmostEqual(share_loss([[1/11]*11], [[1]+[0]*10])[0], 10/11)

    def test_perfect_wrong_and_five_equal_contenders(self):
        self.assertEqual(share_loss([[1,0]], [[1,0]])[0], 0)
        self.assertEqual(share_loss([[0,1]], [[1,0]])[0], 2)
        self.assertAlmostEqual(share_loss([[.2]*5+[0]*6], [[1]+[0]*10])[0], .8)

    def test_fractional_tie_target_is_settlement_share_not_binary_win(self):
        q=np.array([[.5,.5,0]]); y=np.array([[.5,.5,0]])
        self.assertEqual(share_loss(q,y)[0],0)
        categorical_expected=(share_loss(q,[[1,0,0]])[0]+share_loss(q,[[0,1,0]])[0])/2
        self.assertEqual(categorical_expected, .5)
        self.assertEqual(categorical_expected,share_loss(q,y)[0]+1-np.square(y).sum())

    def test_invalid_simplex_is_rejected(self):
        for q in ([[.2,.2]], [[float('nan'),0]], [[-.1,1.1]]):
            with self.assertRaises(ValueError):share_loss(q,[[1,0]])

    def test_binned_decomposition_reports_coarsening_separately(self):
        q=np.array([[.22,.78],[.27,.73],[.24,.76],[.26,.74]])
        y=np.array([[1,0],[0,1],[0,1],[0,1]])
        r=calibration_bins(q,y)
        self.assertAlmostEqual(r['binned_share_mse'],r['reliability']-r['resolution']+r['outcome_share_variance'])
        self.assertAlmostEqual(r['raw_share_mse'],r['binned_share_mse']+r['raw_minus_binned'])
        self.assertNotAlmostEqual(r['raw_minus_binned'],0)


if __name__=='__main__':unittest.main()
