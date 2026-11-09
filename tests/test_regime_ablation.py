import pathlib, sys, unittest
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from engine import forecast

class RegimeAblationTests(unittest.TestCase):
    def test_unconditioned_model_does_not_use_early_wicket_labels(self):
        records = [dict(id=str(j), date='2020-01-01', team='India', venue='Ground', host_nz=True,
                        eligible=True, early_trouble=bool(j % 2),
                        players=[dict(id=str(i), name=f'P{i}', position=i+1, runs=i+j) for i in range(8)])
                   for j in range(10)]
        args = dict(team='India', lineup=[f'P{i}' for i in range(11)], venue='Ground',
                    cutoff='2024-01-01', n_sims=1000, seed=7, use_regime=False)
        a = forecast(records, **args)
        changed = [{**r, 'early_trouble': not r['early_trouble']} for r in records]
        b = forecast(changed, **args)
        np.testing.assert_array_equal(a['scores'], b['scores'])
        self.assertAlmostEqual(sum(p['q'] for p in a['players']), 1.)
        self.assertEqual(a['scenario_weights'], {'Unconditioned': 1.0})

if __name__ == '__main__': unittest.main()
