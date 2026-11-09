import pathlib,sys,unittest
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from ball_engine import _initial_state,_advance_state

class ChaseTerminalTests(unittest.TestCase):
    def test_winning_wide_ends_before_stumping(self):
        s=_initial_state(1,300,np.array([1]))
        _advance_state(s,[0],np.array([[0,1,1,0,0,1,0]]))
        self.assertEqual(s['team_totals'][0],1)
        self.assertEqual(s['wickets'][0],0)
        self.assertEqual(s['batted'].sum(),2)
    def test_winning_running_run_is_capped_but_boundary_is_full(self):
        for boundary,runs,expected in [(0,2,1),(0,4,1),(4,4,4),(6,6,6)]:
            with self.subTest(boundary=boundary,runs=runs):
                s=_initial_state(1,300,np.array([1]))
                _advance_state(s,[0],np.array([[runs,0,0,0,runs%2,0,boundary]]))
                self.assertEqual(s['scores'][0,0],expected)
                self.assertEqual(s['team_totals'][0],expected)
    def test_completed_winning_runs_precede_later_runout(self):
        s=_initial_state(1,300,np.array([2]))
        _advance_state(s,[0],np.array([[2,0,0,0,0,2,0]]))
        self.assertEqual(s['wickets'][0],0)
        self.assertTrue(s['done'][0])

if __name__=='__main__':unittest.main()
