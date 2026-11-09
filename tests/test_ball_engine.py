import pathlib,sys,unittest,copy
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from ball_engine import _initial_state,_advance_state,_simulate_innings,fit_ball_model,forecast_match

class BallEngineTests(unittest.TestCase):
    def step(self,s,event):
        _advance_state(s,np.flatnonzero(~s['done']),np.array([event],dtype=int))
    def test_running_and_over_exchange(self):
        s=_initial_state(1,7,None)
        for _ in range(6):self.step(s,[0,0,0,0,0,0])
        self.assertEqual(s['striker'][0],1)
        self.step(s,[1,0,0,0,1,0])
        np.testing.assert_array_equal(s['scores'][0,:2],[0,1]);self.assertEqual(s['legal_balls'][0],7)
    def test_extras_free_hit_carries_and_wickets(self):
        s=_initial_state(1,12,None)
        self.step(s,[0,1,0,1,0,0]);self.step(s,[0,1,1,0,0,0])
        self.assertTrue(s['free_hit'][0]);self.assertEqual(s['legal_balls'][0],0)
        self.step(s,[0,0,0,0,0,1])
        self.assertEqual(s['wickets'][0],0);self.assertFalse(s['free_hit'][0])
        self.step(s,[0,0,0,0,0,1]);self.assertEqual(s['wickets'][0],1)
        self.assertEqual(s['balls_faced'].sum(),3);self.assertEqual(s['extras'][0],2)
    def test_runout_preserves_identity_after_crossing(self):
        s=_initial_state(1,2,None);self.step(s,[1,0,0,0,1,3])
        self.assertEqual(s['striker'][0],2);self.assertEqual(s['non_striker'][0],0)
        self.assertTrue(s['batted'][0,1]);self.assertEqual(s['balls_faced'][0,1],0)
        self.step(s,[2,0,0,0,0,0]);self.assertEqual(s['scores'][0,2],2)
    def test_no_new_batter_on_last_ball_or_reached_target(self):
        s=_initial_state(1,1,None);self.step(s,[0,0,0,0,0,1])
        self.assertEqual(s['batted'].sum(),2);self.assertEqual(s['scores'][0,2],-1)
        s=_initial_state(1,300,np.array([1]));self.step(s,[1,0,0,0,1,0])
        self.assertTrue(s['done'][0]);self.assertEqual(s['legal_balls'][0],1);self.assertEqual(s['batted'].sum(),2)
    def test_allout_and_illegal_safety_cap(self):
        cdf=np.ones((11,3,2,1));rng=np.random.default_rng(7)
        s=_simulate_innings(cdf,np.array([[0,0,0,0,0,1]]),2,300,1,rng)
        np.testing.assert_array_equal(s['wickets'],[10,10]);np.testing.assert_array_equal(s['legal_balls'],[10,10])
        with self.assertRaises(RuntimeError):_simulate_innings(cdf,np.array([[0,1,1,0,0,0]]),1,1,1,rng)
    def fixture(self):
        players=[{'id':str(i),'name':'RG Sharma' if i==0 else 'P'+str(i)} for i in range(22)]
        innings=[dict(id=str(i),match_id=str(i),date='2020-01-01',team='India' if i==0 else 'NZ',opponent='NZ' if i==0 else 'India',venue='Seddon Park, Hamilton',host_nz=True,innings_order=i+1,eligible=True,players=[dict(id=str(j),name=players[j]['name'],position=j%11+1,runs=2,balls_faced=2) for j in range(i*11,(i+1)*11)]) for i in range(2)]
        counts=np.array([[i,j,0 if j%11<3 else 1 if j%11<7 else 2,p,o,5] for i in range(2) for j in range(i*11,(i+1)*11) for p in range(3) for o in range(2)],dtype=np.int64)
        return dict(players=players,innings=innings,outcomes=[[0,0,0,0,0,0],[1,0,0,0,1,0]],counts=counts)
    def test_fit_future_vocabulary_invariance_alias_and_match_resources(self):
        d=self.fixture();lineups={'India':['Rohit Sharma']+['P'+str(i) for i in range(1,11)],'NZ':['P'+str(i) for i in range(11,22)]}
        args=dict(lineups=lineups,venue='Seddon Park, Hamilton',cutoff='2024-01-01',batting_first='Unknown',overs=20,n_sims=100,seed=7)
        a=forecast_match(d,**args);b=forecast_match(d,**args)
        future=copy.deepcopy(d);future['outcomes'].append([7,0,0,0,1,0]);future['innings'].append({**d['innings'][0],'date':'2025-01-01'})
        future['counts']=np.vstack([d['counts'],[2,0,0,0,2,99999]])
        # A future-only duplicate display name must not change past identity lookup.
        future['players'].append({'id':'future-person','name':'RG Sharma'})
        c=forecast_match(future,**args)
        fit=fit_ball_model(d,lineups,args['venue'],args['cutoff'])
        for pmf in fit['pmfs'].values():np.testing.assert_allclose(pmf.sum(axis=-1),1)
        self.assertEqual(a['teams']['India']['players'][0]['canonical_name'],'RG Sharma')
        for team in lineups:
            np.testing.assert_array_equal(a['teams'][team]['scores'],b['teams'][team]['scores'])
            np.testing.assert_array_equal(a['teams'][team]['scores'],c['teams'][team]['scores'])
            self.assertAlmostEqual(sum(p['q'] for p in a['teams'][team]['players']),1)
            self.assertTrue(np.all(a['legal_balls'][team]<=120))
            np.testing.assert_array_equal(np.maximum(a['teams'][team]['scores'],0).sum(axis=1)+a['extras'][team],a['team_totals'][team])
            np.testing.assert_array_equal(a['teams'][team]['balls_faced'].sum(axis=1),a['legal_balls'][team]+a['no_ball_deliveries'][team])
            self.assertTrue(np.all(a['teams'][team]['scores'][~a['teams'][team]['batted']]==-1))

if __name__=='__main__':unittest.main()
