import sys,pathlib,unittest,numpy as np
import json,tempfile
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from engine import forecast, winner_shares, decay_weight,load_data,_role
from prepare_data import prepare

class EngineTests(unittest.TestCase):
    def records(self):
        return [dict(id=str(j),date='2020-01-01',team='India',opponent='NZ',venue='Ground',host_nz=True,dl=False,no_result=False,eligible=True,early_trouble=bool(j%2),players=[dict(id=str(i),name='P'+str(i),position=i+1,runs=(i+j)%12) for i in range(8)]) for j in range(8)]
    def test_dead_heat_fractional_shares(self):
        s=np.array([[4,4,-1],[0,-1,0]])
        q,p=winner_shares(s)
        np.testing.assert_allclose(q,[.5,.25,.25]);np.testing.assert_allclose(p,[1,.5,.5]);self.assertAlmostEqual(sum(q),1)
    def test_time_decay(self):
        self.assertAlmostEqual(decay_weight('2020-01-01','2024-01-01',4),.5,places=3)
    def test_twelfth_participant_is_not_an_expected_xi_role(self):
        self.assertIsNone(_role(12))
    def test_early_trouble_counts_obstruction_but_not_retired_hurt(self):
        deliveries=[dict(batter='A',non_striker='B',runs={'batter':0},wickets=[{'player_out':'A','kind':kind}]) for kind in ('retired hurt','obstructing the field','bowled')]
        game={'info':{'gender':'male','match_type':'ODI','dates':['2020-01-01'],'teams':['India','NZ'],'players':{'India':['A','B']},'city':'Hamilton'},'innings':[{'team':'India','overs':[{'over':1,'deliveries':deliveries}]}]}
        with tempfile.TemporaryDirectory(dir=pathlib.Path(__file__).parent) as directory:
            pathlib.Path(directory,'1.json').write_text(json.dumps(game))
            self.assertTrue(prepare(directory)[0]['early_trouble'])
    def test_seed_coldstart_and_future_exclusion(self):
        rs=self.records();args=dict(team='India',lineup=['P'+str(i) for i in range(10)]+['New player'],venue='Ground',cutoff='2024-01-01',n_sims=400,seed=3)
        a=forecast(rs,**args);b=forecast(rs,**args)
        np.testing.assert_array_equal(a['scores'],b['scores'])
        future={**rs[0],'date':'2025-01-01','players':[dict(id='0',name='P0',position=1,runs=400)]}
        c=forecast(rs+[future],**args);np.testing.assert_array_equal(a['scores'],c['scores'])
        self.assertAlmostEqual(sum(p['q'] for p in a['players']),1)
        self.assertTrue(any('Cold' in w for w in a['warnings']))
        self.assertTrue(np.all(a['scores'][:,8:]==-1))
        self.assertTrue(all(p['mean_runs']>=0 for p in a['players']))
    def test_forced_regimes_and_full_name_alias_history(self):
        lineup=['Rohit Sharma','Shubman Gill','Virat Kohli','Shreyas Iyer','KL Rahul','New6','New7','New8','New9','New10','New11']
        data=load_data();args=dict(team='India',lineup=lineup,venue='Seddon Park, Hamilton',cutoff='2026-11-09',n_sims=100,seed=4)
        a=forecast(data,scenario='Steadier start',**args);b=forecast(data,scenario='Early trouble',**args)
        self.assertEqual(a['scenario_weights']['Early trouble'],0)
        self.assertEqual(b['scenario_weights']['Early trouble'],1)
        self.assertTrue(all(p['history_n']>0 for p in a['players'][:5]))
        self.assertEqual(a['players'][0]['canonical_name'],'RG Sharma')

if __name__=='__main__':unittest.main()
