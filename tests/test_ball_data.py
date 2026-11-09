"""Extraction tests use synthetic scorebooks, not implementation-shaped mocks."""
import copy
import importlib.util
import pathlib
import tempfile
import unittest
import json
import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[1]

def sample(deliveries,**innings_extra):
    names=['P'+str(i) for i in range(11)];opponents=['Q'+str(i) for i in range(11)]
    return {'info':{'dates':['2020-01-01'],'gender':'male','match_type':'ODI','team_type':'international',
        'teams':['India','New Zealand'],'venue':'Seddon Park, Hamilton','overs':50,'balls_per_over':6,
        'outcome':{'winner':'India'},'players':{'India':names,'New Zealand':opponents},
        'registry':{'people':{n:n.lower() for n in names+opponents}}},
        'innings':[{'team':'India','overs':[{'over':0,'deliveries':deliveries}],**innings_extra}]}

def ball(br=0,ex=None,wickets=None):
    ex=ex or {};d={'batter':'P0','non_striker':'P1','bowler':'Q0','runs':{'batter':br,'extras':sum(ex.values()),'total':br+sum(ex.values())}}
    if ex:d['extras']=ex
    if wickets:d['wickets']=wickets
    return d

class BallDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=ROOT/'prepare_ball_data.py'
        cls.prep=None
        if path.exists():
            spec=importlib.util.spec_from_file_location('ballprep',path);cls.prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.prep)

    def extract(self,g):
        self.assertIsNotNone(self.prep,'Delivery extraction module is not implemented')
        return self.prep.extract_game(g,'match',{'Seddon Park, Hamilton'})

    def test_wides_noballs_and_genuine_zero_ball_non_striker(self):
        g=sample([ball(0,{'wides':2}),ball(2,{'noballs':1}),ball(1),ball(0,wickets=[{'kind':'run out','player_out':'P1'}])])
        rows,events=self.extract(g);r=rows[0]
        self.assertTrue(r['eligible']);self.assertEqual(r['actual_legal_balls'],2)
        self.assertEqual(r['actual_team_runs'],6);self.assertEqual(r['actual_extras'],3)
        players={p['name']:p for p in r['players']}
        self.assertEqual(players['P0']['runs'],3);self.assertEqual(players['P0']['balls_faced'],3)
        self.assertEqual(players['P1']['balls_faced'],0);self.assertEqual(players['P1']['runs'],0)
        self.assertIsNone(players['P2']['balls_faced']);self.assertIsNone(players['P2']['runs'])
        self.assertEqual([e['outcome'] for e in events],[(0,2,1,0,1,0,0),(2,1,0,1,0,0,0),(1,0,0,0,1,0,0),(0,0,0,0,0,3,0)])

    def test_phase_uses_scheduled_budget_before_current_ball(self):
        g=sample([ball() for _ in range(61)])
        rows,events=self.extract(g)
        self.assertEqual([events[i]['phase'] for i in (0,59,60)],[0,0,1])
        self.assertEqual(rows[0]['scheduled_overs'],50)

    def test_nonstandard_events_explicitly_exclude_whole_innings(self):
        variants=[(sample([ball(0,{'noballs':2})]),'nonstandard_noball'),
            (sample([ball(0,wickets=[{'kind':'retired hurt','player_out':'P0'}])]),'unsupported_wicket'),
            (sample([ball()],penalty_runs={'pre':5}),'innings_penalty'),
            (sample([ball(0,wickets=[{'kind':'run out','player_out':'P0'},{'kind':'run out','player_out':'P1'}])]),'multiple_wickets')]
        for g,reason in variants:
            with self.subTest(reason=reason):
                rows,events=self.extract(g);self.assertFalse(rows[0]['eligible']);self.assertTrue(any(reason in x for x in rows[0]['exclusions']))
                self.assertFalse(any(e.get('eligible',False) for e in events))

    def test_extraction_does_not_mutate_input_and_compact_roundtrip_reconciles(self):
        g=sample([ball(4),ball(0,{'wides':1}),ball(0)])
        original=copy.deepcopy(g);rows,events=self.extract(g);self.assertEqual(g,original)
        scratch=ROOT.parents[1]/'work';scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as folder:
            source=pathlib.Path(folder)/'raw';source.mkdir();(source/'match.json').write_text(json.dumps(g),encoding='utf-8')
            dest=pathlib.Path(folder)/'out'
            result=self.prep.prepare(source,dest,nz_venues={'Seddon Park, Hamilton'})
            arr=np.load(dest/'ball_counts.npz')['counts'];meta=json.loads((dest/'ball_metadata.json').read_text())
            self.assertEqual(arr.dtype,np.dtype('int64'));self.assertEqual(arr.shape[1],6)
            self.assertEqual(int(arr[:,5].sum()),3)
            self.assertEqual(sum(meta['outcomes'][int(x[4])][0]*int(x[5]) for x in arr),4)
            self.assertEqual(meta['audit']['eligible_source_deliveries'],3)
            self.assertEqual(len(meta['innings']),2)
            self.assertTrue(meta['audit']['all_reconciliations_passed'])

if __name__=='__main__':unittest.main()
