"""Bounded retrospective chronological diagnostic; fixed settings, no tuning."""
import pathlib,json,statistics,collections
import numpy as np
from engine import load_data,forecast

HALF_LIFE=4;KAPPA=20;N_SIMS=1500;MAX_TARGETS=40

def evaluate():
    records=load_data()
    targets=[r for r in records if r['eligible'] and r['date']>='2023-01-01' and r['team'] in ('India','New Zealand') and len(r['players'])==11][-MAX_TARGETS:]
    details=[]
    for t in targets:
        history=[r for r in records if r['eligible'] and r['date']<t['date']]
        # The XI is conditioned on the recorded match roster. Expected order uses
        # each player's past median position only; ties resolve lexically.
        def expected_position(p):
            previous=[q['position'] for r in history for q in r['players'] if q['id']==p['id'] and q['position'] is not None]
            return statistics.median(previous) if previous else 11
        lineup=[p['name'] for p in sorted(t['players'],key=lambda p:(expected_position(p),p['name']))]
        targetruns={p['name']:p['runs'] for p in t['players'] if p['runs'] is not None};maximum=max(targetruns.values())
        winner={name for name,value in targetruns.items() if value==maximum};y=np.array([1/len(winner) if n in winner else 0 for n in lineup])
        full=forecast(records,t['team'],lineup,t['venue'],t['date'],half_life=HALF_LIFE,kappa=KAPPA,n_sims=N_SIMS,seed=7)
        blind=forecast(records,t['team'],lineup,t['venue'],t['date'],half_life=HALF_LIFE,kappa=KAPPA,n_sims=N_SIMS,seed=7,use_venue=False)
        q=np.array([p['q'] for p in full['players']]);qb=np.array([p['q'] for p in blind['players']])
        counts=collections.Counter()
        for r in history:
            if r['team']!=t['team']:continue
            b=[p for p in r['players'] if p['runs'] is not None]
            if not b:continue
            mx=max(p['runs'] for p in b);ws=[p['name'] for p in b if p['runs']==mx]
            for n in ws:counts[n]+=1/len(ws)
        hist=np.array([counts[n]+1/11 for n in lineup]);hist/=hist.sum()
        distributions={'model':q,'uniform':np.repeat(1/11,11),'historical_top_share':hist,'venue_blind':qb}
        scores={name:float(np.square(prob-y).sum()) for name,prob in distributions.items()}
        details.append(dict(id=t['id'],date=t['date'],team=t['team'],venue=t['venue'],lineup=lineup,winners=sorted(winner),
                            training_max_date=full['training_max_date'],brier=scores,model_q=q.tolist(),observed_shares=y.tolist()))
        assert full['training_max_date']<t['date']
    means={name:float(np.mean([r['brier'][name] for r in details])) for name in ('model','uniform','historical_top_share','venue_blind')}
    out=dict(status='Retrospective diagnostic; not untouched holdout or evidence of betting edge',
             fixed_before_this_run=dict(half_life_years=HALF_LIFE,kappa=KAPPA,n_sims=N_SIMS,seed=7,max_targets=MAX_TARGETS),
             n=len(details),min_date=min(r['date'] for r in details),max_date=max(r['date'] for r in details),
             metric='Multiclass Brier: sum over11 players of (fractional-top-share probability minus fractional observed tie share)^2; lower better',
             mean_brier=means,limitations=['Current match recorded XI is assumed known; order reconstructed from prior median observed position, never current actual position',
             'Targets are the last40 eligible India/NZ team innings since2023; selection and prior full-history exploratory analysis preclude untouched-holdout claims',
             'Fixed4year half-life andkappa20; no tuning. No series-aware uncertainty or probability calibration claimed',
             'No-results and D/L excluded. Actual match odds, weather, lineups before toss and settlement policies not modeled'],details=details)
    (pathlib.Path(__file__).parent/'data/validation.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k!='details'},indent=2))

if __name__=='__main__':evaluate()
