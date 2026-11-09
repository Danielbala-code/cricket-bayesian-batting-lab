"""Parameter sensitivity of the same future XI, not hyperparameter selection."""
import json, math, pathlib
import numpy as np
from engine import load_data, forecast

def run():
    root=pathlib.Path(__file__).parent
    fixture=json.loads((root/'data/fixture.json').read_text())
    records=load_data(); rows=[]
    for h in (2,4,math.inf):
        for k in (5,20):
            f=forecast(records,'India',fixture['lineups']['India'],fixture['venue'],fixture['date'],
                       half_life=h,kappa=k,n_sims=50000,seed=7)
            s=f['scores']; a=s[:,4]>=50
            b=(s[:,0]>=0)&(s[:,0]<=20)&(s[:,1]>=0)&(s[:,1]<=20)
            n=int(b.sum()); hits=int((a&b).sum()); p=hits/n
            rows.append(dict(half_life_years=h if math.isfinite(h) else 'infinity',kappa=k,
                             conditional_P_Rahul50_given_openers_le20=p,condition_n=n,joint_n=hits,
                             monte_carlo_se=float(np.sqrt(p*(1-p)/n)),Rahul_top_share=f['players'][4]['q'],
                             early_trouble_weight=f['scenario_weights']['Early trouble']))
    out=dict(status='Sensitivity only; these future-scenario outputs do not select settings or establish an edge',
             event='Rahul50+ given Rohit and Gill both bat and each score <=20',n_simulations_per_setting=50000,
             fixed_seed=7,rows=rows,
             caveats=['Spread across settings is not a credible interval and does not cover all model uncertainty',
                      'Residual score independence, weather and lineup uncertainty remain',
                      'Settings must be selected from earlier development matches, never from this price/scenario'])
    (root/'data/sensitivity_v2.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2))

if __name__=='__main__':run()
