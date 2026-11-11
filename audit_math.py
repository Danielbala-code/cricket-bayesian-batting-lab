"""Read-only audit of saved evaluation artifacts and a declared fixture scenario.

No fitting/tuning on evaluation labels. Does not overwrite original artifacts.
Run: python audit_math.py [--rankings]
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / 'math_audit_2026_11_10'
EDGES = np.array([0, .05, .10, .15, .20, .30, .50, 1.00000001])


def share_loss(q, y):
    q, y = np.asarray(q, dtype=float), np.asarray(y, dtype=float)
    if q.shape != y.shape or q.ndim != 2:
        raise ValueError('Expected matching [team innings, XI member] matrices.')
    if not (np.isfinite(q).all() and np.isfinite(y).all()):
        raise ValueError('Nonfinite forecasts or targets.')
    if np.any(q < 0) or np.any(y < 0) or not (np.allclose(q.sum(1), 1) and np.allclose(y.sum(1), 1)):
        raise ValueError('Forecast and target shares must lie on the simplex.')
    return np.square(q-y).sum(1)


def calibration_bins(q, y):
    """Descriptive pooled SHARE calibration; y can be fractional, not binary.

    REL/RES/UNC below exactly decompose the loss of bin-mean predictions.
    They do not exactly decompose the original varying forecasts: report the
    coarsening difference. Player rows in an innings are dependent.
    """
    p, a = np.asarray(q).ravel(), np.asarray(y).ravel()
    idx = np.searchsorted(EDGES[1:], p, side='right')
    rows = []; grouped_p = np.empty_like(p); overall = a.mean()
    rel = res = 0.
    for b in range(len(EDGES)-1):
        mask = idx == b
        if not mask.any():
            continue
        pm, am, n = float(p[mask].mean()), float(a[mask].mean()), int(mask.sum())
        weight = n / len(p); grouped_p[mask] = pm
        rel += weight * (pm-am)**2
        res += weight * (am-overall)**2
        rows.append(dict(lower=float(EDGES[b]), upper=min(1.,float(EDGES[b+1])), n=n,
                         mean_forecast_share=pm, mean_observed_share=am))
    unc = float(np.var(a)); raw = float(np.mean((p-a)**2))
    binned = float(np.mean((grouped_p-a)**2))
    assert np.isclose(binned, rel-res+unc)
    return dict(bins=rows, n_dependent_player_rows=len(p),
                share_calibration_absolute_gap=sum(r['n']*abs(r['mean_forecast_share']-r['mean_observed_share']) for r in rows)/len(p),
                scale='Pooled per-player squared share error; multiply by XI size for summed-XI scale',
                raw_share_mse=raw, binned_share_mse=binned,
                reliability=rel, resolution=res, outcome_share_variance=unc,
                raw_minus_binned=raw-binned,
                warning='Descriptive finite-sample, bin-dependent share calibration; not binary P(top) calibration or proof of tradable probabilities.')


def artifact_report(filename, q_key, y_key):
    data = json.loads((ROOT/'data'/filename).read_text())
    details = data['details']
    q = np.asarray([q_key(d) for d in details]); y = np.asarray([d[y_key] for d in details])
    losses = share_loss(q,y)
    uniform = share_loss(np.full_like(q,1/q.shape[1]),y)
    # Expected categorical Brier if a tied leader were selected uniformly as
    # the one categorical outcome. This adds the same target-only constant to
    # every model, leaving paired differences unchanged.
    tie_constant = 1-np.square(y).sum(1)
    models = list(data['mean_brier'])
    reconstruction_errors = {}
    for method in models:
        saved_mean = np.mean([d['brier'][method] for d in details])
        reconstruction_errors[method] = float(saved_mean-data['mean_brier'][method])
        assert abs(reconstruction_errors[method]) < 1e-12
    selected_name = 'v3' if filename.startswith('ball_') else 'selected'
    np.testing.assert_allclose(losses,[d['brier'][selected_name] for d in details],atol=1e-12)
    return dict(n=len(details), unique_games=len({d['id'] for d in details}),
                tied_team_innings=int(np.sum((y>0).sum(1)>1)),
                saved_mean_brier=data['mean_brier'], primary_recomputed=float(losses.mean()),
                uniform_recomputed=float(uniform.mean()),
                skill_vs_uniform=float(1-losses.mean()/uniform.mean()),
                mean_target_tie_correction=float(tie_constant.mean()),
                expected_random_tie_categorical_brier=float(np.mean(losses+tie_constant)),
                saved_mean_errors=reconstruction_errors,
                brier_range=[float(losses.min()),float(losses.max())],
                strict_date_checks=all(d['training_max_date']<d['date'] for d in details),
                calibration=calibration_bins(q,y))


def matched_calibration():
    """Recover missing v2 comparator vectors on the exact 91 v3 cases.

    Exact original seed/settings/lineup. Assert loss matches the frozen table;
    this is replay, not a new fitted candidate or a retuned evaluation.
    """
    import engine
    data=json.loads((ROOT/'data/ball_validation.json').read_text())
    original=json.loads((ROOT/'data/validation_v2.json').read_text())
    old={(d['id'],d['team']):d for d in original['details']}
    rosters={}
    for r in data['player_diagnostics']:
        rosters.setdefault((r['id'],r['team']),[]).append(r)
    records=engine.load_data();qa=[];qb=[];ys=[];rows=[]
    for d in data['details']:
        key=(d['id'],d['team']);saved=old[key]
        lineup=[r['player'] for r in sorted(rosters[key],key=lambda r:r['expected_position'])]
        assert lineup==saved['lineup']
        result=engine.forecast(records,d['team'],lineup,saved['venue'],d['date'],
                               half_life=2,kappa=20,n_sims=3000,seed=7)
        q=[p['q'] for p in result['players']];y=d['observed_shares']
        loss=share_loss([q],[y])[0]
        assert abs(loss-d['brier']['v2_same_cases'])<1e-12
        qa.append(d['q']);qb.append(q);ys.append(y)
        rows.append(dict(id=d['id'],team=d['team'],lineup=lineup,q_v2=q,q_v3=d['q'],y=y))
    return dict(n=len(rows),replay_matches_original_paired_losses=True,
                v2=calibration_bins(qb,ys),v3=calibration_bins(qa,ys),details=rows)


def ranking_report():
    import engine
    import ball_engine
    fixture=json.loads((ROOT/'data/fixture.json').read_text())
    lineups=fixture['lineups']; records=engine.load_data(); data=ball_engine.load_ball_data()
    inputs=dict(venue=fixture['venue'],cutoff=fixture['date'],half_life=2)
    cases={}; fits={}
    def add(name, result):
        scores=np.asarray(result['scores']); top=scores==scores.max(1)[:,None]
        settlements=top/top.sum(1)[:,None]
        chosen=[j for j,p in enumerate(result['players']) if p['player'] in ('Rohit Sharma','Shubman Gill','Virat Kohli','KL Rahul')]
        cases[name]=dict(players=[result['players'][j] for j in chosen],
                         q_standard_errors={result['players'][j]['player']:float(settlements[:,j].std(ddof=1)/np.sqrt(len(scores))) for j in chosen},
                         rohit_minus_gill=dict(difference=float(np.mean(settlements[:,0]-settlements[:,1])),
                             simulation_standard_error=float((settlements[:,0]-settlements[:,1]).std(ddof=1)/np.sqrt(len(scores)))))
    add('v2_default',engine.forecast(records,'India',lineups['India'],**inputs,kappa=20,n_sims=30000,seed=7))
    for label,venue,player in [('v3_default',True,True),('v3_no_venue',False,True),('v3_no_player',True,False)]:
        fit=ball_engine.fit_ball_model(data,lineups,**inputs,prior_strength=240,use_venue=venue,use_player=player)
        fits[label]=fit
        add(label,ball_engine.simulate_match(fit,'Unknown',n_sims=30000,seed=7)['teams']['India'])
    fit=fits['v3_default']
    add('v3_seed_19',ball_engine.simulate_match(fit,'Unknown',n_sims=30000,seed=19)['teams']['India'])
    for first in ('India','New Zealand'):
        add('v3_first_'+first,ball_engine.simulate_match(fit,first,n_sims=30000,seed=7)['teams']['India'])
    swapped={k:list(v) for k,v in lineups.items()};swapped['India'][0],swapped['India'][1]=swapped['India'][1],swapped['India'][0]
    swap_fit=ball_engine.fit_ball_model(data,swapped,**inputs,prior_strength=240)
    # Preserve original identities in the two-player difference summary below.
    swap_result=ball_engine.simulate_match(swap_fit,'Unknown',n_sims=30000,seed=7)['teams']['India']
    order=[swapped['India'].index(n) for n in lineups['India']]
    swap_result={**swap_result,'players':[swap_result['players'][i] for i in order],'scores':swap_result['scores'][:,order]}
    add('v3_opening_strike_swapped',swap_result)
    event=fit['outcomes'];cell_rows=[]
    for j in range(3):
        for phase in range(3):
            for innings in range(2):
                p=fit['pmfs']['India'][j,phase,innings]
                nonwide=np.dot(p,1-event[:,2])
                cell_rows.append(dict(player=lineups['India'][j],phase=phase,innings_order=innings+1,
                    expected_batter_runs_per_nonwide=float(np.dot(p,event[:,0])/nonwide),
                    raw_any_wicket_event_probability=float(np.dot(p,event[:,5]>0))))
    return dict(inputs={**inputs,'lineups':lineups,'v2_kappa_innings':20,'v3_kappa_deliveries':240,'overs':50,'sims':30000},
                cases=cases, fitted_delivery_cells=cell_rows,
                note='Sensitivity scenarios, not evidence of causation or out-of-time performance. Monte Carlo SE excludes parameter and model uncertainty.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--rankings',action='store_true');parser.add_argument('--matched',action='store_true');args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    result=dict(status='Independent arithmetic reproduction; exploratory descriptive diagnostics only',
                bin_edges=EDGES.tolist(),
                v2=artifact_report('validation_v2.json',lambda d:d['q']['selected'],'target_shares'),
                v3=artifact_report('ball_validation.json',lambda d:d['q'],'observed_shares'),
                hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'engine.py',ROOT/'ball_engine.py',ROOT/'data/validation_v2.json',ROOT/'data/ball_validation.json']})
    (OUT/'score_reproduction.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    if args.matched:
        paired=matched_calibration()
        (OUT/'matched_calibration.json').write_text(json.dumps(paired,indent=2,allow_nan=False))
    if args.rankings:
        ranking=ranking_report()
        (OUT/'ranking_trace.json').write_text(json.dumps(ranking,indent=2,allow_nan=False))
        print(json.dumps({name:[{key:p[key] for key in ('player','q','mean_runs','p_bat') if key in p} for p in case['players']] for name,case in ranking['cases'].items()},indent=2))
    print(json.dumps({key:{k:v for k,v in value.items() if k!='calibration'} for key,value in result.items() if key in ('v2','v3')},indent=2))


if __name__=='__main__':main()
