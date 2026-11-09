"""Frozen retrospective predictive distribution diagnostics; never recalibrates."""
import argparse, collections, hashlib, json, pathlib, statistics, time
import numpy as np
from engine import load_data, forecast

ROOT=pathlib.Path(__file__).parent

def role(pos):
    return 'DNB' if pos is None else 'top 1-3' if pos<=3 else 'middle 4-7' if pos<=7 else 'lower 8-11'

def calibration(rows,prob,outcome):
    bins=[]
    cluster_ids=sorted({r['match_id'] for r in rows});rng=np.random.default_rng(1129)
    bootstrap_indices=rng.integers(0,len(cluster_ids),size=(500,len(cluster_ids)))
    for lo,hi in ((0,.1),(.1,.3),(.3,1.0)):
        rs=[r for r in rows if lo<=r[prob] and (r[prob]<hi or hi==1)]
        n=len(rs);p=float(np.mean([r[outcome] for r in rs])) if rs else None
        if n:
            z=1.959963984540054;den=1+z*z/n;center=(p+z*z/(2*n))/den
            spread=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den;wilson=[max(0,center-spread),min(1,center+spread)]
            totals=np.array([sum(r['match_id']==cid for r in rs) for cid in cluster_ids]);hits=np.array([sum(r[outcome] for r in rs if r['match_id']==cid) for cid in cluster_ids])
            bn=totals[bootstrap_indices].sum(axis=1);bh=hits[bootstrap_indices].sum(axis=1)
            boot=(bh[bn>0]/bn[bn>0]);cluster_interval=np.quantile(boot,[.025,.975]).tolist() if len(boot) else None
        else:wilson=None;cluster_interval=None
        bins.append(dict(lower=lo,upper=hi,n=n,n_matches=len({r['match_id'] for r in rs}),mean_prediction=float(np.mean([r[prob] for r in rs])) if rs else None,
                         observed_rate=p,observed_wilson95=wilson,observed_match_cluster_bootstrap95=cluster_interval,sparse=n<30))
    return dict(bins=bins,brier=float(np.mean([(r[prob]-r[outcome])**2 for r in rows])),
                ece=sum(b['n']*abs(b['mean_prediction']-b['observed_rate']) for b in bins if b['n'])/len(rows),
                log_loss=float(np.mean([-r[outcome]*np.log(np.clip(r[prob],1e-12,1-1e-12))-(1-r[outcome])*np.log(np.clip(1-r[prob],1e-12,1)) for r in rows])),
                note='Fixed bins [0,.1),[.1,.3),[.3,1]. Wilson95 is a marginal binomial diagnostic; player rows within matches are dependent. Match-cluster bootstrap95 resamples whole match ids500 times; it does not capture series dependence. No normality assumption for runs. Log-loss clipped at1e-12 only for numerical reporting.')

def summary(rows):
    n=len(rows); b=[r for r in rows if r['observed_bat']]; expected_b=sum(r['p_bat'] for r in rows)
    def avg(key,rs=rows):return float(np.mean([r[key] for r in rs])) if rs else None
    def cond(key):return sum(r[key] for r in rows)/expected_b if expected_b else None
    return dict(n=n,n_batted=len(b),n_dnb=n-len(b),p_bat_predicted=avg('p_bat'),p_bat_observed=len(b)/n if n else None,
                zero_conditional_bat_predicted=cond('p_zero_and_bat'),zero_conditional_bat_observed=avg('observed_zero',b),
                p50_conditional_bat_predicted=cond('p50'),p50_conditional_bat_observed=avg('observed_50',b),
                p100_conditional_bat_predicted=cond('p100'),p100_conditional_bat_observed=avg('observed_100',b),
                zero_and_bat_predicted=avg('p_zero_and_bat'),zero_and_bat_observed=sum(r['observed_zero'] for r in rows)/n if n else None,
                central80_coverage=avg('covered80',b),central80_mean_width=avg('width80',b),
                conditional_mean_runs_mae=avg('absolute_mean_error',b),conditional_runs_crps=avg('crps',b),
                observed_median_runs=float(np.median([r['observed_runs'] for r in b])) if b else None,
                predicted_median_runs=float(np.median([r['predicted_median_conditional_bat'] for r in b])) if b else None)

def settings(args):
    path=ROOT/'data/tuning_v2.json'
    if args.h is not None and args.k is not None:return args.h,args.k,'explicit CLI settings'
    if not path.exists():raise ValueError('tuning_v2.json absent: provide explicit --h and --k; no implicit final-data tuning')
    data=json.loads(path.read_text(encoding='utf-8'))
    selected=data.get('selected_config',data.get('selected',data.get('best_config',{})))
    h=selected.get('half_life_years',selected.get('half_life',selected.get('h')))
    k=selected.get('kappa',selected.get('k'))
    if h is None or k is None:raise ValueError('Selected tuning config lacks recognized half-life/kappa keys')
    return float(h),float(k),'data/tuning_v2.json selected development settings'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--h',type=float);ap.add_argument('--k',type=float)
    ap.add_argument('--max-targets',type=int,default=40);ap.add_argument('--sims',type=int,default=4000)
    ap.add_argument('--venue-blind',action='store_true');args=ap.parse_args();h,k,origin=settings(args)
    records=sorted(load_data(),key=lambda r:(r['date'],r['id'],r['team']))
    eligible=[r for r in records if r['eligible']]
    targets=[r for r in eligible if r['date']>='2023-01-01' and r['team'] in ('India','New Zealand') and len(r['players'])==11]
    all_target_count=len(targets);targets=targets[-args.max_targets:] if args.max_targets else targets
    player_rows=[];team_rows=[];started=time.time()
    for number,t in enumerate(targets):
        hist=[r for r in eligible if r['date']<t['date']]
        by_id=collections.defaultdict(list)
        for r in hist:
            for p in r['players']:
                if p.get('position') is not None:by_id[p['id']].append(p['position'])
        def expected(p):return statistics.median(by_id[p['id']][-5:]) if by_id[p['id']] else 6
        ordered=sorted(t['players'],key=lambda p:(expected(p),p['name']));names=[p['name'] for p in ordered]
        f=forecast(records,t['team'],names,t['venue'],t['date'],half_life=h,kappa=k,n_sims=args.sims,seed=7,use_venue=not args.venue_blind)
        assert f['training_max_date']<t['date'];sim=np.asarray(f['scores']);assert sim.shape==(args.sims,11)
        actual={p['name']:p for p in t['players']}; observed=[p for p in t['players'] if p['runs'] is not None]
        top=max(p['runs'] for p in observed);winners={p['name'] for p in observed if p['runs']==top}
        q=np.array([p['q'] for p in f['players']]);y=np.array([1/len(winners) if name in winners else 0 for name in names])
        assert abs(q.sum()-1)<1e-8 and abs(y.sum()-1)<1e-8
        team_rows.append(dict(id=t['id'],date=t['date'],team=t['team'],venue=t['venue'],q_brier=float(np.square(q-y).sum()),
                              training_max_date=f['training_max_date'],lineup=names))
        for j,pred in enumerate(f['players']):
            name=pred['player'];a=actual[name];s=sim[:,j];active=s>=0;xs=np.sort(s[active].astype(float));n=len(xs)
            assert n>0,'No active draws for '+name
            lo,hi=np.quantile(xs,[.1,.9]);ob=a['runs'];b=ob is not None
            # Empirical CRPS = E|X-y| - .5 E|X-X'|; includes discreteness and skew.
            crps=float(np.mean(np.abs(xs-ob))-np.dot(2*np.arange(1,n+1)-n-1,xs)/(n*n)) if b else None
            r=dict(match_id=t['id'],date=t['date'],team=t['team'],player=name,expected_position=j+1,expected_role=role(j+1),
                   realised_position=a.get('position'),realised_role=role(a.get('position')),observed_runs=ob,observed_bat=int(b),
                   observed_zero=int(b and ob==0),observed_50=int(b and ob>=50),observed_100=int(b and ob>=100),observed_top=int(name in winners),
                   observed_top_share=1/len(winners) if name in winners else 0,p_bat=float(active.mean()),p_zero_and_bat=float(np.mean(s==0)),
                   p50=float(np.mean(s>=50)),p100=float(np.mean(s>=100)),p_top=float(pred['p_top']),q=float(pred['q']),
                   predicted_mean_conditional_bat=float(xs.mean()),predicted_median_conditional_bat=float(np.median(xs)),
                   lower10_conditional_bat=float(lo),upper90_conditional_bat=float(hi),
                   covered80=int(lo<=ob<=hi) if b else None,width80=float(hi-lo),
                   absolute_mean_error=abs(float(xs.mean())-ob) if b else None,crps=crps)
            assert b==(a.get('position') is not None),'Observed DNB/runs mismatch'
            assert 0<=r['p_zero_and_bat']<=r['p_bat']<=1 and 0<=r['p100']<=r['p50']<=r['p_bat']
            assert not b or crps>=-1e-9
            player_rows.append(r)
        if number%10==0:print('distribution checks',number+1,'/',len(targets),flush=True)
    assert len(player_rows)==11*len(targets)
    out=dict(status='Retrospective diagnostic, manually reviewed with caveats; not untouched holdout, not evidence of calibration or betting edge',
             n_team_innings=len(targets),n_player_rows=len(player_rows),all_eligible_target_count=all_target_count,
             min_date=min(r['date'] for r in targets),max_date=max(r['date'] for r in targets),
             settings=dict(half_life_years=h,kappa=k,n_sims=args.sims,max_targets=args.max_targets,use_venue=not args.venue_blind,origin=origin,seed_policy='fixed seed7 for each target'),
             aggregate=summary(player_rows),by_expected_role={g:summary([r for r in player_rows if r['expected_role']==g]) for g in ('top 1-3','middle 4-7','lower 8-11')},
             by_realised_role={g:summary([r for r in player_rows if r['realised_role']==g]) for g in ('top 1-3','middle 4-7','lower 8-11','DNB')},
             by_team={g:summary([r for r in player_rows if r['team']==g]) for g in ('India','New Zealand')},
             by_year={g:summary([r for r in player_rows if r['date'][:4]==g]) for g in sorted({r['date'][:4] for r in player_rows})},
             calibration={'p50':calibration(player_rows,'p50','observed_50'),'p_top':calibration(player_rows,'p_top','observed_top')},
             mean_q_brier=float(np.mean([r['q_brier'] for r in team_rows])),
             accounting={'n_batted':sum(r['observed_bat'] for r in player_rows),'n_dnb':sum(not r['observed_bat'] for r in player_rows),
                         'n_batted_zero':sum(r['observed_zero'] for r in player_rows),'all_probability_checks_passed':True,'q_and_targets_sum_to_one':True},
             definitions={'conditional_bat_rates':'Predicted aggregate rate=sum(event probability)/sum(Pbat); observed=sum(event)/number actually batted. DNB is excluded from the conditional denominator and never entered as run zero.',
                          'score_diagnostics':'MAE, CRPS and inclusive central10th/90th percentile coverage apply to observed batters, using simulation scores conditional on batting; discrete intervals need not attain exactly80%.',
                          'medians':'observed_median_runs is median actual runs among observed batters. predicted_median_runs is the median of their individual conditional predictive medians; it is not the median of a pooled predictive mixture.',
                          'roles':'Expected role uses median of last5 strictly earlier observed positions with fallback6, then lexical names. Realised roles are diagnostic slices only; never forecast inputs.',
                          'p50_calibration':'Unconditional event: scored50+ in the team innings; DNB outcome false, distinct from batted zero.',
                          'p_top_calibration':'Binary membership of tied top set; not normalized. q Brier separately uses fractional1/k target.',
                          'target_eligibility':'Same eligible flag as engine/evaluation; India/NZ records with11 listed players and dates>=2023; chronological final40 by default. Historical recorded roster is an oracle assumption.'},
             limitations=['No recalibration, model selection or changes fitted on retrospective outcomes.','Previously explored retrospective data; cannot certify calibration from this bounded dependent sample.',
                          'Calibration intervals resample match clusters; series dependence and hyperparameter selection uncertainty remain unmodeled. Small bins and realised-role slices are descriptive.','Intervals are predictive runs conditional on batting, not posterior uncertainty intervals for mean ability.',
                          'Forecast simulation integrates categorical PMFs; DNB and a genuine batted zero remain distinct.'],
             engine_sha256=hashlib.sha256((ROOT/'engine.py').read_bytes()).hexdigest(),runtime_seconds=time.time()-started,
             team_details=team_rows,details=player_rows)
    path=ROOT/'data/distribution_checks_v2.json';path.write_text(json.dumps(out,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:out[k] for k in ('n_team_innings','settings','aggregate','mean_q_brier','runtime_seconds')},indent=2))

if __name__=='__main__':main()
