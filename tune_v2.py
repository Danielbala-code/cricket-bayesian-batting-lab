"""Predeclared bounded chronological tuning; engine is read-only."""
import json,pathlib,datetime,time,hashlib,statistics,collections,sys
import numpy as np
from engine import load_data,forecast

BASE=pathlib.Path(__file__).parent
GRID=[{'half_life_years':h,'kappa':k} for h in (2,4,'infinity') for k in (5,20)]
SEED=7;SIMS=4000;DEVELOPMENT_CAP=60
def numeric_h(h):return float('inf') if h=='infinity' else h
def key(r):return r['id']+'|'+r['team']
def split(records):
    eligible=sorted([r for r in records if r['eligible'] and r['team'] in ('India','New Zealand') and len(r['players'])==11],key=lambda r:(r['date'],r['id'],r['team']))
    development=[r for r in eligible if '2019-01-01'<=r['date']<'2023-01-01']
    if len(development)>DEVELOPMENT_CAP:development=[development[i] for i in np.linspace(0,len(development)-1,DEVELOPMENT_CAP,dtype=int)]
    test=[r for r in eligible if r['date']>='2023-01-01']
    assert not ({key(r) for r in development}&{key(r) for r in test})
    assert max(r['date'] for r in development)<min(r['date'] for r in test)
    assert len({key(r) for r in development})==len(development)
    return development,test
def inputs(records,t):
    history=[r for r in records if r['eligible'] and r['date']<t['date']]
    by_id=collections.defaultdict(list)
    for r in sorted(history,key=lambda r:(r['date'],r['id'])):
        for p in r['players']:
            if p['position'] is not None and p['position']<=11:by_id[p['id']].append(p['position'])
    lineup=[p['name'] for p in sorted(t['players'],key=lambda p:(statistics.median(by_id[p['id']][-5:]) if by_id[p['id']] else 6,p['name']))]
    b=[p for p in t['players'] if p['runs'] is not None];mx=max(p['runs'] for p in b);winners=[p['name'] for p in b if p['runs']==mx]
    y=np.array([1/len(winners) if n in winners else 0 for n in lineup])
    baseline=collections.Counter()
    for r in history:
        if r['team']!=t['team']:continue
        bs=[p for p in r['players'] if p['runs'] is not None]
        if not bs:continue
        top=max(p['runs'] for p in bs);ws=[p['name'] for p in bs if p['runs']==top]
        for n in ws:baseline[n]+=1/len(ws)
    qh=np.array([baseline[n]+1/11 for n in lineup]);qh/=qh.sum()
    assert all(r['date']<t['date'] for r in history)
    return lineup,y,qh
def predict(records,t,lineup,config,sims,use_venue=True,use_regime=True):
    result=forecast(records,t['team'],lineup,t['venue'],t['date'],half_life=numeric_h(config['half_life_years']),kappa=config['kappa'],n_sims=sims,seed=SEED,use_venue=use_venue,use_regime=use_regime)
    assert result['training_max_date']<t['date']
    q=np.array([p['q'] for p in result['players']]);assert np.isclose(q.sum(),1)
    return q,result['training_max_date']
def summary(details):
    names=list(details[0]['brier']);return {name:float(np.mean([r['brier'][name] for r in details])) for name in names}
def paired_bootstrap(details,baseline):
    grouped=collections.defaultdict(list)
    for r in details:grouped[r['id']].append(r['brier']['selected']-r['brier'][baseline])
    a=np.array([[sum(v),len(v)] for v in grouped.values()]);rng=np.random.default_rng(20261109)
    w=rng.multinomial(len(a),np.repeat(1/len(a),len(a)),size=2000);s=w@a;d=s[:,0]/s[:,1]
    return dict(delta_selected_minus_baseline=float(a[:,0].sum()/a[:,1].sum()),p025=float(np.quantile(d,.025)),p975=float(np.quantile(d,.975)),
                bootstrap='2000 unique-game resamples preserving all selected team cases per game; seed20261109; series/era dependence unmodeled')
def run(plan_only=False,resume=False):
    records=load_data();dev,test=split(records)
    plan=dict(status='PLAN FROZEN BEFORE FIT',plan_frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              grid=GRID,selection_metric='Mean per-team-innings multiclass Brier: sum over11 players of(q-y)^2; lower better',
              development_window=['2019-01-01','2022-12-31'],test_window=['2023-01-01',max(r['date'] for r in records)],
              development_cap=60,cap_method='Sorted date,id,team then numpy.linspace integer indices spanning entire development period',
              development_available_n=sum(r['eligible'] and r['team'] in ('India','New Zealand') and len(r['players'])==11 and '2019-01-01'<=r['date']<'2023-01-01' for r in records),
              development_n=len(dev),development_unique_games=len({r['id'] for r in dev}),test_n=len(test),test_unique_games=len({r['id'] for r in test}),
              development_keys=[key(r) for r in dev],test_keys=[key(r) for r in test],
              seed=SEED,n_sims=SIMS,runtime_rule='If projected total grid runtime exceeds300seconds based only on first forecast timing, use2000simulations for ALL candidates before inspecting any score; never inspect partial Brier to choose cases',
              order_policy='Recorded target XI assumed known. Expected order sorted by median of last5 strictlyprior observed batting positions1..11; unknownfallback6; lexical name tiebreak. No current realised order used.',
              comparability='v2 order policy differs from old all-history-median/fallback11; recompute4y/k20 baseline on identical v2 lineups, cases and simulation count',
              data_sha256=hashlib.sha256((BASE/'data/innings.json').read_bytes()).hexdigest(),engine_sha256=hashlib.sha256((BASE/'engine.py').read_bytes()).hexdigest(),
              no_untouched_claim='Retrospective exploratory development/test split; prior full-history analyses and previous grids have been seen. No calibration or betting-edge claim.')
    path=BASE/'data/tuning_v2.json'
    if path.exists():
        existing=json.loads(path.read_text())
        # --plan writes first; subsequent --run must reuse that exact declaration.
        if existing.get('status')=='PLAN FROZEN BEFORE FIT' or resume:plan=existing
    if resume:
        assert 'selected_config' in plan and 'grid_results' in plan
        plan['engine_original_sha256']=plan['engine_sha256']
        plan['engine_sha256']=hashlib.sha256((BASE/'engine.py').read_bytes()).hexdigest()
        plan['authorized_engine_change']='Root added optionaluse_regime=False ablation; defaultTrue mathematics unchanged and regression-tested. Original grid results and selection timestamp preserved. First test attempt stopped at source-hash guard before results were written or inspected.'
        plan['final_test_ablation_added_before_resumed_test']='selected_no_regime; diagnostic only, never a grid/selection candidate'
        plan['test_resume_declared_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    path.write_text(json.dumps(plan,indent=2),encoding='utf-8')
    print(json.dumps({k:plan[k] for k in ('status','plan_frozen_utc','development_n','development_unique_games','test_n','test_unique_games','grid')},indent=2),flush=True)
    if plan_only:return
    cached={key(t):inputs(records,t) for t in dev+test}
    # Cheap invariants exercise split boundaries and current-position non-use.
    synthetic=[{'eligible':True,'team':'India','id':str(i),'date':d,'players':[{}]*11} for i,d in enumerate(['2018-12-31','2019-01-01','2022-12-31','2023-01-01'])]
    sd,st=split(synthetic);assert len(sd)==2 and len(st)==1
    mutated={**dev[0],'players':[{**p,'position':11-(p['position'] or 0)} for p in dev[0]['players']]}
    assert inputs(records,mutated)[0]==cached[key(dev[0])][0]
    plan['self_tests_passed']=['Development2019-2022/test2023+ boundary split','No overlapping case keys','Deterministic cap unique indices','Current target realised positions do not affect expected order','Strict training date per forecast']
    if resume:
        selected=plan['selected_config'];results=plan['grid_results'];sims=plan['n_sims']
    else:
        first=dev[0];start=time.perf_counter();predict(records,first,cached[key(first)][0],GRID[0],SIMS);seconds=time.perf_counter()-start
        projected=seconds*len(dev)*len(GRID);sims=2000 if projected>300 else SIMS
        plan.update(first_forecast_seconds=seconds,projected_grid_seconds=projected,n_sims=sims,fit_started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        path.write_text(json.dumps(plan,indent=2),encoding='utf-8');print('Timing',seconds,'projected',projected,'sims',sims,flush=True)
        results=[]
        for config in GRID:
            cases=[]
            for t in dev:
                lineup,y,qh=cached[key(t)];q,mx=predict(records,t,lineup,config,sims)
                cases.append(dict(id=t['id'],date=t['date'],team=t['team'],lineup=lineup,training_max_date=mx,brier=float(np.square(q-y).sum()),q=q.tolist(),target_shares=y.tolist()))
            results.append(dict(config=config,mean_brier=float(np.mean([c['brier'] for c in cases])),n=len(cases),cases=cases))
        selected=min(results,key=lambda r:r['mean_brier'])['config']
        plan.update(status='SELECTED ON DEVELOPMENT ONLY; TEST NOT YET RUN',grid_results=results,selected_config=selected,selected_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    path.write_text(json.dumps(plan,indent=2),encoding='utf-8');print('SELECTED',selected,[(r['config'],r['mean_brier']) for r in results],flush=True)
    details=[]
    test_started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    for t in test:
        lineup,y,qh=cached[key(t)];qs,mx=predict(records,t,lineup,selected,sims)
        qb,_=predict(records,t,lineup,{'half_life_years':4,'kappa':20},sims)
        qv,_=predict(records,t,lineup,selected,sims,False)
        qn,_=predict(records,t,lineup,selected,sims,use_regime=False)
        predictions={'selected':qs,'old_4y_20_recomputed':qb,'selected_venue_blind':qv,'selected_no_regime':qn,'historical_top_share':qh,'uniform':np.repeat(1/11,11)}
        details.append(dict(id=t['id'],date=t['date'],year=t['date'][:4],team=t['team'],venue=t['venue'],host_nz=t['host_nz'],lineup=lineup,
                            training_max_date=mx,target_shares=y.tolist(),q={name:q.tolist() for name,q in predictions.items()},
                            brier={name:float(np.square(q-y).sum()) for name,q in predictions.items()}))
    folds=[]
    for team in ('India','New Zealand'):
        for year in sorted({r['year'] for r in details if r['team']==team}):
            ds=[r for r in details if r['team']==team and r['year']==year]
            folds.append(dict(team=team,year=year,n=len(ds),unique_games=len({r['id'] for r in ds}),mean_brier=summary(ds)))
    validation=dict(status='Exploratory retrospective test after development-only selection; not untouched validation',
                    selected_config=selected,n_sims=sims,seed=SEED,test_started_utc=test_started,selection_utc=plan['selected_utc'],
                    n=len(details),unique_games=len({r['id'] for r in details}),min_date=min(r['date'] for r in details),max_date=max(r['date'] for r in details),
                    mean_brier=summary(details),team_year_folds=folds,
                    paired_game_bootstrap={b:paired_bootstrap(details,b) for b in ('old_4y_20_recomputed','selected_venue_blind','selected_no_regime')},
                    strict_date_checks_passed=all(r['training_max_date']<r['date'] for r in details),
                    engine_unchanged=hashlib.sha256((BASE/'engine.py').read_bytes()).hexdigest()==plan['engine_sha256'],
                    engine_original_sha256=plan.get('engine_original_sha256',plan['engine_sha256']),engine_test_sha256=plan['engine_sha256'],
                    authorized_engine_change=plan.get('authorized_engine_change'),self_tests_passed=plan['self_tests_passed'],
                    order_policy=plan['order_policy'],comparability=plan['comparability'],limitations=[plan['no_untouched_claim'],
                    'XI composition is taken from recorded match roster and assumed known. Only expected order is reconstructed prior to match.',
                    'Test cases are all eligible2023+ India/NZ team innings with exactly11 roster players; no cap.',
                    'No-results/DL excluded. Six candidate development search incurs selection uncertainty. Monte Carlo differences remain.',
                    'Paired-game intervals preserve both teams per game but not series/era dependence. No calibration, causal or betting-edge claim.'],details=details)
    assert validation['engine_unchanged'];assert plan['selected_utc']<test_started
    (BASE/'data/validation_v2.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
    plan.update(status='DEVELOPMENT SELECTION AND SEPARATE RETROSPECTIVE TEST COMPLETE',test_completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    path.write_text(json.dumps(plan,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in validation.items() if k not in ('details','team_year_folds','limitations')},indent=2),flush=True)

if __name__=='__main__':run('--plan' in sys.argv,'--resume-test' in sys.argv)
