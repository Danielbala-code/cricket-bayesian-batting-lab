"""Frozen v3 comparison. All empirical inputs precede the match date."""
from pathlib import Path
import collections, datetime, hashlib, json, statistics, time, sys
import numpy as np
import engine

ROOT=Path(__file__).parent
GRID=[dict(half_life=h,prior_strength=k) for h in (2,4) for k in (60,240)]
DEV_SIMS=1500
TEST_SIMS=3000
SEED=7

def stamp(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def case_key(r): return str(r['id'])+'|'+r['team']

def supported_cases(records, metadata):
    lookup={(str(r['match_id']),r['team']):r for r in metadata['innings'] if r['innings_order'] in (1,2)}
    accepted=[];excluded=[]
    for t in records:
        if not(t['eligible'] and t['team'] in ('India','New Zealand') and len(t['players'])==11 and t['date']>='2019-01-01'):continue
        a=lookup.get((str(t['id']),t['team'])); b=lookup.get((str(t['id']),t['opponent']))
        reasons=[]
        for label,row in [('target',a),('opponent',b)]:
            if row is None: reasons.append(label+':missing metadata');continue
            if not row['eligible']:reasons.extend(label+':'+x for x in row['exclusions'])
            if row['scheduled_overs']!=50:reasons.append(label+':not scheduled50')
            if len(row['players'])!=11:reasons.append(label+':not11players')
        if reasons:excluded.append(dict(key=case_key(t),date=t['date'],reasons=reasons));continue
        accepted.append(t)
    accepted.sort(key=lambda t:(t['date'],t['id'],t['team']))
    dev=[t for t in accepted if t['date']<'2023-01-01']
    if len(dev)>60:dev=[dev[i] for i in np.linspace(0,len(dev)-1,60,dtype=int)]
    test=[t for t in accepted if t['date']>='2023-01-01']
    assert set(map(case_key,dev)).isdisjoint(map(case_key,test))
    assert max(t['date'] for t in dev)<min(t['date'] for t in test)
    return dev,test,lookup,excluded

def make_inputs(records,t,lookup):
    prior=sorted((r for r in records if r['eligible'] and r['date']<t['date']),key=lambda r:(r['date'],r['id']))
    positions=collections.defaultdict(list)
    for r in prior:
        for p in r['players']:
            if p['position'] is not None and 1<=p['position']<=11:positions[p['id']].append(p['position'])
    lineups={}
    for team in (t['team'],t['opponent']):
        ps=lookup[(str(t['id']),team)]['players']
        lineups[team]=[p['name'] for p in sorted(ps,key=lambda p:(statistics.median(positions[p['id']][-5:]) if positions[p['id']] else 6,p['name']))]
    first=next(team for team in lineups if lookup[(str(t['id']),team)]['innings_order']==1)
    truth={p['name']:p for p in t['players']}; bat=[p for p in t['players'] if p['runs'] is not None]
    mx=max(p['runs'] for p in bat); tops={p['name'] for p in bat if p['runs']==mx}
    y=np.array([1/len(tops) if name in tops else 0 for name in lineups[t['team']]])
    return lineups,first,y,truth

def score_q(result,team,y):
    q=np.array([p['q'] for p in result['teams'][team]['players']])
    assert np.isclose(q.sum(),1) and np.isclose(y.sum(),1)
    return q,float(np.square(q-y).sum())

def simulate(ball_data,t,args,config,sims,cache,use_venue=True,use_player=True):
    from ball_engine import forecast_match
    lineups,first,_,_=args
    if cache.get('_match_id')!=str(t['id']):
        cache.clear();cache['_match_id']=str(t['id'])
    key=(str(t['id']),tuple(config.items()),sims,use_venue,use_player)
    if key not in cache:
        cache[key]=forecast_match(ball_data,lineups=lineups,venue=t['venue'],cutoff=t['date'],batting_first=first,
                                 overs=50,n_sims=sims,seed=SEED,use_venue=use_venue,use_player=use_player,**config)
        assert cache[key]['training_max_date']<t['date']
    return cache[key]

def paired_interval(details, baseline):
    groups=collections.defaultdict(list)
    for d in details:groups[d['id']].append(d['brier']['v3']-d['brier'][baseline])
    a=np.array([[sum(v),len(v)] for v in groups.values()]);rng=np.random.default_rng(20261109)
    weights=rng.multinomial(len(a),np.full(len(a),1/len(a)),size=2000);draw=weights@a;d=draw[:,0]/draw[:,1]
    return dict(delta_v3_minus_comparator=float(a[:,0].sum()/a[:,1].sum()),p025=float(np.quantile(d,.025)),p975=float(np.quantile(d,.975)),
                assumption='Resample unique matches, preserving paired team cases; series/era dependence and selection uncertainty not resolved')

def distribution_checks(result,t,lineup,lookup):
    tr=result['teams'][t['team']];scores=np.asarray(tr['scores']);balls=np.asarray(tr['balls_faced'])
    batted=np.asarray(tr.get('batted',scores>=0),dtype=bool);truth={p['name']:p for p in lookup[(str(t['id']),t['team'])]['players']}
    out=[]
    for j,name in enumerate(lineup):
        p=truth[name];active=batted[:,j];s=scores[active,j];bf=balls[active,j]
        assert len(s)>0 and np.all(scores[~active,j]==-1)
        actual=p['runs']; actual_b=p['balls_faced'];was_batted=actual is not None
        q10,q90=np.quantile(s,[.1,.9]);b10,b90=np.quantile(bf,[.1,.9])
        out.append(dict(id=t['id'],date=t['date'],team=t['team'],player=name,expected_position=j+1,
            n_sims=len(scores),p_bat=float(active.mean()),p50=float(np.mean(scores[:,j]>=50)),p100=float(np.mean(scores[:,j]>=100)),
            p_zero_and_bat=float(np.mean(active&(scores[:,j]==0))),mean_balls_if_bats=float(bf.mean()),
            actual_runs=actual,actual_balls=actual_b,actual_batted=was_batted,
            run80_covered=bool(q10<=actual<=q90) if was_batted else None,
            balls80_covered=bool(b10<=actual_b<=b90) if was_batted else None,
            balls_mean_absolute_error=abs(float(bf.mean())-actual_b) if was_batted else None))
    return out

def run(resume_test=False):
    from ball_engine import load_ball_data
    records=engine.load_data();metadata=json.loads((ROOT/'data/ball_metadata.json').read_text());data=load_ball_data()
    dev,test,lookup,excluded=supported_cases(records,metadata)
    plan_path=ROOT/'data/ball_tuning.json';result_path=ROOT/'data/ball_validation.json'
    if plan_path.exists() and not resume_test:raise RuntimeError('v3 protocol already exists; retain results and explicitly version a new run rather than overwrite selection')
    plan=dict(status='FROZEN BEFORE FIT',created_utc=stamp(),grid=GRID,development_n=len(dev),test_n=len(test),
              development_keys=list(map(case_key,dev)),test_keys=list(map(case_key,test)),excluded_cases=excluded,
              development_sims=DEV_SIMS,test_sims=TEST_SIMS,seed=SEED,primary='Mean fractional-outcome multiclass Brier, lower better',
              temporal_rule='Strict earlier calendar dates; expected order from last5 prior appearances; recorded rosters/innings order assumed known post-toss',
              target_rule='Simulated first innings supplies chase target; actual first score/duration never input',
              cohort='Same supported paired cases for v2/v3; exclusions use unsupported outcomes and narrow retrospective population, not knowable pre-match',
              no_untouched_claim='All retrospective/exploratory; previous final-period results informed model development',
              engine_sha256=hashlib.sha256((ROOT/'ball_engine.py').read_bytes()).hexdigest(),
              data_sha256=hashlib.sha256((ROOT/'data/ball_counts.npz').read_bytes()).hexdigest())
    if resume_test:
        plan=json.loads(plan_path.read_text(encoding='utf-8'))
        assert 'selected_config' in plan and 'grid_results' in plan
        assert hashlib.sha256((ROOT/'ball_engine.py').read_bytes()).hexdigest()==plan['engine_sha256']
        assert hashlib.sha256((ROOT/'data/ball_counts.npz').read_bytes()).hexdigest()==plan['data_sha256']
        plan['evaluation_resume_note']='Original final comparison reached provenance assertion with equal wall-clock timestamps from Windows15.625ms clock resolution; no final metrics were saved/inspected. Preserve earlier selection and rerun unchanged test only. Timestamp ordering now permits equal wall ticks and records monotonic ordering.'
        plan['test_resume_utc']=stamp()
    plan_path.write_text(json.dumps(plan,indent=2),encoding='utf-8');print('FROZEN',len(dev),'dev',len(test),'test',flush=True)
    cached={case_key(t):make_inputs(records,t,lookup) for t in dev+test}
    grid=[]
    for config in ([] if resume_test else GRID):
        start=time.time();cache={};cases=[]
        for t in dev:
            args=cached[case_key(t)];r=simulate(data,t,args,config,DEV_SIMS,cache)
            q,brier=score_q(r,t['team'],args[2]);cases.append(dict(key=case_key(t),brier=brier))
        grid.append(dict(config=config,mean_brier=float(np.mean([x['brier'] for x in cases])),cases=cases,runtime_seconds=time.time()-start))
        print('GRID',config,grid[-1]['mean_brier'],'seconds',grid[-1]['runtime_seconds'],flush=True)
    chosen=plan['selected_config'] if resume_test else min(grid,key=lambda x:x['mean_brier'])['config']
    assert hashlib.sha256((ROOT/'ball_engine.py').read_bytes()).hexdigest()==plan['engine_sha256'],'Engine changed during development; retain run and version any retry.'
    if not resume_test:plan.update(status='SELECTED ON DEVELOPMENT ONLY',selected_config=chosen,grid_results=grid,selected_utc=stamp())
    selection_loaded_monotonic_ns=time.perf_counter_ns()
    plan_path.write_text(json.dumps(plan,indent=2),encoding='utf-8');print('SELECTED',chosen,flush=True)
    start=stamp();test_started_monotonic_ns=time.perf_counter_ns();details=[];diagnostics=[];cache={}
    for number,t in enumerate(test):
        args=cached[case_key(t)];lineups,first,y,_=args
        result=simulate(data,t,args,chosen,TEST_SIMS,cache);q,brier=score_q(result,t['team'],y)
        no_venue=simulate(data,t,args,chosen,TEST_SIMS,cache,use_venue=False);_,bv=score_q(no_venue,t['team'],y)
        role=simulate(data,t,args,chosen,TEST_SIMS,cache,use_player=False);_,br=score_q(role,t['team'],y)
        old=engine.forecast(records,t['team'],lineups[t['team']],t['venue'],t['date'],half_life=2,kappa=20,n_sims=TEST_SIMS,seed=SEED)
        oq=np.array([p['q'] for p in old['players']]);bo=float(np.square(oq-y).sum())
        details.append(dict(id=t['id'],date=t['date'],team=t['team'],batting_first=first,
            brier=dict(v3=brier,v2_same_cases=bo,v3_without_venue=bv,v3_role_only=br,uniform=float(np.square(np.full(11,1/11)-y).sum())),
            q=q.tolist(),observed_shares=y.tolist(),training_max_date=result['training_max_date']))
        diagnostics.extend(distribution_checks(result,t,lineups[t['team']],lookup))
        if number%10==0:print('TEST',number+1,'/',len(test),flush=True)
    names=details[0]['brier'];means={name:float(np.mean([x['brier'][name] for x in details])) for name in names}
    def diagnostics_summary(rows):
        b=[x for x in rows if x['actual_batted']];exposure=sum(x['p_bat'] for x in rows)
        return dict(n=len(rows),n_batted=len(b),predicted_p_bat=float(np.mean([x['p_bat'] for x in rows])),observed_p_bat=len(b)/len(rows),
             predicted_50_if_bats=sum(x['p50'] for x in rows)/exposure,observed_50_if_bats=sum(x['actual_runs']>=50 for x in b)/len(b),
             predicted_100_if_bats=sum(x['p100'] for x in rows)/exposure,observed_100_if_bats=sum(x['actual_runs']>=100 for x in b)/len(b),
             runs80_coverage=float(np.mean([x['run80_covered'] for x in b])),balls80_coverage=float(np.mean([x['balls80_covered'] for x in b])),
             balls_mean_absolute_error=float(np.mean([x['balls_mean_absolute_error'] for x in b])))
    out=dict(status='Exploratory chronological v3 comparison, not calibrated or untouched confirmation',selected_config=chosen,
        selection_utc=plan['selected_utc'],test_started_utc=start,n=len(details),unique_games=len({d['id'] for d in details}),
        min_date=min(d['date'] for d in details),max_date=max(d['date'] for d in details),mean_brier=means,
        paired_game_bootstrap={name:paired_interval(details,name) for name in names if name!='v3'},
        distribution=diagnostics_summary(diagnostics),by_team={t:diagnostics_summary([d for d in diagnostics if d['team']==t]) for t in ('India','New Zealand')},
        folds=[dict(year=y,team=t,n=len(ds),mean_brier={name:float(np.mean([x['brier'][name] for x in ds])) for name in names})
               for y,t in sorted({(d['date'][:4],d['team']) for d in details})
               if (ds:=[x for x in details if x['date'][:4]==y and x['team']==t])],
        exclusions=excluded,strict_date_checks_passed=all(d['training_max_date']<d['date'] for d in details),
        caveats=['Recorded rosters and post-toss innings order assumed known; source vintage unavailable',
                 'Unsupported events excluded retrospectively, narrowing population; no claim to all-match coverage',
                 'Scheduled50 overs only evaluated; user shorter limits are unvalidated scenarios, not DLS/weather forecasts',
                 'Previous evaluations and full-history summaries have been seen; no profit, causal or calibration claim'],
        details=details,player_diagnostics=diagnostics)
    out['selection_order_audit']=dict(selection_loaded_monotonic_ns=selection_loaded_monotonic_ns,test_started_monotonic_ns=test_started_monotonic_ns,
        wall_clock_resolution_seconds=time.get_clock_info('time').resolution,logical_order_passed=selection_loaded_monotonic_ns<test_started_monotonic_ns)
    assert out['strict_date_checks_passed'] and out['selection_utc']<=out['test_started_utc'] and out['selection_order_audit']['logical_order_passed']
    assert hashlib.sha256((ROOT/'ball_engine.py').read_bytes()).hexdigest()==plan['engine_sha256'],'Engine changed during evaluation; do not present mixed-version results.'
    result_path.write_text(json.dumps(out,indent=2,allow_nan=False),encoding='utf-8')
    plan.update(status='RETROSPECTIVE COMPARISON COMPLETE',completed_utc=stamp());plan_path.write_text(json.dumps(plan,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k in ('n','unique_games','mean_brier','paired_game_bootstrap','distribution')},indent=2))

if __name__=='__main__':run(resume_test='--resume-test' in sys.argv)
