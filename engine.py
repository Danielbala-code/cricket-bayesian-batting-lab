"""Discounted empirical-Bayes categorical predictive demonstration.

Not calibrated betting probabilities. Shared observed early-wicket regimes and
batting-depth templates provide some dependence; scores remain conditionally
independent. Dirichlet posterior predictive integrates categorical probabilities.
"""
import json,pathlib,datetime
import numpy as np

# Explicit identity crosswalk, verified against the supplied registry-bearing data.
ALIASES={'Rohit Sharma':'RG Sharma','Virat Kohli':'V Kohli','Shreyas Iyer':'SS Iyer','Axar Patel':'AR Patel',
         'Tim Seifert':'TL Seifert','Will Young':'WA Young','Henry Nicholls':'HM Nicholls','Daryl Mitchell':'DJ Mitchell',
         'Tom Latham':'TWM Latham','Michael Bracewell':'MG Bracewell','Mitchell Santner':'MJ Santner',
         'Nathan Smith':'NG Smith','Kyle Jamieson':'KA Jamieson','Jacob Duffy':'JA Duffy'}

def load_data():
    return json.loads((pathlib.Path(__file__).parent/'data/innings.json').read_text(encoding='utf-8'))

def decay_weight(date,cutoff,half_life=4):
    age=(datetime.date.fromisoformat(str(cutoff)[:10])-datetime.date.fromisoformat(str(date)[:10])).days/365.25
    return 2**(-age/half_life)

def winner_shares(scores):
    s=np.asarray(scores);top=s.max(axis=1);ties=(s==top[:,None])&(s>=0)
    sizes=ties.sum(axis=1)
    return np.divide(ties,sizes[:,None],where=sizes[:,None]>0,out=np.zeros_like(s,dtype=float)).mean(axis=0),ties.mean(axis=0)

def _role(pos):return 0 if 1<=pos<=3 else 1 if 4<=pos<=7 else 2 if 8<=pos<=11 else None

def forecast(records,team,lineup,venue,cutoff,half_life=4,kappa=20,n_sims=10000,seed=7,scenario='Mixed',use_venue=True,use_regime=True):
    cutoff=str(cutoff)[:10]
    if len(lineup)!=11 or len(set(lineup))!=11:raise ValueError('Provide exactly eleven distinct players in expected batting order.')
    canonical=[ALIASES.get(name,name) for name in lineup]
    if len(set(canonical))!=11:raise ValueError('Lineup contains duplicate aliases for the same player.')
    if half_life<=0 or kappa<=0 or n_sims<1:raise ValueError('Half life, prior strength and simulations must be positive.')
    train=[r for r in records if r.get('eligible',False) and r['date']<cutoff]
    if not train:raise ValueError('No eligible history strictly before cutoff.')
    rng=np.random.default_rng(seed);warnings=[]
    weights=np.array([decay_weight(r['date'],cutoff,half_life) for r in train])
    venue_rows=[r for r in train if r['venue']==venue]
    target_nz=venue_rows[0]['host_nz'] if venue_rows else ('Hamilton' in venue or 'Seddon' in venue)
    if not venue_rows:warnings.append('Unseen ground: exact-ground prior falls back to broader location history.')
    by_name={p['name']:p['id'] for r in train for p in r['players']}
    # Every category from zero to observed maximum has a small positive prior.
    maximum=max([300]+[p['runs'] for r in train for p in r['players'] if p.get('runs') is not None])
    support=np.arange(maximum+1);base=np.exp(-support/25);base/=base.sum()
    trouble=np.array([bool(r['early_trouble']) for r in train]);team_mask=np.array([r['team']==team for r in train])
    location=np.array([r['host_nz']==target_nz for r in train]);ground=np.array([r['venue']==venue for r in train])
    def context_mean(values,mask,default):
        w=weights*mask;return (float(np.dot(w,values))+kappa*default)/(w.sum()+kappa)
    prior_rate=float(np.dot(weights,trouble)/weights.sum())
    if use_venue:
        prior_rate=context_mean(trouble,location,prior_rate);prior_rate=context_mean(trouble,ground,prior_rate)
    rate=context_mean(trouble,team_mask,prior_rate)
    label=str(scenario).lower()
    if label in ('early trouble','trouble','early wickets'):rate=1.0
    elif label in ('settled','settled start','steadier start','good start','no early trouble'):rate=0.0
    if not use_regime:rate=0.0
    regimes=rng.random(n_sims)<rate
    scores=np.full((n_sims,11),-1,dtype=int)
    # Conditional empirical batting depth; same draw gates all players in a row.
    depth=np.array([min(11,max([p.get('position') or 0 for p in r['players']])) for r in train])
    for regime in (False,True):
        rows=np.flatnonzero(regimes==regime)
        if not len(rows):continue
        mask=(trouble==regime) if use_regime else np.ones(len(train),dtype=bool);dcounts=np.bincount(depth,weights=weights*mask,minlength=12)[0:12]
        base_depth=np.bincount(depth,weights=weights,minlength=12)[0:12];base_depth[:2]=0;base_depth/=base_depth.sum()
        dprior=(dcounts+kappa*base_depth)/(dcounts.sum()+kappa)
        for ctx in ([location,ground] if use_venue else []):
            dc=np.bincount(depth,weights=weights*mask*ctx,minlength=12)[0:12];dprior=(dc+kappa*dprior)/(dc.sum()+kappa)
        dc=np.bincount(depth,weights=weights*mask*team_mask,minlength=12)[0:12];dprior=(dc+kappa*dprior)/(dc.sum()+kappa)
        depths=rng.choice(np.arange(12),size=len(rows),p=dprior)
        for j,name in enumerate(lineup):
            pid=by_name.get(canonical[j],canonical[j]);role=_role(j+1)
            global_counts=np.zeros(maximum+1);country_counts=global_counts.copy();ground_counts=global_counts.copy();own=global_counts.copy();n=0;wn=0
            for i,r in enumerate(train):
                if use_regime and bool(r['early_trouble'])!=regime:continue
                for p in r['players']:
                    if p.get('position') is None or p.get('runs') is None or _role(p['position'])!=role:continue
                    value=p['runs'];w=weights[i]
                    if p['id']==pid:
                        own[value]+=w;n+=1;wn+=w
                    else:
                        global_counts[value]+=w
                        if location[i]:country_counts[value]+=w
                        if ground[i]:ground_counts[value]+=w
            prior=(global_counts+.5*base)/(global_counts.sum()+.5)
            if use_venue:
                prior=(country_counts+kappa*prior)/(country_counts.sum()+kappa)
                prior=(ground_counts+kappa*prior)/(ground_counts.sum()+kappa)
            pmf=(own+kappa*prior)/(own.sum()+kappa)
            values=rng.choice(support,size=len(rows),p=pmf)
            scores[rows[depths>=j+1],j]=values[depths>=j+1]
    q,p_top=winner_shares(scores);players=[]
    for j,name in enumerate(lineup):
        pid=by_name.get(canonical[j],canonical[j]);hist=[(r,p) for r in train for p in r['players'] if p['id']==pid and p.get('position') and _role(p['position'])==_role(j+1) and p.get('runs') is not None]
        observed=scores[:,j]>=0;values=scores[observed,j]
        if not hist:warnings.append('Cold start for '+name+': pooled role/context prior used.')
        players.append(dict(player=name,canonical_name=canonical[j],player_id=pid,position=j+1,mean_runs=float(values.mean()) if len(values) else 0.,median_runs=float(np.median(values)) if len(values) else 0.,
                            p50plus=float(np.mean(scores[:,j]>=50)),p_top=float(p_top[j]),q=float(q[j]),history_n=len(hist),
                            weighted_n=float(sum(decay_weight(r['date'],cutoff,half_life) for r,p in hist)),p_bat=float(observed.mean())))
    warnings.append('Exploratory empirical-Bayes model; probabilities are uncalibrated. Scores conditionally independent within shared regime/depth.')
    warnings.append('Training excludes no-results and D/L. Actual historical roles are matched to user expected roles; venue location is NZ versus outside NZ.')
    return dict(players=players,scenario_weights=({'Early trouble':rate,'Settled start':1-rate} if use_regime else {'Unconditioned':1.0}),scores=scores,warnings=warnings,
                training_max_date=max(r['date'] for r in train),training_n=len(train),model='Discounted Dirichlet posterior predictive with shared regime and empirical depth')
