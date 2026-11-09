"""Delivery-level empirical-Bayes model with shared legal-ball resources.

Probabilities are posterior-predictive categorical means, not a full joint
hierarchical posterior. Parameters are in DELIVERIES. No DLS/weather process,
bowling identities or target-pressure response is fitted.
"""
import datetime,json,pathlib
import numpy as np
from engine import ALIASES,winner_shares

# Fixed law-valid support, independent of the archive's future vocabulary.
BASE_OUTCOMES=((0,0,0,0,0,0),(1,0,0,0,1,0),(2,0,0,0,0,0),
               (3,0,0,0,1,0),(4,0,0,0,0,0),(6,0,0,0,0,0),
               (0,1,1,0,0,0),(0,1,0,1,0,0),(0,0,0,0,0,1))
NZ_CITIES=('Hamilton','Auckland','Napier','Wellington','Christchurch','Mount Maunganui','Tauranga','Nelson','Dunedin','Whangarei','Queenstown')

def _event_tuple(event):
    event=tuple(int(x) for x in event)
    return event if len(event)==7 else event+(event[0] if event[0] in (4,6) else 0,)

def load_ball_data():
    directory=pathlib.Path(__file__).parent/'data'
    metadata=json.loads((directory/'ball_metadata.json').read_text(encoding='utf-8'))
    with np.load(directory/'ball_counts.npz',allow_pickle=False) as archive:
        metadata['counts']=archive['counts'].copy()
    return metadata

def _role(position):return 0 if 1<=position<=3 else 1 if 4<=position<=7 else 2 if 8<=position<=11 else None

def fit_ball_model(data,lineups,venue,cutoff,half_life=2,prior_strength=120,use_venue=True,use_player=True):
    """Fit strict-past PMFs indexed [expected player, phase, innings order, event].

    For each role/phase/order, exclude the target player's counts from every
    context. Background counts plus a symmetric0.5 pseudo-count base start the
    prior. Disjoint NZ-other and ground counts update sequentially with kappa.
    Empty context cells fall back to that context's role aggregate. Own matching
    role/phase/order counts enter exactly once in the final update.
    """
    cutoff=str(cutoff)[:10]
    if len(lineups)!=2 or any(len(v)!=11 for v in lineups.values()):raise ValueError('Two expected XIs of eleven players are required.')
    if half_life<=0 or prior_strength<=0:raise ValueError('Half life and prior strength must be positive.')
    metadata=data.get('metadata',data);counts=np.asarray(data['counts'],dtype=np.int64)
    innings=metadata['innings'];people=metadata['players'];catalog=metadata['outcomes']
    past=np.array([bool(r.get('eligible')) and r['date']<cutoff for r in innings])
    keep=past[counts[:,0]] if len(counts) else np.zeros(0,dtype=bool)
    rows=counts[keep & (counts[:,5]>0)]
    if not len(rows):raise ValueError('No eligible delivery history strictly before cutoff.')
    valid_innings=np.unique(rows[:,0]);training_max=max(innings[i]['date'] for i in valid_innings)
    assert training_max<cutoff
    support=sorted({_event_tuple(x) for x in BASE_OUTCOMES}|{_event_tuple(catalog[i]) for i in np.unique(rows[:,4])})
    outcomes=np.array(support,dtype=np.int64);support_index={event:i for i,event in enumerate(support)}
    oi=np.array([support_index[_event_tuple(catalog[i])] for i in rows[:,4]],dtype=int)
    k=len(support);strength=float(prior_strength)
    ages=np.array([(datetime.date.fromisoformat(cutoff)-datetime.date.fromisoformat(r['date'])).days/365.25 if past[i] else 0. for i,r in enumerate(innings)])
    discount=2.**(-ages/half_life);weighted=rows[:,5]*discount[rows[:,0]]
    order=np.array([int(r.get('innings_order',1))-1 for r in innings])[rows[:,0]]
    if np.any((order<0)|(order>1)):raise ValueError('Eligible delivery history has invalid innings order.')
    is_nz=any(city.lower() in venue.lower() for city in NZ_CITIES)
    old_ground=[innings[i] for i in valid_innings if innings[i]['venue']==venue]
    if old_ground:is_nz=bool(old_ground[0]['host_nz'])
    context=np.zeros(len(rows),dtype=int)
    if use_venue:
        if is_nz:
            context=np.array([1 if innings[i]['host_nz'] else 0 for i in rows[:,0]])
        context=np.where(np.array([innings[i]['venue']==venue for i in rows[:,0]]),2,context)
    # Contexts are mutually exclusive; role/phase/order are categorical cell keys.
    pool=np.zeros((3,3,3,2,k),dtype=float)
    np.add.at(pool,(context,rows[:,2],rows[:,3],order,oi),weighted)
    # Resolve identity only from past eligible rosters; future display-name
    # collisions must not change an earlier forecast's identity lookup.
    past_ids={p['id'] for i in valid_innings for p in innings[i].get('players',[])}
    name_indexes={}
    for i,p in enumerate(people):
        if p['id'] in past_ids:name_indexes.setdefault(p['name'],[]).append(i)
    pmfs={};player_audits={};warnings=[];canon_lineups={}
    if not old_ground:warnings.append('No eligible past exact-ground data; broader role/context priors carry the forecast.')
    for team,names in lineups.items():
        canonical=[ALIASES.get(n,n) for n in names]
        if len(set(canonical))!=11:raise ValueError('The expected XI contains duplicate names or aliases.')
        canon_lineups[team]=canonical;table=np.zeros((11,3,2,k));audits=[]
        for j,(name,canonical_name) in enumerate(zip(names,canonical)):
            indexes=name_indexes.get(canonical_name,[])
            ids={people[i]['id'] for i in indexes}
            if len(ids)>1:raise ValueError('Ambiguous registry name; select a unique identity: '+canonical_name)
            index=indexes[0] if indexes else -1;pid=people[index]['id'] if index>=0 else canonical_name
            own=np.zeros_like(pool);own_mask=rows[:,1]==index
            if np.any(own_mask):np.add.at(own,(context[own_mask],rows[own_mask,2],rows[own_mask,3],order[own_mask],oi[own_mask]),weighted[own_mask])
            other=np.maximum(pool-own,0);role=_role(j+1);fallback_cells=0
            for phase in range(3):
                for batting_order in range(2):
                    def cell(ctx):
                        nonlocal fallback_cells
                        vector=other[ctx,role,phase,batting_order]
                        if not vector.sum():
                            fallback_cells+=1
                            vector=other[ctx,role].sum(axis=(0,1))
                        return vector
                    background=cell(0);prior=(background+0.5/k)/(background.sum()+0.5)
                    if use_venue:
                        for ctx in ((1,2) if is_nz else (2,)):
                            vector=cell(ctx);prior=(vector+strength*prior)/(vector.sum()+strength)
                    observations=own[:,role,phase,batting_order].sum(axis=0) if use_player else np.zeros(k)
                    table[j,phase,batting_order]=(observations+strength*prior)/(observations.sum()+strength)
            history=[]
            for i in valid_innings:
                history.extend((i,p) for p in innings[i].get('players',[]) if p['id']==pid and p.get('position') and _role(p['position'])==role)
            weighted_n=float(own[:,role].sum());raw_n=int(rows[own_mask & (rows[:,2]==role),5].sum())
            if raw_n==0:warnings.append('Cold start or new role for '+name+': other-player role/context prior used.')
            audits.append(dict(player=name,canonical_name=canonical_name,player_id=pid,position=j+1,
                               history_n=len({i for i,p in history}),weighted_n=weighted_n,
                               weighted_delivery_n=weighted_n,raw_delivery_n=raw_n,role=role,
                               fallback_context_cells=fallback_cells,
                               other_context_weighted_n={str(c):float(other[c,role].sum()) for c in range(3)}))
        assert np.allclose(table.sum(axis=-1),1) and np.all(table>=0)
        pmfs[team]=table;player_audits[team]=audits
    return dict(pmfs=pmfs,cdfs={team:np.minimum(np.cumsum(p,axis=-1),1) for team,p in pmfs.items()},outcomes=outcomes,
                lineups=lineups,player_audits=player_audits,training_max_date=training_max,training_n=len(valid_innings),warnings=warnings,
                settings=dict(half_life_years=half_life,prior_strength_deliveries=strength,use_venue=use_venue,use_player=use_player,
                              active_outcome_count=k,weak_symmetric_mass=0.5,context_pools='outsideNZ / NZother / exact for NZ; background / exact otherwise; own excluded from every pool',
                              role_phase_order_fallback='Empty context cell uses same-context other-player role aggregate, then own exact cell once',
                              phase='First20%,middle60%,last20% of declared legal-ball budget; shortened-budget extrapolation',
                              venue=venue,cutoff=cutoff))

def _initial_state(n,budget,targets):
    scores=np.full((n,11),-1,dtype=np.int64);scores[:,:2]=0
    batted=np.zeros((n,11),dtype=bool);batted[:,:2]=True
    return dict(scores=scores,batted=batted,balls_faced=np.zeros((n,11),dtype=np.int64),
                striker=np.zeros(n,dtype=int),non_striker=np.ones(n,dtype=int),next_player=np.full(n,2,dtype=int),
                team_totals=np.zeros(n,dtype=np.int64),extras=np.zeros(n,dtype=np.int64),legal_balls=np.zeros(n,dtype=np.int64),
                wickets=np.zeros(n,dtype=np.int64),first10_wickets=np.zeros(n,dtype=np.int64),
                wide_deliveries=np.zeros(n,dtype=np.int64),no_ball_deliveries=np.zeros(n,dtype=np.int64),
                free_hit=np.zeros(n,dtype=bool),done=np.zeros(n,dtype=bool),budget=int(budget),
                targets=np.zeros(n,dtype=np.int64) if targets is None else np.asarray(targets,dtype=np.int64))

def _advance_state(state,indices,events):
    """Apply a vector of decoded events to currently active simulations."""
    a=np.asarray(indices,dtype=int);events=np.asarray(events,dtype=np.int64)
    if not len(a):return
    if len(events)==1 and len(a)>1:events=np.repeat(events,len(a),axis=0)
    br,ex,wide,nb,parity,wicket=events[:,:6].copy().T
    boundary=events[:,6] if events.shape[1]==7 else np.where(np.isin(br,(4,6)),br,0)
    wide=wide.astype(bool);nb=nb.astype(bool);legal=~(wide|nb)
    # A completed winning run ends the chase; a boundary retains its allowance.
    # A winning wide/no-ball penalty takes precedence over later actions.
    needed=state['targets'][a]-state['team_totals'][a]
    chasing=state['targets'][a]>0
    penalty=(wide|nb).astype(int)
    penalty_wins=chasing&(needed<=penalty)
    running_wins=chasing&~penalty_wins&(boundary==0)&(br+ex>=needed)
    if np.any(running_wins):
        hit=running_wins&(br>0)
        br[hit]=np.maximum(needed[hit]-penalty[hit],0)
        ex[hit]=penalty[hit]
        nonhit=running_wins&~hit
        ex[nonhit]=needed[nonhit]
        parity[running_wins]=(needed[running_wins]-penalty[running_wins])%2
        wicket[running_wins]=0
    br[penalty_wins]=0;ex[penalty_wins]=needed[penalty_wins];parity[penalty_wins]=0;wicket[penalty_wins]=0
    original_striker=state['striker'][a].copy();original_non=state['non_striker'][a].copy()
    old_free=state['free_hit'][a].copy();before_legal=state['legal_balls'][a].copy()
    state['scores'][a,original_striker]+=br;state['balls_faced'][a,original_striker]+=(~wide).astype(int)
    state['extras'][a]+=ex;state['team_totals'][a]+=br+ex
    state['wide_deliveries'][a]+=wide.astype(int);state['no_ball_deliveries'][a]+=nb.astype(int)
    state['legal_balls'][a]+=legal.astype(int)
    actual_wicket=(wicket>0)&~((wicket==1)&(old_free|nb))
    state['wickets'][a]+=actual_wicket.astype(int)
    state['first10_wickets'][a]+=(actual_wicket&(before_legal<60)).astype(int)
    swap=parity.astype(bool)
    state['striker'][a]=np.where(swap,original_non,original_striker)
    state['non_striker'][a]=np.where(swap,original_striker,original_non)
    ended=(state['wickets'][a]>=10)|(state['legal_balls'][a]>=state['budget'])|((state['targets'][a]>0)&(state['team_totals'][a]>=state['targets'][a]))
    state['done'][a]=ended
    replace=actual_wicket&~ended
    if np.any(replace):
        ix=a[replace];dismissed=np.where(wicket[replace]==3,original_non[replace],original_striker[replace]);new=state['next_player'][ix]
        if np.any(new>=11):raise RuntimeError('Cannot introduce a twelfth expected-XI batter.')
        at_striker=state['striker'][ix]==dismissed
        if np.any(~at_striker&(state['non_striker'][ix]!=dismissed)):raise RuntimeError('Dismissed batter identity lost.')
        state['striker'][ix]=np.where(at_striker,new,state['striker'][ix]);state['non_striker'][ix]=np.where(at_striker,state['non_striker'][ix],new)
        state['scores'][ix,new]=0;state['batted'][ix,new]=True;state['next_player'][ix]+=1
    over_finished=legal&(state['legal_balls'][a]%6==0)
    ix=a[over_finished];old=state['striker'][ix].copy();state['striker'][ix]=state['non_striker'][ix];state['non_striker'][ix]=old
    state['free_hit'][a]=nb|(old_free&~legal)

def _simulate_innings(cdf,outcomes,n_sims,budget,innings_order,rng,targets=None):
    state=_initial_state(n_sims,budget,targets)
    # At most20 events per scheduled legal ball plus100. Never silently truncate.
    for _ in range(20*int(budget)+100):
        active=np.flatnonzero(~state['done'])
        if not len(active):return state
        fraction=state['legal_balls'][active]/budget
        phase=np.where(fraction<.2,0,np.where(fraction<.8,1,2))
        probabilities=cdf[state['striker'][active],phase,innings_order-1]
        draw=rng.random(len(active));event_index=np.sum(draw[:,None]>probabilities,axis=1)
        event_index=np.minimum(event_index,len(outcomes)-1)
        _advance_state(state,active,outcomes[event_index])
    raise RuntimeError('Illegal-delivery safety cap reached; simulation not silently truncated.')

def simulate_match(fit,batting_first,overs=50,n_sims=3000,seed=7):
    if int(overs)!=overs or not 20<=overs<=50:raise ValueError('Declared overs must be an integer from20 to50.')
    if n_sims<1:raise ValueError('At least one simulation is required.')
    names=list(fit['lineups']);rng=np.random.default_rng(seed);budget=int(overs)*6
    if batting_first=='Unknown':first=rng.integers(0,2,size=n_sims)
    elif batting_first in names:first=np.full(n_sims,names.index(batting_first),dtype=int)
    else:raise ValueError('Batting first must name a selected team or Unknown.')
    audit_keys=('team_totals','extras','legal_balls','wickets','first10_wickets','wide_deliveries','no_ball_deliveries','targets')
    audits={key:{team:np.zeros(n_sims,dtype=np.int64) for team in names} for key in audit_keys}
    arrays={team:dict(scores=np.full((n_sims,11),-1,dtype=np.int64),balls_faced=np.zeros((n_sims,11),dtype=np.int64),batted=np.zeros((n_sims,11),dtype=bool)) for team in names}
    for batting_order in (1,2):
        for j,team in enumerate(names):
            selected=np.flatnonzero(first==j if batting_order==1 else first!=j)
            if not len(selected):continue
            target=None if batting_order==1 else audits['team_totals'][names[1-j]][selected]+1
            state=_simulate_innings(fit['cdfs'][team],fit['outcomes'],len(selected),budget,batting_order,rng,target)
            for key in audit_keys:audits[key][team][selected]=state[key]
            for key in arrays[team]:arrays[team][key][selected]=state[key]
    teams={}
    for team in names:
        result=arrays[team];q,p_top=winner_shares(result['scores']);players=[]
        for j,audit in enumerate(fit['player_audits'][team]):
            b=result['batted'][:,j];runs=result['scores'][b,j];balls=result['balls_faced'][b,j]
            players.append({**audit,'mean_runs':float(runs.mean()) if len(runs) else 0.,'median_runs':float(np.median(runs)) if len(runs) else 0.,
                            'mean_balls':float(balls.mean()) if len(balls) else 0.,'median_balls':float(np.median(balls)) if len(balls) else 0.,
                            'p50plus':float(np.mean(result['scores'][:,j]>=50)),'p_top':float(p_top[j]),'q':float(q[j]),'p_bat':float(b.mean())})
        assert np.all(result['balls_faced'].sum(axis=1)==audits['legal_balls'][team]+audits['no_ball_deliveries'][team])
        assert np.all(np.maximum(result['scores'],0).sum(axis=1)+audits['extras'][team]==audits['team_totals'][team])
        assert np.all(audits['legal_balls'][team]<=budget) and np.all(audits['wickets'][team]<=10)
        teams[team]={**result,'players':players,**{key:audits[key][team] for key in audit_keys}}
    warnings=list(fit['warnings'])+['Experimental delivery empirical-Bayes model; uncalibrated probabilities. No bowler, weather, pressure-response or DLS model.',
        'Scores share balls, wickets and a generated chase target. Categorical event probabilities are fixed within each role/phase/order context.',
        'Expected XIs/order are scenario inputs. Historical lineups and toss are reconstructed assumptions without vintage proof.']
    if overs!=50:warnings.append('Shortened declared budget is an unvalidated extrapolation from standard-innings delivery history; not DLS pricing.')
    if batting_first=='Unknown':warnings.append('Batting first is a declared50:50 scenario, not a learned toss forecast.')
    return dict(teams=teams,**audits,batting_first=np.array(names)[first],overs=int(overs),budget=budget,
                warnings=warnings,training_max_date=fit['training_max_date'],training_n=fit['training_n'],
                settings={**fit['settings'],'overs':int(overs),'n_sims':n_sims,'seed':seed,'illegal_event_cap':20*budget+100},
                model='Delivery empirical-Bayes with shared balls/wickets and generated chase target')

def forecast_match(data,lineups,venue,cutoff,batting_first,overs=50,half_life=2,prior_strength=120,n_sims=3000,seed=7,use_venue=True,use_player=True):
    fit=fit_ball_model(data,lineups,venue,cutoff,half_life,prior_strength,use_venue,use_player)
    return simulate_match(fit,batting_first,overs,n_sims,seed)
