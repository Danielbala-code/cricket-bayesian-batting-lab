"""Prepare portable delivery counts from user-owned Cricsheet JSON; no network."""
import argparse, collections, hashlib, json, pathlib
import numpy as np

ROOT=pathlib.Path(__file__).parent
STANDARD_WICKETS={'bowled','caught','caught and bowled','lbw','stumped','hit wicket'}
COUNT_COLUMNS=['innings_index','player_index','role_index','phase_index','outcome_index','count']

def extract_game(game,match_id,nz_venues):
    info=game['info'];dates=info.get('dates',[]);date=min(dates) if dates else ''
    reasons=[]
    if date<'2016-01-01':reasons.append('before_window')
    if info.get('gender')!='male' or info.get('match_type')!='ODI':reasons.append('not_men_odi')
    if info.get('outcome',{}).get('result')=='no result':reasons.append('no_result')
    if info.get('outcome',{}).get('method') in ('D/L','DLS','VJD'):reasons.append('revised_target_method')
    if info.get('balls_per_over',6)!=6:reasons.append('nonstandard_over')
    scheduled=info.get('overs',50)
    if scheduled!=50:reasons.append('nonstandard_scheduled_overs')
    teams=info.get('teams',[]);registry=info.get('registry',{}).get('people',{});rosters=info.get('players',{})
    if len(teams)!=2:reasons.append('nonstandard_teams')
    for team in teams:
        if len(rosters.get(team,[]))!=11 or len(set(rosters.get(team,[])))!=11:reasons.append('non11_roster:'+team)
    rows=[];events=[];ordinary=0;seen=set()
    for source_index,inn in enumerate(game.get('innings',[])):
        team=inn.get('team');seen.add(team);local=list(reasons)
        if inn.get('super_over'):local.append('super_over');inning_order=0
        else:ordinary+=1;inning_order=ordinary
        if inning_order>2:local.append('extra_ordinary_innings')
        if inn.get('penalty_runs'):local.append('innings_penalty')
        if inn.get('absent_hurt'):local.append('absent_hurt')
        order=[];runs=collections.Counter();bf=collections.Counter();legal=extra=total=nonwide=noballs=wides=0;raw=[]
        for over in inn.get('overs',[]):
            for delivery in over.get('deliveries',[]):
                striker=delivery['batter'];non=delivery['non_striker']
                for n in (striker,non):
                    if n not in order:order.append(n)
                    if n not in rosters.get(team,[]):local.append('unrostered_player:'+n)
                run=delivery['runs'];ex=delivery.get('extras',{});br=int(run['batter']);er=int(run['extras'])
                wide=int(ex.get('wides',0)>0);nb=int(ex.get('noballs',0)>0)
                if br<0 or er<0 or run['total']!=br+er or er!=sum(ex.values()):local.append('invalid_run_accounting')
                if wide and nb:local.append('simultaneous_wide_noball')
                if ex.get('noballs',0)>1:local.append('nonstandard_noball')
                if ex.get('penalty',0):local.append('delivery_penalty')
                if set(ex)-{'wides','noballs','byes','legbyes','penalty'}:local.append('unsupported_extra')
                if wide and br:local.append('wide_batter_runs')
                if any(ex.get(k,0) for k in ('byes','legbyes')) and br:local.append('mixed_batter_bye_runs')
                wicket=0;ws=delivery.get('wickets',[])
                if len(ws)>1:local.append('multiple_wickets')
                for w in ws:
                    who=w['player_out'];kind=w['kind']
                    if who not in order:order.append(who)
                    if kind=='run out':
                        if who==striker:wicket=2
                        elif who==non:wicket=3
                        else:local.append('runout_unknown_end')
                    elif kind in STANDARD_WICKETS and who==striker:wicket=1
                    else:local.append('unsupported_wicket:'+kind)
                    if nb and wicket==1:local.append('ordinary_wicket_on_noball')
                if delivery.get('replacements'):local.append('delivery_replacement')
                pos=order.index(striker)+1
                if pos>11:local.append('more_than11_batters')
                role=0 if pos<=3 else 1 if pos<=7 else 2
                phase=0 if legal < scheduled*6*.2 else 1 if legal<scheduled*6*.8 else 2
                # Penalty for an ordinary wide/no-ball does not move the batters.
                parity=(br+er-int(wide)-int(nb)-int(ex.get('penalty',0)))%2
                boundary=0
                if not run.get('non_boundary',False):
                    if br in (4,6):boundary=br
                    elif ex.get('byes')==4 or ex.get('legbyes')==4 or ex.get('wides')==5:boundary=4
                raw.append(dict(source_index=source_index,player_id=registry.get(striker,'missing:'+striker),role=role,phase=phase,
                                outcome=(br,er,wide,nb,parity,wicket,boundary)))
                runs[striker]+=br;extra+=er;total+=int(run['total'])
                if not wide:bf[striker]+=1;nonwide+=1
                if nb:noballs+=1
                if wide:wides+=1
                if not wide and not nb:legal+=1
        roster=list(rosters.get(team,[]))
        for n in order:
            if n not in roster:roster.append(n)
        if len(order)>11:local.append('more_than11_batters')
        if not raw:local.append('missing_deliveries')
        if legal>scheduled*6:local.append('over_budget_or_miscount')
        if inn.get('miscounted_overs'):local.append('miscounted_overs')
        players=[]
        for name in sorted(roster):
            if name not in registry:local.append('missing_registry:'+name)
            b=name in order
            players.append(dict(id=registry.get(name,'missing:'+name),name=name,position=order.index(name)+1 if b else None,
                                runs=int(runs[name]) if b else None,balls_faced=int(bf[name]) if b else None))
        penalties=sum(inn.get('penalty_runs',{}).values())
        row=dict(id=f'{match_id}:{source_index}',match_id=match_id,date=date,team=team,
                 opponent=next((t for t in teams if t!=team),None),venue=info.get('venue',''),host_nz=info.get('venue') in nz_venues,
                 innings_order=inning_order,scheduled_overs=scheduled,eligible=not local,exclusions=sorted(set(local)),players=players,
                 actual_legal_balls=legal,actual_team_runs=total+penalties,actual_extras=extra+penalties,
                 actual_batter_runs=sum(runs.values()),actual_deliveries=len(raw),actual_nonwide_deliveries=nonwide,
                 actual_noball_deliveries=noballs,actual_wide_deliveries=wides)
        assert sum(p['runs'] or 0 for p in players)==row['actual_batter_runs']
        assert sum(p['balls_faced'] or 0 for p in players)==nonwide==legal+noballs or not row['eligible']
        assert row['actual_batter_runs']+row['actual_extras']==row['actual_team_runs']
        for event in raw:event['eligible']=row['eligible'];event['local_innings_index']=len(rows)
        rows.append(row);events.extend(raw)
    for team in teams:
        if team not in seen:
            row=dict(id=f'{match_id}:missing:{team}',match_id=match_id,date=date,team=team,
                opponent=next((t for t in teams if t!=team),None),venue=info.get('venue',''),host_nz=info.get('venue') in nz_venues,
                innings_order=2,scheduled_overs=scheduled,eligible=False,exclusions=sorted(set(reasons+['missing_innings'])),
                players=[dict(id=registry.get(n,'missing:'+n),name=n,position=None,runs=None,balls_faced=None) for n in sorted(rosters.get(team,[]))],
                actual_legal_balls=0,actual_team_runs=0,actual_extras=0,actual_batter_runs=0,actual_deliveries=0,
                actual_nonwide_deliveries=0,actual_noball_deliveries=0,actual_wide_deliveries=0)
            rows.append(row)
    return rows,events

def prepare(source,destination,nz_venues=None):
    source=pathlib.Path(source);destination=pathlib.Path(destination)
    if nz_venues is None:
        v2=ROOT/'data/innings.json'
        if not v2.exists():raise ValueError('Pass audited NZ venue set or provide existing data/innings.json')
        nz_venues={r['venue'] for r in json.loads(v2.read_text(encoding='utf-8')) if r['host_nz']}
    all_rows=[];raw_events=[];names={};source_hash=hashlib.sha256();source_files=0;before_window=0
    for path in sorted(source.rglob('*.json')):
        encoded=path.read_bytes();game=json.loads(encoded.decode('utf-8-sig'));source_files+=1
        source_hash.update(path.name.encode('utf-8'));source_hash.update(encoded)
        rows,events=extract_game(game,path.stem,nz_venues)
        if all(r['date']<'2016-01-01' for r in rows):before_window+=1;continue
        offset=len(all_rows)
        for event in events:
            if event['eligible']:raw_events.append((offset+event['local_innings_index'],event))
        all_rows.extend(rows)
    all_rows_indices=list(range(len(all_rows)))
    for i in all_rows_indices:
        for p in all_rows[i]['players']:names.setdefault(p['id'],p['name'])
    players=[dict(id=pid,name=names[pid]) for pid in sorted(names)];player_idx={p['id']:i for i,p in enumerate(players)}
    outcomes=sorted({event['outcome'] for _,event in raw_events});outcome_idx={o:i for i,o in enumerate(outcomes)}
    counter=collections.Counter((i,player_idx[e['player_id']],e['role'],e['phase'],outcome_idx[e['outcome']]) for i,e in raw_events)
    counts=np.array([(*key,value) for key,value in sorted(counter.items())],dtype=np.int64).reshape((-1,6))
    # Independently reconcile compact decoding against unaggregated extraction truth.
    checked=0
    for i,r in enumerate(all_rows):
        if not r['eligible']:continue
        subset=counts[counts[:,0]==i];decoded=[(outcomes[int(x[4])],int(x[5])) for x in subset]
        assert sum(n for o,n in decoded)==r['actual_deliveries']
        assert sum(o[0]*n for o,n in decoded)==r['actual_batter_runs']
        assert sum(o[1]*n for o,n in decoded)==r['actual_extras']
        assert sum((not o[2] and not o[3])*n for o,n in decoded)==r['actual_legal_balls']
        assert sum((not o[2])*n for o,n in decoded)==r['actual_nonwide_deliveries']
        for p in r['players']:
            pc=subset[subset[:,1]==player_idx[p['id']]]
            assert sum(outcomes[int(x[4])][0]*int(x[5]) for x in pc)==(p['runs'] or 0)
            assert sum((not outcomes[int(x[4])][2])*int(x[5]) for x in pc)==(p['balls_faced'] or 0)
        checked+=1
    excluded=collections.Counter(reason for r in all_rows if not r['eligible'] for reason in r['exclusions'])
    eligible=[r for r in all_rows if r['eligible']]
    metadata=dict(schema_version='1.1',innings=all_rows,players=players,outcomes=outcomes,count_columns=COUNT_COLUMNS,
        definitions={'outcome':['batter_runs','extras','is_wide','is_noball','running_parity','wicket_kind','boundary_allowance'],
          'wicket_kind':{'0':'none','1':'ordinary striker dismissal','2':'striker run-out','3':'nonstriker run-out'},
          'role':'0=positions1-3,1=4-7,2=8-11; historical encounter position is a training label only',
          'phase':'0=first20%,1=middle60%,2=last20%; running legal deliveries BEFORE event divided by scheduled_overs*6; never actual final duration',
          'balls_faced':'Non-wide deliveries, including no-balls. DNB=null; observed zero-ball batter=0. Total BF=legal deliveries+no-ball deliveries for eligible innings.',
          'parity':'(batter_runs+extras-wide penalty1-no-ball penalty1-five-run penalty)%2; unsupported penalties excluded',
          'boundary':'4/6 allowance from batter or bye/legbye/wide boundary, respecting runs.non_boundary. Extra-boundary inference follows recorded run breakdown; temporal order of boundary versus completed running is not recorded.',
          'source_scope':'Men ODI start>=2016-01-01; exclude no-results,D/L,DLS,VJD,non11rosters,superovers,unsupported events; explicit per-innings reasons',
          'future_support':'Global outcomes catalog includes future categories. Consumers MUST filter eligible innings date<cutoff first, decode active observed tuples, union fixed law-valid base and sort tuples. Never place prior mass across this global vocabulary.',
          'decision_time':'Post-toss/declared XI, before match; realised runs/BF/duration/positions are labels only. Metadata innings_order known post-toss; scheduled50 used standard evaluation.',
          'ordering':'Global player IDs sorted, outcome tuples lexicographically sorted; no fuzzy identity matching'},
        audit=dict(source=str(source),source_files=source_files,before_window_files_omitted=before_window,source_sha256=source_hash.hexdigest(),
          innings_metadata=len(all_rows),eligible_innings=len(eligible),excluded_innings=len(all_rows)-len(eligible),exclusion_reasons=dict(excluded),
          eligible_matches=len({r['match_id'] for r in eligible}),eligible_source_deliveries=len(raw_events),compressed_count_rows=len(counts),
          actual_legal_balls=sum(r['actual_legal_balls'] for r in eligible),actual_nonwide_deliveries=sum(r['actual_nonwide_deliveries'] for r in eligible),
          actual_noball_deliveries=sum(r['actual_noball_deliveries'] for r in eligible),actual_batter_runs=sum(r['actual_batter_runs'] for r in eligible),
          actual_extras=sum(r['actual_extras'] for r in eligible),reconciled_innings=checked,all_reconciliations_passed=True,
          min_date=min((r['date'] for r in eligible),default=None),max_date=max((r['date'] for r in eligible),default=None),
          nz_venues=sorted(nz_venues),npz_columns=COUNT_COLUMNS))
    destination.mkdir(parents=True,exist_ok=True);np.savez_compressed(destination/'ball_counts.npz',counts=counts)
    (destination/'ball_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    reloaded=np.load(destination/'ball_counts.npz',allow_pickle=False)['counts'];assert np.array_equal(reloaded,counts)
    return metadata

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',default=r'D:\odis_male_json');parser.add_argument('--output',default=str(ROOT/'data'))
    args=parser.parse_args();result=prepare(args.source,args.output);print(json.dumps(result['audit'],indent=2))
