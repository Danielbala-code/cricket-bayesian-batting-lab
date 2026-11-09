"""Read raw Cricsheet ODI files; write a compact derived archive, never raw data."""
import json,pathlib,collections

def prepare(source):
    records=[]
    for path in sorted(pathlib.Path(source).glob('*.json')):
        g=json.loads(path.read_text(encoding='utf-8-sig'));info=g['info'];date=min(info.get('dates',[]),default='')
        if info.get('gender')!='male' or info.get('match_type')!='ODI' or date<'2016-01-01':continue
        venue=info.get('venue','');reg=info.get('registry',{}).get('people',{})
        host_nz=info.get('city') in {'Auckland','Hamilton','Napier','Wellington','Christchurch','Mount Maunganui','Tauranga','Nelson','Dunedin','Whangarei','Queenstown','New Plymouth'}
        for team in info['teams']:
            inns=[i for i in g.get('innings',[]) if i.get('team')==team and not i.get('super_over',False)]
            inn=inns[0] if inns else None;order=[];runs=collections.Counter();early_wickets=0
            for over in inn.get('overs',[]) if inn else []:
                for d in over.get('deliveries',[]):
                    for name in (d['batter'],d['non_striker']):
                        if name not in order:order.append(name)
                    runs[d['batter']]+=d['runs']['batter']
                    for wicket in d.get('wickets',[]):
                        if wicket['player_out'] not in order:order.append(wicket['player_out'])
                        if over['over']<10 and wicket['kind'] not in ('retired hurt','retired not out'):early_wickets+=1
            dl=info.get('outcome',{}).get('method') in ('D/L','DLS')
            nr=info.get('outcome',{}).get('result')=='no result'
            roster=set(info.get('players',{}).get(team,[]))|set(order)
            records.append(dict(id=path.stem,date=date,team=team,opponent=next(t for t in info['teams'] if t!=team),venue=venue,
                                host_nz=host_nz,dl=dl,no_result=nr,eligible=bool(inn) and bool(order) and not nr and not dl,
                                early_trouble=early_wickets>=2,players=[dict(id=reg.get(n,n),name=n,position=order.index(n)+1 if n in order else None,
                                runs=runs[n] if n in order else None) for n in sorted(roster)]))
    return sorted(records,key=lambda r:(r['date'],r['id'],r['team']))

if __name__=='__main__':
    target=pathlib.Path(__file__).parent/'data/innings.json'
    rows=prepare(r'D:\odis_male_json');target.write_text(json.dumps(rows,separators=(',',':')),encoding='utf-8')
    print('Derived team rows:',len(rows),'eligible:',sum(r['eligible'] for r in rows),'latest:',max(r['date'] for r in rows))
