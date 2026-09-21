"""Export all 196 screening trajectories, preserving the paper's training labels.

Read-only source trajectories. Each case receives source SHA256, interval and
cumulative events, conservation checks, temporal blocks and a label comparison.
Differences from the author-approved training table are observations, not edits.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse,csv,hashlib,json,math,shutil,subprocess,time,traceback

ROOT=Path(__file__).resolve().parent
SPECIES=['water','Li','Cl','Mg']
TARGETS=['FWater','FLi','FCl','FMg']

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def js(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+'.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');temp.replace(p)

def table(p,rows):
    if not rows:return
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

def case_for(row,raw):
    h=float(row['Hydrophilicity']);q=int(row['Charge']);delta=round(float(row['Sigma'])-2.846421,6)
    sign=lambda x:format(x,'+g') if x>0 else format(x,'g')
    rel=Path(f'{h:.1f}q')/f'charge_{sign(q)}_sigma_{sign(delta)}'/'equilibrium.lammpstrJ'
    return rel,raw/rel

def derive(src,dest,row):
    totals={s:{'legacy':0,'plane':0,'transit':0} for s in SPECIES}
    initial={};prev=None;frames=0;first=None;last=None;max_balance=0;blocks={}
    with src.open(encoding='utf-8-sig',newline='') as f,dest.open('w',encoding='utf-8',newline='') as out:
        reader=csv.DictReader(f)
        fields=['frame_index','step','time_ns','interval_start_ns','interval_end_ns','interval_duration_ns','interval_observed']
        for s in SPECIES:
            fields += [s+'_'+v for v in ['legacy_forward','legacy_reverse','legacy_net','legacy_cumulative',
              'plane_forward','plane_reverse','plane_net','plane_cumulative','transit_forward','transit_reverse','transit_net','transit_cumulative',
              'feed','membrane','permeate','wrap_net']]
        writer=csv.DictWriter(out,fieldnames=fields);writer.writeheader()
        for record in reader:
            d={k:int(v) for k,v in record.items()};step=d['step'];t=step*1e-6
            if first is None:first=t
            if last is not None and step-last!=10000:raise ValueError('Unexpected dump interval')
            prior=t if prev is None else prev['step']*1e-6;duration=t-prior
            output={'frame_index':frames,'step':step,'time_ns':format(t,'.8g'),'interval_start_ns':format(prior,'.8g'),
                    'interval_end_ns':format(t,'.8g'),'interval_duration_ns':format(duration,'.8g'),'interval_observed':int(prev is not None)}
            if prev is not None:
                block_index=min(11,max(0,math.ceil(step/10000000)-1))
                b=blocks.setdefault(block_index,{'block_index':block_index+1,'start_ns':prior,'end_ns':t,'duration_ns':0.0})
                b['end_ns']=t;b['duration_ns']+=duration
            for s in SPECIES:
                k=s+'_';total=d[k+'feed']+d[k+'membrane']+d[k+'permeate']
                initial.setdefault(s,total)
                if initial[s]!=total:raise ValueError('Species population changed')
                if prev is not None:
                    residual=d[k+'permeate']-prev[k+'permeate']-(d[k+'plane_forward']-d[k+'plane_reverse']-d[k+'wrap_net'])
                    max_balance=max(max_balance,abs(residual))
                    if residual:raise ValueError('Mass balance residual')
                for method in ['legacy','plane','transit']:
                    forward=d[k+method+'_forward'];reverse=d[k+method+'_reverse'];net=forward-reverse
                    totals[s][method]+=net
                    output.update({k+method+'_forward':forward,k+method+'_reverse':reverse,k+method+'_net':net,k+method+'_cumulative':totals[s][method]})
                    if prev is not None:
                        for name,value in [('forward',forward),('reverse',reverse),('net',net)]:
                            key=k+method+'_'+name;b[key]=b.get(key,0)+value
                for col in ['feed','membrane','permeate','wrap_net']:output[k+col]=d[k+col]
            writer.writerow(output);frames+=1;last=step;prev=d
    if not frames:raise ValueError('No complete frame')
    expected={s:int(row[c]) for s,c in zip(SPECIES,TARGETS)}
    observed={s:totals[s]['legacy'] for s in SPECIES}
    info={'frames':frames,'first_time_ns':first,'last_time_ns':last*1e-6,'observed_duration_ns':last*1e-6-first,
          'complete_120ns_saved_window':frames==12000 and first==.01 and last==120000000,
          'mass_balance_max_error':max_balance,'initial_species_counts':initial,'training_counts':expected,
          'recounted_counts':totals,'matches_training':expected==observed,
          'training_values_retained':True}
    return info,list(blocks.values())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--raw-root',type=Path,required=True)
    ap.add_argument('--training-csv',type=Path,required=True);ap.add_argument('--counter',type=Path,required=True)
    ap.add_argument('--workers',type=int,default=4);ap.add_argument('--resume',action='store_true');a=ap.parse_args()
    raw=a.raw_root.resolve();training=a.training_csv.resolve();counter=a.counter.resolve()
    original_sha=sha(training)
    with training.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    assert len(rows)==196 and len({tuple(r[k] for k in ['Hydrophilicity','Charge','Sigma']) for r in rows})==196
    todo=[(i+1,r,*case_for(r,raw)) for i,r in enumerate(rows)]
    missing=[str(p) for _,_,_,p in todo if not p.exists()]
    if missing:raise SystemExit('Missing source trajectories: '+str(missing))
    for d in ['training','events','counts','case_audits','logs','blocks']: (ROOT/d).mkdir(exist_ok=True)
    fixed=ROOT/'training/paper_training_data.csv'
    if fixed.exists() and sha(fixed)!=original_sha:raise SystemExit('Frozen training source changed')
    if training != fixed.resolve():
        shutil.copy2(training,fixed)
    counter_sha=sha(counter)
    js(ROOT/'protocol.json',{'training_sha256':original_sha,'counter_sha256':counter_sha,'raw_root':str(raw),
       'cases':196,'workers':a.workers,'species':SPECIES,'primary_training_source':'training/paper_training_data.csv',
       'author_decision':'Retain the manuscript model training labels, including the two disclosed differing rows.',
       'coordinates_A':{'lower':39.803101,'upper':53.043098,'legacy_slab_upper':58.043098},
       'definitions':{'legacy':'float32 positions in common open slab in adjacent frames; signed crossing of upper plane',
           'plane':'full upper-plane forward-minus-reverse crossings with minimum-image periodic correction',
           'transit':'completed side-to-side transfer with periodic reservoir wrapping reset'},
       'frame_zero':'first saved frame is baseline; its zero events do not infer events before it',
       'raw_source_hash':'SHA256 of complete source file after parsing; size and mtime must remain unchanged'})
    start=time.time();records=[]
    def run(item):
        idx,row,rel,p=item;key=f'S{idx:03d}';t=time.time();before=p.stat();audit_path=ROOT/'case_audits'/f'{key}.json'
        if a.resume and audit_path.exists():
            old=json.loads(audit_path.read_text(encoding='utf-8'))
            if old.get('processing_status')=='complete' and old.get('source_bytes')==before.st_size and old.get('source_mtime_ns')==before.st_mtime_ns and old.get('training_sha256')==original_sha and old.get('counter_sha256')==counter_sha:
                if all(sha(ROOT/x['path'])==x['sha256'] for x in old['outputs']):return old
        base={'case_id':key,'row_id_0based':idx-1,'Hydrophilicity':float(row['Hydrophilicity']),'Charge':int(row['Charge']),
              'Sigma':float(row['Sigma']),'source_relative':rel.as_posix(),'source_bytes':before.st_size,
              'source_mtime_ns':before.st_mtime_ns,'training_sha256':original_sha,'counter_sha256':counter_sha}
        try:
            counts=ROOT/'counts'/f'{key}.csv';events=ROOT/'events'/f'{key}.csv';blockpath=ROOT/'blocks'/f'{key}.csv'
            ret=subprocess.run([str(counter),str(p),str(counts)],capture_output=True,text=True)
            (ROOT/'logs'/f'{key}.txt').write_text(ret.stdout+'\n'+ret.stderr,encoding='utf-8')
            if ret.returncode:raise RuntimeError('Counter rejected trajectory: '+ret.stderr.strip())
            details,blocks=derive(counts,events,row);table(blockpath,blocks)
            source_sha=sha(p);after=p.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise RuntimeError('Source changed while read')
            record={**base,**details,'source_sha256':source_sha,'processing_status':'complete','seconds':round(time.time()-t,3),
                    'counter_stdout':ret.stdout.strip(),'outputs':[{'path':x.relative_to(ROOT).as_posix(),'sha256':sha(x)} for x in [counts,events,blockpath]]}
        except Exception as e:
            record={**base,'processing_status':'failed','error':str(e),'traceback':traceback.format_exc(),'seconds':round(time.time()-t,3)}
        js(audit_path,record);return record
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for future in as_completed([pool.submit(run,x) for x in todo]):
            record=future.result();records.append(record)
            js(ROOT/'progress.json',{'completed':len(records),'total':196,'elapsed_seconds':round(time.time()-start),
                'processing_failures':sum(r['processing_status']!='complete' for r in records),
                'label_differences':sum(r.get('matches_training') is False for r in records),
                'latest':record['case_id'],'finished':False})
            print(f"{len(records):3d}/196 {record['case_id']} {record['processing_status']} frames={record.get('frames')} matches_training={record.get('matches_training')} seconds={record['seconds']}",flush=True)
    records.sort(key=lambda r:r['row_id_0based']);summary=[];allblocks=[]
    for r in records:
        row={k:r.get(k) for k in ['case_id','row_id_0based','Hydrophilicity','Charge','Sigma','source_relative','processing_status','frames',
             'first_time_ns','last_time_ns','observed_duration_ns','complete_120ns_saved_window','mass_balance_max_error','matches_training','source_sha256']}
        for species,target in zip(SPECIES,TARGETS):
            row[target+'_paper']=r.get('training_counts',{}).get(species)
            for method in ['legacy','plane','transit']:row[target+'_'+method]=r.get('recounted_counts',{}).get(species,{}).get(method)
            row[target+'_difference']=row[target+'_legacy']-row[target+'_paper'] if row[target+'_legacy'] is not None else None
        summary.append(row)
        bp=ROOT/'blocks'/f"{r['case_id']}.csv"
        if r['processing_status']=='complete':
            with bp.open(encoding='utf-8-sig') as f:
                for b in csv.DictReader(f):allblocks.append({'case_id':r['case_id'],**b})
    table(ROOT/'recount_summary.csv',summary);table(ROOT/'training_vs_recount_differences.csv',[r for r in summary if r['matches_training'] is False])
    table(ROOT/'blocks_10ns_all.csv',allblocks)
    assert sha(training)==original_sha==sha(fixed)
    js(ROOT/'recount_manifest.json',{'cases':records,'complete_processing':all(r['processing_status']=='complete' for r in records),
        'full_196_exported':len(records)==196 and all(r['processing_status']=='complete' for r in records),
        'label_difference_count':sum(r.get('matches_training') is False for r in records),'training_unchanged':True,
        'elapsed_seconds':round(time.time()-start),'protocol':'protocol.json'})
    js(ROOT/'progress.json',{'completed':196,'total':196,'finished':True,'processing_failures':sum(r['processing_status']!='complete' for r in records),
       'label_differences':sum(r.get('matches_training') is False for r in records),'elapsed_seconds':round(time.time()-start)})
    print('DONE; authoritative paper training table unchanged.',flush=True)
    if any(r['processing_status']!='complete' for r in records):raise SystemExit(2)

if __name__=='__main__':main()
