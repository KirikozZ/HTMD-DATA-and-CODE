import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import numpy as np,pandas as pd,json,math
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'optimized_transport'
( ROOT/'tables').mkdir(exist_ok=True)
RESULT=json.loads((BASE/'results.json').read_text(encoding='utf-8'))
CASES=[x['case'] for x in RESULT['summaries']]
SP=['water','Li','Cl','Mg'];IONS=['Li','Cl','Mg'];AREA=RESULT['area_A2'];LZ=87.576096
LO,HI=39.803101,53.043098
RNG=np.random.default_rng(20260917);NBOOT=5000
# Shared resamples preserve covariance of numerator/denominator and paired observables.
WEIGHTS={n:RNG.multinomial(n,np.ones(n)/n,size=NBOOT).astype(float) for n in [12,10,6,4]}
def ratio(num,den,nb=12):
 a=np.asarray(num,float);b=np.asarray(den,float)
 if a.ndim==1:a=a[:,None]
 if b.ndim==1:b=b[:,None]
 a,b=np.broadcast_arrays(a,b)
 with np.errstate(divide='ignore',invalid='ignore'):
  value=a.sum(0)/b.sum(0)
  samples=(WEIGHTS[nb]@a)/(WEIGHTS[nb]@b)
 valid=np.isfinite(samples).sum(0)
 with np.errstate(invalid='ignore'):
  sem=np.nanstd(samples,axis=0,ddof=1)
  low=np.nanquantile(samples,.025,axis=0)
  high=np.nanquantile(samples,.975,axis=0)
 support=(b>0).sum(0)
 bad=(support<3)|(valid<.95*NBOOT)|~np.isfinite(value)
 sem[bad]=np.nan;low[bad]=np.nan;high[bad]=np.nan
 return value,sem,low,high,support,samples
def statrow(est,j=0):
 v,e,l,h,n,_=est
 return dict(mean=float(v[j]),temporal_SEM=float(e[j]),ci95_low=float(l[j]),ci95_high=float(h[j]),support_blocks=int(n[j]))
def save(name,rows):
 d=pd.DataFrame(rows);d.to_csv(ROOT/'tables'/f'{name}.csv',index=False,encoding='utf-8-sig',float_format='%.12g');return d
def group(a,k):return np.asarray(a).reshape((12//k,k)+np.asarray(a).shape[1:]).sum(1)
def sensitivity(case,metric,num,den,extra=None):
 rows=[]
 for desc,n,k,start in [('10ns',12,1,0),('20ns',6,2,0),('30ns',4,3,0),('20to120ns',10,1,2)]:
  a=np.asarray(num)[start:];b=np.asarray(den)[start:]
  if k>1:a=group(a,k);b=group(b,k)
  st=statrow(ratio(a,b,n));rows.append(dict(case=case,metric=metric,window=desc,**(extra or {}),**st))
 return rows
def getarr(df,column):
 return df.pivot(index='block',columns='z_A',values=column).sort_index().sort_index(axis=1).to_numpy()
def main():
 transport=[];profile=[];orient=[];hyd=[];partition=[];residence=[];survival=[];rdfrows=[];senses=[];support=[];block_tables=[];rdfchecks=[]
 for case in CASES:
  raw=ROOT/'raw'/case
  meta=next(x for x in RESULT['summaries'] if x['case']==case)
  flux=pd.read_csv(BASE/'recounted'/f'{case}.csv')
  block=((flux.step.to_numpy()-1)//10000000).astype(int)
  dt=np.full(12,10.);dt[0]=9.99;exposure=dt.sum()
  counts={}
  for sp in SP:
   for method in ['legacy','plane']:
    nn=(flux[f'{sp}_{method}_forward']-flux[f'{sp}_{method}_reverse']).to_numpy().copy();nn[0]=0
    cc=np.bincount(block,weights=nn,minlength=12);counts[method,sp]=cc
    factor=exposure if method=='legacy' else 1e9/(AREA*1e-20*RESULT['Avogadro'])
    unit='count' if method=='legacy' else 'mol m^-2 s^-1'
    if method=='plane' and sp=='water':factor*=RESULT['water_molar_volume_m3_mol']*3.6e6;unit='LMH'
    st=ratio(cc*factor,dt)
    transport.append(dict(case=case,method=method,species=sp,metric='count' if method=='legacy' else 'flux',unit=unit,**statrow(st)))
    for b in range(12):block_tables.append(dict(case=case,method=method,species=sp,block=b+1,duration_ns=dt[b],net_count=cc[b]))
    senses+=sensitivity(case,method+'_'+sp,cc*factor,dt)
  li=counts['legacy','Li'];mg=counts['legacy','Mg']
  lf=(WEIGHTS[12]@li)/(WEIGHTS[12]@dt)*exposure
  mf=(WEIGHTS[12]@mg)/(WEIGHTS[12]@dt)*exposure
  for metric,fn in [('Selectivity',lambda l,m:(l+1)/(m+1)),('ln_Selectivity',lambda l,m:np.log((l+1)/(m+1)))]:
   boots=fn(lf,mf)
   transport.append(dict(case=case,method='legacy',species='Li_Mg',metric=metric,unit='dimensionless',
    mean=float(fn(li.sum(),mg.sum())),temporal_SEM=float(boots.std(ddof=1)),ci95_low=float(np.quantile(boots,.025)),ci95_high=float(np.quantile(boots,.975)),support_blocks=12))
  for name,n,k,start in [('10ns',12,1,0),('20ns',6,2,0),('30ns',4,3,0),('20to120ns',10,1,2)]:
   lc=li[start:];mc=mg[start:];tt=dt[start:]
   if k>1:lc=group(lc,k);mc=group(mc,k);tt=group(tt,k)
   bb=np.log(((WEIGHTS[n]@lc)/(WEIGHTS[n]@tt)*exposure+1)/((WEIGHTS[n]@mc)/(WEIGHTS[n]@tt)*exposure+1))
   senses.append(dict(case=case,metric='ln_Selectivity_119.99ns_equivalent',window=name,mean=float(np.log((lc.sum()/tt.sum()*exposure+1)/(mc.sum()/tt.sum()*exposure+1))),temporal_SEM=float(bb.std(ddof=1)),ci95_low=float(np.quantile(bb,.025)),ci95_high=float(np.quantile(bb,.975)),support_blocks=n))
  if mg.sum()==0:
   support.append(dict(case=case,metric='Mg zero-event bound',value=-math.log(.05)/exposure,unit='events/ns',note='One-sided 95% Poisson upper bound; separate from zero observed temporal SEM.'))
  prof=pd.read_csv(raw/'profiles_blocks.csv')
  reg=pd.read_csv(raw/'regions_blocks.csv')
  frames=reg[reg.region==0].sort_values('block').frames.to_numpy()
  assert np.all(frames==1000)
  z=np.sort(prof.z_A.unique());dz=prof.dz_A.iloc[0]
  # Independent population reconciliation with earlier, separately implemented plane counter.
  for ir,name in enumerate(['feed','membrane','permeate']):
   rr=reg[reg.region==ir].sort_values('block')
   for sp in SP:
    reference=np.bincount(block,weights=flux[f'{sp}_{name}'],minlength=12)
    np.testing.assert_array_equal(rr[f'{sp}_n'],reference)
  nw=getarr(prof,'water_n')
  for sp in SP:
   n=getarr(prof,sp+'_n')
   st=ratio(n,frames[:,None]*AREA*dz/1000)
   for j,zz in enumerate(z):profile.append(dict(case=case,metric='number_density',species=sp,z_A=zz,unit='nm^-3',**statrow(st,j)))
  for field,label in [('cos_sum','mean_cos_theta'),('P2_sum','P2')]:
   st=ratio(getarr(prof,field),nw)
   for j,zz in enumerate(z):profile.append(dict(case=case,metric=label,species='water',z_A=zz,unit='dimensionless',**statrow(st,j)))
   mm=reg[reg.region==1].sort_values('block')
   senses+=sensitivity(case,label,mm[field],mm.water_n)
  for sp in IONS:
   st=ratio(getarr(prof,sp+'_CN_sum'),getarr(prof,sp+'_n'))
   for j,zz in enumerate(z):profile.append(dict(case=case,metric='hydration_CN',species=sp,z_A=zz,unit='water/ion',**statrow(st,j)))
  od=pd.read_csv(raw/'orientation_blocks.csv')
  oh=od.pivot(index='block',columns='cos_theta',values='count').to_numpy()
  oc=np.sort(od.cos_theta.unique());oo=ratio(oh,oh.sum(1)[:,None]*.05)
  for j,c in enumerate(oc):orient.append(dict(case=case,cos_theta=c,**statrow(oo,j)))
  regions={i:reg[reg.region==i].sort_values('block') for i in range(3)}
  for sp in IONS:
   cn={}
   for ir,label in enumerate(['feed','membrane','permeate']):
    rr=regions[ir]
    for cut,field in [('base','_CN_sum'),('minus_0.1A','_CNlo_sum'),('plus_0.1A','_CNhi_sum')]:
     st=ratio(rr[sp+field],rr[sp+'_n'])
     hyd.append(dict(case=case,species=sp,region=label,cutoff=cut,ion_frames=int(rr[sp+'_n'].sum()),**statrow(st)))
     if cut=='base':cn[ir]=st
    if ir==1:senses+=sensitivity(case,'membrane_CN',rr[sp+'_CN_sum'],rr[sp+'_n'],dict(species=sp))
   # Matched membrane/feed CN difference with joint block resampling.
   samples=cn[1][5]-cn[0][5]
   hyd.append(dict(case=case,species=sp,region='membrane_minus_feed',cutoff='base',ion_frames=int(regions[1][sp+'_n'].sum()),
    mean=float(cn[1][0][0]-cn[0][0][0]),temporal_SEM=float(np.nanstd(samples,ddof=1)) if np.isfinite(cn[1][1][0]) else np.nan,ci95_low=float(np.nanquantile(samples,.025)) if np.isfinite(cn[1][1][0]) else np.nan,ci95_high=float(np.nanquantile(samples,.975)) if np.isfinite(cn[1][1][0]) else np.nan,
    support_blocks=int(cn[1][4][0])))
   nmem=regions[1][sp+'_n'].to_numpy();nfeed=regions[0][sp+'_n'].to_numpy()
   st=ratio(nmem*LO/(HI-LO),nfeed)
   partition.append(dict(case=case,species=sp,metric='geometric_membrane_feed_ratio',**statrow(st)))
   senses+=sensitivity(case,'partition_geometric',nmem*LO/(HI-LO),nfeed,dict(species=sp))
   wm=regions[1].water_n.to_numpy();wf=regions[0].water_n.to_numpy()
   with np.errstate(divide='ignore',invalid='ignore'):
    boot=(WEIGHTS[12]@nmem)/(WEIGHTS[12]@wm)/((WEIGHTS[12]@nfeed)/(WEIGHTS[12]@wf))
   val=nmem.sum()/wm.sum()/(nfeed.sum()/wf.sum())
   partition.append(dict(case=case,species=sp,metric='water_normalized_enrichment',mean=float(val),temporal_SEM=float(np.nanstd(boot,ddof=1)),
    ci95_low=float(np.nanquantile(boot,.025)),ci95_high=float(np.nanquantile(boot,.975)),support_blocks=12))
  ev=pd.read_csv(raw/'residence_episodes.csv')
  for isp,sp in enumerate(SP):
   e=ev[(ev.species==isp)&(ev.left_censored==0)].copy()
   for tau in [.1,1.,5.]:
    ee=e[e.entry_ns<=120-tau+1e-9]
    den=np.bincount(ee.entry_block,minlength=12)
    durations=np.minimum(ee.duration_ns.to_numpy(),tau)
    num=np.bincount(ee.entry_block,weights=durations,minlength=12)
    # Every selected entry has a full tau-long follow-up, so its restricted duration is observed.
    assert np.all(ee.loc[ee.right_censored==1,'duration_ns']>=tau-1e-8)
    st=ratio(num,den)
    residence.append(dict(case=case,species=sp,tau_ns=tau,n_entries=len(ee),completed_before_tau=int((ee.duration_ns<tau-1e-9).sum()),
     reached_tau=int((ee.duration_ns>=tau-1e-9).sum()),initial_left_censored=int(((ev.species==isp)&(ev.left_censored==1)).sum()),
     terminal_right_censored=int(((ev.species==isp)&(ev.right_censored==1)).sum()),**statrow(st)))
    if tau==1:
     senses+=sensitivity(case,'residence_RMST1ns',num,den,dict(species=sp))
     for lag in np.linspace(0,1,51):
      nn=np.bincount(ee.entry_block,weights=(ee.duration_ns.to_numpy()>lag+1e-9),minlength=12)
      ss=ratio(nn,den)
      survival.append(dict(case=case,species=sp,lag_ns=lag,**statrow(ss)))
  rdf=pd.read_csv(raw/'rdf_blocks.csv')
  for isp,sp in enumerate(IONS):
   sub=rdf[rdf.species==isp]
   radii=np.sort(sub.r_A.unique())
   num=sub.pivot(index='block',columns='r_A',values='pair_count').to_numpy()
   den=sub.pivot(index='block',columns='r_A',values='normalizer').to_numpy()
   st=ratio(num,den)
   for j,rad in enumerate(radii):rdfrows.append(dict(case=case,species=sp,r_A=rad,**statrow(st,j)))
   intervals={'Li':(2.35,3.4),'Cl':(3.5,4.4),'Mg':(2.35,3.3)}
   a,b=intervals[sp];sm=np.convolve(st[0],np.ones(5)/5,'same');ids=np.flatnonzero((radii>=a)&(radii<=b))
   minimum=ids[np.argmin(sm[ids])]
   rdfchecks.append(dict(case=case,species=sp,fixed_cutoff_A={'Li':2.75,'Cl':3.85,'Mg':2.65}[sp],diagnostic_min_A=radii[minimum],diagnostic_smoothed_g=sm[minimum],note='Minimum in prespecified first-shell search interval; fixed cutoff retained across systems.'))
  print('SUMMARIZED',case,flush=True)
 for name,rows in [('transport_summary',transport),('transport_blocks',block_tables),('profiles',profile),('orientation_distribution',orient),
   ('hydration_summary',hyd),('partition_summary',partition),('residence_summary',residence),('residence_survival',survival),
   ('rdf_profiles',rdfrows),('block_window_sensitivity',senses),('rare_event_limits',support),('rdf_cutoff_check',rdfchecks)]:
  save(name,rows)
 (ROOT/'statistics_protocol.json').write_text(json.dumps(dict(bootstrap_draws=NBOOT,seed=20260917,block_ns=10,independent_runs_per_case=1,
  error_bars='SD of block-bootstrap estimates (temporal SEM), not SD across independent trajectories',profile_dz_A=dz,residence_tau_ns=1),indent=2))
 print('ALL STATISTICS COMPLETE',flush=True)
if __name__=='__main__':main()
