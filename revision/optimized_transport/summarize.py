from pathlib import Path
import json,math
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
SPECIES=['water','Li','Cl','Mg']
AVOGADRO=6.02214076e23
AREA_A2=25.5027*29.448
AREA_M2=AREA_A2*1e-20
MOLAR_PER_COUNT_NS=1e9/AREA_M2/AVOGADRO
WATER_VOLUME_M3_MOL=18.01528e-6

def summarize(meta):
    d=pd.read_csv(ROOT/'recounted'/f"{meta['case']}.csv")
    assert len(d)==12000 and d.step.iloc[0]==10000 and d.step.iloc[-1]==120000000
    assert np.all(np.diff(d.step)==10000)
    exposure=(d.step.iloc[-1]-d.step.iloc[0])*1e-6
    summary=dict(meta)
    phys=[];blocks=[]
    for species in SPECIES:
        fwd=int(d[species+'_plane_forward'].sum());rev=int(d[species+'_plane_reverse'].sum());net=fwd-rev
        legacy=int(d[species+'_legacy_forward'].sum()-d[species+'_legacy_reverse'].sum())
        summary['F'+('Water' if species=='water' else species)]=legacy
        rate=net/exposure
        pr={**meta,'species':species,'forward':fwd,'reverse':rev,'net_count':net,
            'transit_forward':int(d[species+'_transit_forward'].sum()),
            'transit_reverse':int(d[species+'_transit_reverse'].sum()),
            'legacy_count':legacy,'legacy_minus_plane_net':legacy-net,
            'start_ns':float(d.step.iloc[0]*1e-6),'end_ns':float(d.step.iloc[-1]*1e-6),
            'duration_ns':float(exposure),'area_nm2':AREA_A2*.01,
            'count_per_ns':rate,'count_per_nm2_ns':rate/(AREA_A2*.01),
            'flux_mol_m2_s':rate*MOLAR_PER_COUNT_NS,
            'water_LMH':rate*MOLAR_PER_COUNT_NS*WATER_VOLUME_M3_MOL*3.6e6 if species=='water' else None,
            'feed_mean_count':float(d[species+'_feed'].iloc[1:].mean())}
        phys.append(pr)
        total=d[species+'_feed']+d[species+'_membrane']+d[species+'_permeate']
        assert total.nunique()==1
        assert np.all(np.diff(d[species+'_permeate']) == (d[species+'_plane_forward']-d[species+'_plane_reverse']-d[species+'_wrap_net']).iloc[1:])
        for lo,hi in zip(range(0,120,10),range(10,121,10)):
            mask=(d.step>max(lo*1e6,10000))&(d.step<=hi*1e6)
            duration=hi-max(lo,.01)
            netblock=int((d.loc[mask,species+'_plane_forward']-d.loc[mask,species+'_plane_reverse']).sum())
            blocks.append({**meta,'species':species,'start_ns':max(lo,.01),'end_ns':hi,'duration_ns':duration,
                'net_count':netblock,'flux_mol_m2_s':netblock/duration*MOLAR_PER_COUNT_NS,
                'water_LMH':netblock/duration*MOLAR_PER_COUNT_NS*WATER_VOLUME_M3_MOL*3.6e6 if species=='water' else None})
        assert sum(x['net_count'] for x in blocks if x['species']==species)==net
    summary['Selectivity_score']=(summary['FLi']+1)/(summary['FMg']+1)
    summary['ln_Selectivity']=math.log(summary['Selectivity_score'])
    li,mg=phys[1],phys[3]
    physical_selectivity=li['net_count']/mg['net_count']*mg['feed_mean_count']/li['feed_mean_count'] if mg['net_count']>0 and li['net_count']>0 else None
    summary.update({'physical_Li_net':li['net_count'],'physical_Mg_net':mg['net_count'],
        'physical_selectivity':physical_selectivity,'ln_physical_selectivity':math.log(physical_selectivity) if physical_selectivity is not None else None,
        'physical_selectivity_status':'finite' if physical_selectivity is not None else 'not finitely estimable: Mg net = 0',
        'water_flux_LMH':phys[0]['water_LMH'],'Li_flux_mol_m2_s':li['flux_mol_m2_s'],
        'Cl_flux_mol_m2_s':phys[2]['flux_mol_m2_s'],'Mg_flux_mol_m2_s':mg['flux_mol_m2_s']})
    return summary,phys,blocks

def main():
    cases=json.loads((ROOT/'cases.json').read_text())
    rows=[summarize(meta) for meta in cases]
    summaries=[r[0] for r in rows];physical=[x for r in rows for x in r[1]];blocks=[x for r in rows for x in r[2]]
    for name,values in [('transport_counts',summaries),('physical_flux',physical),('flux_blocks_10ns',blocks)]:
        pd.DataFrame(values).to_csv(ROOT/(name+'.csv'),index=False,encoding='utf-8-sig',float_format='%.12g')
    result={'area_A2':AREA_A2,'Avogadro':AVOGADRO,'water_molar_volume_m3_mol':WATER_VOLUME_M3_MOL,'summaries':summaries,'physical':physical,'blocks':blocks}
    (ROOT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print('Summarized',len(cases),'optimized-condition trajectories.')
if __name__=='__main__':main()
