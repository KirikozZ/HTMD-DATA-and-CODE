"""Numerical water-density conversion and molecule-frame-weighted angle profiles."""
from pathlib import Path
import numpy as np,pandas as pd
from summarize import CASES,ratio,statrow,getarr
ROOT=Path(__file__).resolve().parent
def main():
    profile=pd.read_csv(ROOT/'tables/profiles.csv')
    water=profile[(profile.species=='water')&(profile.metric=='number_density')].copy()
    water[['mean','temporal_SEM','ci95_low','ci95_high']]*=18.01528/6.02214076e23*1e21
    water['metric']='mass_density';water['unit']='g cm^-3'
    water.to_csv(ROOT/'tables/water_mass_density_profiles.csv',index=False,encoding='utf-8-sig')
    rows=[];blocks=[]
    for case in CASES:
        d=pd.read_csv(ROOT/'angle_z'/case/'angle_blocks.csv')
        old=pd.read_csv(ROOT/'raw'/case/'profiles_blocks.csv')
        np.testing.assert_array_equal(d.water_n,old.water_n)
        np.testing.assert_allclose(d.cos_sum,old.cos_sum,rtol=1e-11,atol=1e-8)
        np.testing.assert_allclose(d.P2_sum,old.P2_sum,rtol=1e-11,atol=1e-8)
        stats=ratio(getarr(d,'theta_sum_deg'),getarr(d,'water_n'))
        for j,z in enumerate(sorted(d.z_A.unique())):rows.append(dict(case=case,z_A=z,unit='degree',**statrow(stats,j)))
        blocks.append(d.assign(case=case))
    pd.DataFrame(rows).to_csv(ROOT/'angle_z/angle_z_statistics.csv',index=False,encoding='utf-8-sig')
    pd.concat(blocks,ignore_index=True).to_csv(ROOT/'angle_z/angle_z_blocks.csv',index=False,encoding='utf-8-sig')
    print('Exported water mass density and mean-angle profiles.')
if __name__=='__main__':main()
