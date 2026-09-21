"""Export main seed-42 and complete five-seed numeric result tables, without layout software."""
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import norm
ROOT=Path(__file__).resolve().parent
TARGETS=['FWater','FLi','FCl','FMg','ln_selectivity'];SHORT=['Water','Li','Cl','Mg','Score']
def main():
    out=ROOT/'publication';out.mkdir(exist_ok=True)
    pred=pd.read_csv(ROOT/'diagnostics/selected_outer_residuals.csv')
    met=pd.read_csv(ROOT/'validation/nested/all_metrics.csv')
    cov=pd.read_csv(ROOT/'diagnostics/coverage_all_methods.csv')
    for label,seed in [('main_seed42',42),('SI_five_seeds',None)]:
        folder=out/label;folder.mkdir(exist_ok=True)
        p=pred if seed is None else pred[pred.seed==seed]
        m=met if seed is None else met[met.seed==seed]
        c=cov if seed is None else cov[cov.seed==seed]
        for name,df in [('predictions',p),('fold_metrics',m),('interval_coverage',c)]:df.to_csv(folder/(name+'.csv'),index=False,encoding='utf-8-sig')
        chosen=m[(m.family=='selected')&(m.space!='log1p')]
        perf=chosen.groupby('target')[['r2','mae','rmse']].agg(['mean','std']).reindex(TARGETS)
        perf.columns=['_'.join(c) for c in perf.columns]
        perf.reset_index().to_csv(folder/'performance.csv',index=False,encoding='utf-8-sig')
        for target,short in zip(TARGETS,SHORT):
            q=p[p.target==target];assert len(q)==(196 if seed else 980)
            z=np.sort(q.standardized_residual_log.to_numpy());x=norm.ppf((np.arange(len(z))+.5)/len(z))
            qq=pd.DataFrame({'normal_quantile':x,'residual_quantile':z})
            qq.to_csv(folder/('qq_'+short+'.csv'),index=False,encoding='utf-8-sig')
    print('Exported seed-42 and five-seed result tables.')
if __name__=='__main__':main()
