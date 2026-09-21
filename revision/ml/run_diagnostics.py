"""Outer-test diagnostics, matched calibration comparison and model selection counts."""
import os
for var in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[var]="1"
import json
import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from threadpoolctl import threadpool_limits

from ml_core import ROOT,FEATURES,TARGETS,OUTPUTS,dataset,dump,predict_bundle,truth_logs,metrics
from run_validation import result_tables

OUT=ROOT/"diagnostics"


def main():
    OUT.mkdir(exist_ok=True)
    df=dataset(); x=df[FEATURES].to_numpy(float); raw=df[TARGETS].to_numpy(float)
    root=ROOT/"validation"/"nested"
    folders=sorted(root.glob("repeat_*_fold_*"))
    if len(folders)!=25 or not all((p/"complete.json").exists() for p in folders):
        raise RuntimeError("All 25 outer folds must be complete before diagnostics.")
    selected=[]
    native_m=[]; native_p=[]; native_c=[]
    with threadpool_limits(limits=1):
        for folder in folders:
            split=json.loads((folder/"split.json").read_text())
            for role,search in [("outer",folder/"search"),("proper",folder/"calibration"/"search")]:
                meta=json.loads((search/"search.json").read_text())
                for target,ci in zip(TARGETS,meta["choices"]["selected"]):
                    selected.append({"role":role,"fold_id":folder.name,"target":target,
                                     "family":meta["specs"][ci]["family"],"candidate":ci})
            cal_split=json.loads((folder/"calibration"/"split.json").read_text())
            model=joblib.load(folder/"calibration"/"search"/"models.joblib")["selected"]
            test=np.array(split["test_ids"])
            tag={"stage":"nested","repeat":split["repeat"],"seed":split["seed"],"fold":split["fold"],
                 "family":"selected_proper_uncalibrated","n_train":len(cal_split["proper_train_ids"]),
                 "n_calibration":len(cal_split["calibration_ids"]),"n_test":len(test)}
            m,p,c=result_tables(model,x[test],raw[test],test,tag)
            native_m+=m; native_p+=p; native_c+=c
    pd.DataFrame(native_m).to_csv(OUT/"proper_model_native_metrics.csv",index=False)
    pd.DataFrame(native_p).to_csv(OUT/"proper_model_native_predictions.csv",index=False)
    pd.DataFrame(native_c).to_csv(OUT/"proper_model_native_coverage.csv",index=False)
    selection=pd.DataFrame(selected)
    selection.to_csv(OUT/"selected_model_per_fold.csv",index=False)
    selection.groupby(["role","target","family"]).size().rename("n_folds").reset_index().to_csv(
        OUT/"model_selection_frequencies.csv",index=False)

    predictions=pd.read_csv(root/"all_predictions.csv")
    predictions["residual_log"]=predictions.actual_log-predictions.prediction_log
    predictions["standardized_residual_log"]=predictions.residual_log/predictions.scale_log
    predictions[predictions.family=="selected"].to_csv(OUT/"selected_outer_residuals.csv",index=False)
    qq=[]; dispersion=[]
    for family in predictions.family.unique():
        for target in OUTPUTS:
            g=predictions[(predictions.family==family)&(predictions.target==target)]
            if not len(g): continue
            z=np.sort(g.standardized_residual_log.to_numpy())
            theoretical=norm.ppf((np.arange(len(z))+.5)/len(z))
            for a,b in zip(theoretical,z): qq.append({"family":family,"target":target,"normal_quantile":a,"residual_quantile":b})
            if target in TARGETS:
                pearson=(g.actual-g.prediction)**2/np.maximum(g.prediction,1e-8)
                dispersion.append({"family":family,"target":target,"pearson_residual_second_moment":pearson.mean(),
                                   "mean_count_residual":(g.actual-g.prediction).mean(),
                                   "mean_log_residual":g.residual_log.mean(),"zero_observations":int((g.actual==0).sum()),
                                   "prediction_rows":len(g),"distinct_systems":g.row_id.nunique()})
    pd.DataFrame(qq).to_csv(OUT/"normal_qq_source.csv",index=False)
    pd.DataFrame(dispersion).to_csv(OUT/"count_residual_dispersion.csv",index=False)
    pair=predictions[(predictions.family=="selected")&predictions.target.isin(["FLi","FMg"])].pivot(
        index=["repeat","fold","row_id"],columns="target",values="residual_log")
    dump(OUT/"li_mg_residual_correlation.json",{"pearson":float(pair.corr().loc["FLi","FMg"]),
                                              "n_prediction_pairs":len(pair),"n_distinct_systems":196,
                                              "interpretation":"descriptive repeated-OOF residual dependence; no independent-sample p-value"})
    coverage=pd.read_csv(root/"all_coverage.csv")
    coverage=pd.concat([coverage,pd.DataFrame(native_c)],ignore_index=True)
    coverage.to_csv(OUT/"coverage_all_methods.csv",index=False)
    coverage.groupby(["family","target","nominal","interval"])[["coverage","mean_width_log"]].agg(
        ["mean","std","min","max"]).to_csv(OUT/"coverage_summary.csv")
    bins=[]
    for family in ["selected","selected_split_calibrated"]:
        for target in ["FLi","FCl","FMg"]:
            g=predictions[(predictions.family==family)&(predictions.target==target)].copy()
            g["count_bin"]=pd.cut(g.actual,[-.1,.1,5,20,np.inf],labels=["zero","1-5","6-20",">20"])
            for label,block in g.groupby("count_bin",observed=True):
                hit=(block.actual_log>=block["lower_log_0.95"])&(block.actual_log<=block["upper_log_0.95"])
                bins.append({"family":family,"target":target,"count_bin":str(label),"n_predictions":len(block),
                             "n_distinct_systems":block.row_id.nunique(),"coverage_95":hit.mean(),
                             "log_rmse":np.sqrt(np.mean(block.residual_log**2)),"log_bias":block.residual_log.mean()})
    pd.DataFrame(bins).to_csv(OUT/"low_count_diagnostics.csv",index=False)
    metrics_df=pd.read_csv(root/"all_metrics.csv")
    score=metrics_df[metrics_df.target=="ln_selectivity"].pivot(index=["repeat","fold"],columns="family",values="rmse")
    contrasts=[]
    for a,b in [("gpr_mean","quadratic"),("gpr_mean","poisson"),("gpr_hetero","gpr_mean"),
                ("gpr_mean","gpr_median"),("selected","poisson")]:
        difference=score[a]-score[b]
        contrasts.append({"model_a":a,"model_b":b,"mean_rmse_difference_a_minus_b":difference.mean(),
                          "sd_fold_difference":difference.std(),"fraction_folds_a_better":(difference<0).mean(),
                          "n_outer_folds":len(difference)})
    pd.DataFrame(contrasts).to_csv(OUT/"paired_selectivity_comparisons.csv",index=False)
    dump(OUT/"interpretation_notes.json",{
        "normal_qq":"Descriptive reference for transformed residuals; count data and selected-model mixtures need not be Gaussian.",
        "dispersion":"Predictive Pearson residuals include model error; these are not replicate-based trajectory variances.",
        "calibration_comparison":"Native versus conformal intervals use the exact same proper-trained selected model and test observations.",
        "statistics":"No independent-sample p-values or confidence intervals are assigned to repeated-fold observations."})
    print("Outer-test diagnostic tables complete.",flush=True)


if __name__=="__main__":
    main()
