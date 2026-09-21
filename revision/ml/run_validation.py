"""Run nested, blocked, learning-curve and final-model analyses."""
import os
for var in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[var]="1"
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from threadpoolctl import threadpool_limits

from ml_core import (ROOT,FEATURES,TARGETS,OUTPUTS,LEVELS,dataset,dump,sha256,
                     candidate_specs,search_and_fit,predict_bundle,truth_logs,metrics,
                     calibrate,calibrated_intervals,joint_lower)


def result_tables(bundle,x,raw,ids,tag,levels=LEVELS,calibration=None):
    p=predict_bundle(bundle,x,levels if calibration is None else ())
    true=truth_logs(raw)
    metric_rows=[]
    pred_rows=[]
    coverage_rows=[]
    for ti,target in enumerate(OUTPUTS):
        a=raw[:,ti] if ti<4 else true[:,ti]
        b=p["count"][:,ti] if ti<4 else p["log"][:,ti]
        metric_rows.append({**tag,"target":target,"space":"count" if ti<4 else "log_ratio",**metrics(a,b)})
        if ti<4:
            metric_rows.append({**tag,"target":target,"space":"log1p",**metrics(true[:,ti],p["log"][:,ti])})
        for i,idx in enumerate(ids):
            pred_rows.append({**tag,"target":target,"row_id":int(idx),"actual":a[i],"prediction":b[i],
                              "actual_log":true[i,ti],"prediction_log":p["log"][i,ti],
                              "scale_log":p["scale"][i,ti]})
    for level in levels:
        if calibration is None:
            lower,upper=p[f"lower_{level}"],p[f"upper_{level}"]
        else:
            lower,upper=calibrated_intervals(p,calibration,level)
        for ti,target in enumerate(OUTPUTS):
            covered=(true[:,ti]>=lower[:,ti])&(true[:,ti]<=upper[:,ti])
            coverage_rows.append({**tag,"target":target,"nominal":level,
                                  "coverage":float(covered.mean()),
                                  "mean_width_log":float(np.mean(upper[:,ti]-lower[:,ti])),
                                  "interval":"split_conformal" if calibration else "native"})
            for i in range(len(ids)):
                pred_rows[ti*len(ids)+i][f"lower_log_{level}"]=lower[i,ti]
                pred_rows[ti*len(ids)+i][f"upper_log_{level}"]=upper[i,ti]
        if calibration:
            lower=joint_lower(p,calibration,level)
            covered=(true[:,[0,1,4]]>=lower[:,[0,1,4]]).all(axis=1)
            coverage_rows.append({**tag,"target":"joint_water_li_score","nominal":level,
                                  "coverage":float(covered.mean()),"mean_width_log":np.nan,
                                  "interval":"joint_one_sided"})
    return metric_rows,pred_rows,coverage_rows


def save_tables(directory,metrics_rows,predictions,coverage):
    pd.DataFrame(metrics_rows).to_csv(directory/"metrics.csv",index=False)
    pd.DataFrame(predictions).to_csv(directory/"predictions.csv",index=False)
    pd.DataFrame(coverage).to_csv(directory/"coverage.csv",index=False)


def evaluate_split(df,train_ids,test_ids,directory,tag,split_calibration=False):
    directory.mkdir(parents=True,exist_ok=True)
    x=df[FEATURES].to_numpy(float)
    raw=df[TARGETS].to_numpy(float)
    assert not np.intersect1d(train_ids,test_ids).size
    dump(directory/"split.json",{**tag,"train_ids":train_ids,"test_ids":test_ids,
                                 "data_sha256":sha256(ROOT/"data.csv")})
    fitted=search_and_fit(x[train_ids],raw[train_ids],train_ids,directory/"search")
    metric_rows,pred_rows,coverage_rows=[],[],[]
    for family,bundle in fitted.items():
        m,p,c=result_tables(bundle,x[test_ids],raw[test_ids],test_ids,
                            {**tag,"family":family,"n_train":len(train_ids),"n_test":len(test_ids)})
        metric_rows+=m; pred_rows+=p; coverage_rows+=c
    save_tables(directory,metric_rows,pred_rows,coverage_rows)
    if split_calibration:
        proper,cal=train_test_split(train_ids,test_size=.2,random_state=tag["seed"]*100+tag["fold"])
        proper,cal=np.sort(proper),np.sort(cal)
        cal_dir=directory/"calibration"
        cal_dir.mkdir(exist_ok=True)
        dump(cal_dir/"split.json",{"proper_train_ids":proper,"calibration_ids":cal,"test_ids":test_ids})
        cal_models=search_and_fit(x[proper],raw[proper],proper,cal_dir/"search",all_families=False)
        calibrated=calibrate(cal_models["selected"],x[cal],raw[cal],cal)
        dump(cal_dir/"calibration.json",calibrated)
        m,p,c=result_tables(cal_models["selected"],x[test_ids],raw[test_ids],test_ids,
                            {**tag,"family":"selected_split_calibrated","n_train":len(proper),
                             "n_calibration":len(cal),"n_test":len(test_ids)},calibration=calibrated)
        save_tables(cal_dir,m,p,c)
    dump(directory/"complete.json",{"data_sha256":sha256(ROOT/"data.csv"),"tag":tag})


def aggregate(stage):
    root=ROOT/"validation"/stage
    for name in ["metrics","predictions","coverage"]:
        paths=list(root.glob(f"*/{name}.csv"))+list(root.glob(f"*/calibration/{name}.csv"))
        if not paths:
            continue
        df=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)
        df.to_csv(root/f"all_{name}.csv",index=False)
        if name=="metrics":
            group=["family","target","space"]
            if stage=="blocked": group+=["kind"]
            if stage=="learning": group+=["fraction"]
            summary=df.groupby(group)[["r2","mae","rmse"]].agg(["mean","std","median","min","max","count"])
            summary.to_csv(root/"metrics_summary.csv")


def nested(df,resume):
    root=ROOT/"validation"/"nested"
    for repeat,seed in enumerate(range(42,47),1):
        for fold,(tr,te) in enumerate(KFold(5,shuffle=True,random_state=seed).split(df),1):
            directory=root/f"repeat_{repeat}_fold_{fold}"
            if resume and (directory/"complete.json").exists(): continue
            start=time.time()
            evaluate_split(df,tr,te,directory,{"stage":"nested","repeat":repeat,"seed":seed,"fold":fold},True)
            print(f"nested repeat {repeat}/5 fold {fold}/5: {time.time()-start:.1f}s",flush=True)
    aggregate("nested")


def blocked_splits(df):
    ids=np.arange(len(df))
    for col in FEATURES:
        levels=sorted(df[col].unique())
        for j,level in enumerate(levels):
            mask=np.isclose(df[col],level)
            kind="boundary_extrapolation" if j in (0,len(levels)-1) else "interior_level"
            yield ids[~mask],ids[mask],{"name":f"{col}_{j}","kind":kind,"descriptor":col,"level":float(level)}
    for a,b in [(0,1),(0,2),(1,2)]:
        for high in [False,True]:
            ca,cb=FEATURES[a],FEATURES[b]
            if high:
                mask=(df[ca]>=df[ca].median())&(df[cb]>=df[cb].median())
            else:
                mask=(df[ca]<=df[ca].median())&(df[cb]<=df[cb].median())
            yield ids[~mask],ids[mask],{"name":f"corner_{ca}_{cb}_{int(high)}","kind":"two_descriptor_block",
                                      "descriptor":ca+"_"+cb,"level":"high" if high else "low"}


def blocked(df,resume):
    for fold,(tr,te,tag) in enumerate(blocked_splits(df),1):
        directory=ROOT/"validation"/"blocked"/tag["name"]
        if resume and (directory/"complete.json").exists(): continue
        start=time.time()
        evaluate_split(df,tr,te,directory,{"stage":"blocked","fold":fold,"seed":42,**tag})
        print(f"blocked {fold}/24 {tag['name']}: {time.time()-start:.1f}s",flush=True)
    aggregate("blocked")


def learning(df,resume):
    for seed in range(42,47):
        outer_train,test=next(KFold(5,shuffle=True,random_state=seed).split(df))
        pool=np.random.default_rng(seed+1000).permutation(outer_train)
        for fraction in [.25,.5,.75,1.]:
            tr=np.sort(pool[:max(20,int(round(len(pool)*fraction)))])
            directory=ROOT/"validation"/"learning"/f"seed_{seed}_fraction_{fraction}"
            if resume and (directory/"complete.json").exists(): continue
            start=time.time()
            evaluate_split(df,tr,test,directory,{"stage":"learning","seed":seed,"fold":1,"fraction":fraction})
            print(f"learning seed {seed}, fraction {fraction}: {time.time()-start:.1f}s",flush=True)
    aggregate("learning")


def final_models(df):
    x=df[FEATURES].to_numpy(float)
    raw=df[TARGETS].to_numpy(float)
    ids=np.arange(len(df))
    full_dir=ROOT/"models"/"full_data"
    fits=search_and_fit(x,raw,ids,full_dir)
    dump(full_dir/"role.json",{"role":"full-data mean screening; not an independent test model","rows":ids})
    selected=fits["selected"]
    p=predict_bundle(selected,x)
    out=df.copy()
    for i,target in enumerate(TARGETS): out[target+"_fitted_mean"]=p["count"][:,i]
    out["fitted_log_selectivity"]=p["log"][:,4]
    out.to_csv(full_dir/"fitted_predictions.csv",index=False)
    proper,cal=train_test_split(ids,test_size=.2,random_state=2026)
    proper,cal=np.sort(proper),np.sort(cal)
    cal_dir=ROOT/"models"/"calibrated_design"
    calibrated_fits=search_and_fit(x[proper],raw[proper],proper,cal_dir,all_families=False)
    calibration=calibrate(calibrated_fits["selected"],x[cal],raw[cal],cal)
    dump(cal_dir/"calibration.json",calibration)
    dump(cal_dir/"split.json",{"proper_train_ids":proper,"calibration_ids":cal,"seed":2026,
                               "role":"separate split-calibrated screening model; no full-data refit"})
    print("Final full-data and separately calibrated design models saved.",flush=True)


def prepare():
    df=dataset()
    (ROOT/"provenance").mkdir(exist_ok=True)
    audit={"input_sha256":sha256(ROOT/"data.csv"),"rows":len(df),
           "columns":list(df.columns),
           "zeros":{t:int((df[t]==0).sum()) for t in TARGETS},
           "python":sys.version,"packages":{p:importlib.metadata.version(p) for p in
                                              ["numpy","pandas","scipy","scikit-learn","joblib"]},
           "source_hashes":{p.name:sha256(p) for p in ROOT.glob("*.py")}}
    manifest=ROOT/"provenance"/"input_and_environment.json"
    if manifest.exists() and json.loads(manifest.read_text())["input_sha256"]!=audit["input_sha256"]:
        raise RuntimeError("Input data changed after analysis started; use a new run directory.")
    dump(manifest,audit)
    dump(ROOT/"provenance"/"candidate_specs.json",candidate_specs())
    return df


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=["nested","blocked","learning","final","all"],default="all")
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    df=prepare()
    start=time.time()
    with threadpool_limits(limits=1):
        for name,fn in [("nested",nested),("blocked",blocked),("learning",learning),("final",final_models)]:
            if args.stage in (name,"all"):
                fn(df) if name=="final" else fn(df,args.resume)
    dump(ROOT/"provenance"/("completed_"+args.stage+".json"),{"elapsed_seconds":time.time()-start,
                                                               "input_sha256":sha256(ROOT/"data.csv")})


if __name__=="__main__":
    main()
