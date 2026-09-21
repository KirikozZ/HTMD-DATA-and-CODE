"""Exact three-feature attributions, interactions and partial dependence."""
import os
for var in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[var]="1"
from itertools import combinations
import json
import math

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ml_core import ROOT,FEATURES,TARGETS,OUTPUTS,dataset,dump,predict_bundle,sha256

OUT=ROOT/"interpretation"


def exact_shap(bundle,explained,background):
    n,b=len(explained),len(background)
    mixed=[]
    for mask in range(8):
        z=np.tile(background,(n,1))
        for j in range(3):
            if mask & (1<<j): z[:,j]=np.repeat(explained[:,j],b)
        mixed.append(z)
    # Grid hybrids repeat heavily: evaluate each unique descriptor vector once.
    unique,inverse=np.unique(np.vstack(mixed),axis=0,return_inverse=True)
    values=predict_bundle(bundle,unique)["log"][inverse].reshape(8,n,b,5).mean(axis=2)
    phi=np.zeros((n,5,3))
    interaction=np.zeros((n,5,3,3))
    for j in range(3):
        for mask in range(8):
            if mask & (1<<j): continue
            size=mask.bit_count()
            weight=math.factorial(size)*math.factorial(2-size)/math.factorial(3)
            phi[:,:,j]+=weight*(values[mask|(1<<j)]-values[mask])
    for j,k in combinations(range(3),2):
        for mask in range(8):
            if mask & ((1<<j)|(1<<k)): continue
            size=mask.bit_count()
            weight=math.factorial(size)*math.factorial(1-size)/(2*math.factorial(2))
            interaction[:,:,j,k]+=weight*(values[mask|(1<<j)|(1<<k)]-values[mask|(1<<j)]
                                          -values[mask|(1<<k)]+values[mask])
        interaction[:,:,k,j]=interaction[:,:,j,k]
    for j in range(3): interaction[:,:,j,j]=phi[:,:,j]-interaction[:,:,j,:].sum(axis=2)
    assert np.allclose(phi.sum(axis=2)+values[0],values[7],atol=1e-9,rtol=1e-9)
    assert np.allclose(interaction.sum(axis=(2,3)),phi.sum(axis=2),atol=1e-9,rtol=1e-9)
    return phi,interaction,values[0],values[7]


def balanced_background(x,seed):
    rng=np.random.default_rng(seed)
    rows=[]
    for level in np.unique(x[:,0]):
        ids=np.flatnonzero(np.isclose(x[:,0],level))
        rows.extend(rng.choice(ids,min(8,len(ids)),replace=False).tolist())
    return x[np.array(rows)]


def export_explanation(bundle,x,bg,ids,tag):
    phi,inter,base,pred=exact_shap(bundle,x,bg)
    importance=[]
    details=[]
    interactions=[]
    for ti,target in enumerate(OUTPUTS):
        avg=np.abs(phi[:,ti,:]).mean(axis=0)
        relative=avg/max(avg.sum(),1e-12)
        for j,feature in enumerate(FEATURES):
            importance.append({**tag,"target":target,"feature":feature,
                               "mean_abs_shap":avg[j],"relative_importance":relative[j],
                               "n_explained":len(x),"n_background":len(bg)})
            for i,idx in enumerate(ids):
                details.append({**tag,"target":target,"feature":feature,"row_id":str(idx),
                                "feature_value":x[i,j],"shap_value":phi[i,ti,j],
                                "base_value":base[i,ti],"prediction":pred[i,ti]})
        for j,k in combinations(range(3),2):
            interactions.append({**tag,"target":target,"feature_a":FEATURES[j],"feature_b":FEATURES[k],
                                 "mean_abs_interaction_entry":np.abs(inter[:,ti,j,k]).mean(),
                                 "mean_abs_pair_total":np.abs(2*inter[:,ti,j,k]).mean(),
                                 "mean_signed_pair_total":(2*inter[:,ti,j,k]).mean(),
                                 "n_explained":len(x)})
    return importance,details,interactions


def pdp(bundle,x):
    one=[]; two=[]
    for j,feature in enumerate(FEATURES):
        grid=np.arange(-9,10) if j==1 else np.linspace(x[:,j].min(),x[:,j].max(),61)
        z=np.tile(x,(len(grid),1))
        z[:,j]=np.repeat(grid,len(x))
        unique,inverse=np.unique(z,axis=0,return_inverse=True)
        p=predict_bundle(bundle,unique)["log"][inverse].reshape(len(grid),len(x),5).mean(axis=1)
        for i,value in enumerate(grid):
            for ti,target in enumerate(OUTPUTS): one.append({"feature":feature,"value":value,"target":target,"partial_dependence":p[i,ti]})
    for a,b in combinations(range(3),2):
        ga=np.arange(-9,10) if a==1 else np.linspace(x[:,a].min(),x[:,a].max(),31)
        gb=np.arange(-9,10) if b==1 else np.linspace(x[:,b].min(),x[:,b].max(),31)
        grid=np.array([(i,j) for i in ga for j in gb])
        z=np.tile(x,(len(grid),1))
        z[:,a]=np.repeat(grid[:,0],len(x)); z[:,b]=np.repeat(grid[:,1],len(x))
        unique,inverse=np.unique(z,axis=0,return_inverse=True)
        p=predict_bundle(bundle,unique)["log"][inverse].reshape(len(grid),len(x),5).mean(axis=1)
        for i,(va,vb) in enumerate(grid):
            for ti,target in enumerate(OUTPUTS):
                two.append({"feature_a":FEATURES[a],"feature_b":FEATURES[b],"value_a":va,"value_b":vb,
                            "target":target,"partial_dependence":p[i,ti]})
    pd.DataFrame(one).to_csv(OUT/"partial_dependence_1d.csv",index=False)
    pd.DataFrame(two).to_csv(OUT/"partial_dependence_2d.csv",index=False)


def main():
    OUT.mkdir(exist_ok=True)
    df=dataset(); x=df[FEATURES].to_numpy(float)
    full=joblib.load(ROOT/"models"/"full_data"/"models.joblib")["selected"]
    all_i=[]; all_d=[]; all_int=[]
    with threadpool_limits(limits=1):
        backgrounds={"full_grid":x,"balanced_32":balanced_background(x,42),
                     "high_sigma":x[x[:,2]>=np.sort(np.unique(x[:,2]))[-2]]}
        for name,bg in backgrounds.items():
            i,d,it=export_explanation(full,x,bg,np.arange(len(x)),{"model":"full_data","background":name,"fold_id":"full"})
            all_i+=i; all_d+=d; all_int+=it
        for folder in sorted((ROOT/"validation"/"nested").glob("repeat_*_fold_*")):
            split=json.loads((folder/"split.json").read_text())
            train,test=np.array(split["train_ids"]),np.array(split["test_ids"])
            model=joblib.load(folder/"search"/"models.joblib")["selected"]
            for name,bg in {"outer_training":x[train],"balanced_32":balanced_background(x[train],42)}.items():
                i,d,it=export_explanation(model,x[test],bg,test,
                                          {"model":"outer_selected","background":name,"fold_id":folder.name})
                all_i+=i; all_d+=d; all_int+=it
            print("Attributions and interactions:",folder.name,flush=True)
        pd.DataFrame(all_i).to_csv(OUT/"global_importance.csv",index=False)
        pd.DataFrame(all_d).to_csv(OUT/"attribution_values.csv",index=False)
        pd.DataFrame(all_int).to_csv(OUT/"pair_interactions.csv",index=False)
        runs=pd.read_csv(ROOT/"optimization"/"all_optimizer_runs.csv")
        local=[]
        points=[("old_reported",np.array([.6554,-2,4.2608]))]
        for label in ["full_mean_unconstrained","calibrated_lcb95_unconstrained"]:
            group=runs[(runs.label==label)&runs.feasible]
            if len(group):
                best=group.sort_values("screening_objective",ascending=False).iloc[0]
                points.append((label,best[FEATURES].to_numpy(float)))
        for name,point in points:
            _,details,_=export_explanation(full,point[None,:],x,[name],
                                          {"model":"full_data","background":"full_grid","fold_id":"local"})
            local+=details
        pd.DataFrame(local).to_csv(OUT/"local_candidates.csv",index=False)
        pdp(full,x)
    importance=pd.DataFrame(all_i)
    importance[importance.model=="outer_selected"].groupby(["background","target","feature"])["relative_importance"].agg(
        ["mean","std","min","max"]).to_csv(OUT/"attribution_stability_summary.csv")
    dump(OUT/"metadata.json",{"method":"exact interventional Shapley values and pair interactions",
                              "scale":"log1p predicted count means and their Li-minus-Mg difference",
                              "physical_causality":False,"pdp_background":"full 196-system factorial grid",
                              "interaction_convention":"off-diagonal entries are symmetric; pair total is twice one entry",
                              "local_candidate_model":"full-data mean model, including when the location was selected by the separate LCB model",
                              "additivity_checked":True,"data_sha256":sha256(ROOT/"data.csv")})


if __name__=="__main__":
    main()
