"""Audit saved partitions, model provenance, metric exports and optimizer feasibility."""
import os
for var in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[var]="1"
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from ml_core import ROOT,FEATURES,TARGETS,dataset,dump,sha256,metrics,predict_bundle,calibrate


def check_search(folder,allowed,forbidden):
    meta=json.loads((folder/"search.json").read_text())
    for fold in meta["folds"]:
        train=set(fold["train_rows"]); valid=set(fold["validation_rows"])
        assert not train&valid and train|valid==allowed
        assert not (train|valid)&forbidden
    saved=joblib.load(folder/"models.joblib")
    df=dataset(); ids=sorted(allowed)
    for family,bundle in saved.items():
        for target in TARGETS:
            np.testing.assert_allclose(bundle[target].context.x,df.iloc[ids][FEATURES].to_numpy(float),rtol=0,atol=0)
    return len(saved)


def check_metrics(folder):
    predictions=pd.read_csv(folder/"predictions.csv")
    table=pd.read_csv(folder/"metrics.csv")
    checked=0
    for (family,target),p in predictions.groupby(["family","target"]):
        row=table[(table.family==family)&(table.target==target)&(table.space!="log1p")].iloc[0]
        values=metrics(p.actual.to_numpy(),p.prediction.to_numpy())
        for metric,value in values.items(): np.testing.assert_allclose(row[metric],value,rtol=1e-8,atol=1e-8,equal_nan=True)
        checked+=1
    return checked


def main():
    manifest=json.loads((ROOT/"provenance/input_and_environment.json").read_text())
    assert sha256(ROOT/"data.csv")==manifest["input_sha256"]
    checks=0; models=0; splits=0
    for stage,expected in [("nested",25),("blocked",24),("learning",20)]:
        folders=[p for p in (ROOT/"validation"/stage).iterdir() if p.is_dir() and (p/"complete.json").exists()]
        assert len(folders)==expected,(stage,len(folders))
        for folder in folders:
            split=json.loads((folder/"split.json").read_text())
            train=set(split["train_ids"]); test=set(split["test_ids"])
            assert not train&test
            models+=check_search(folder/"search",train,test)
            checks+=check_metrics(folder); splits+=1
            if stage=="nested":
                cal=json.loads((folder/"calibration/split.json").read_text())
                proper=set(cal["proper_train_ids"]); calibration=set(cal["calibration_ids"])
                assert proper|calibration==train and not proper&calibration
                models+=check_search(folder/"calibration/search",proper,calibration|test)
                checks+=check_metrics(folder/"calibration")
    nested=pd.read_csv(ROOT/"validation/nested/all_predictions.csv")
    assert not nested.duplicated(["family","repeat","target","row_id"]).any()
    assert nested.groupby(["family","repeat","target"]).size().eq(196).all()
    assert nested.groupby(["family","target","row_id"]).size().eq(5).all()
    # A full-size learning point must reproduce its identical first outer fold.
    learning=pd.read_csv(ROOT/"validation/learning/all_metrics.csv")
    outer=pd.read_csv(ROOT/"validation/nested/all_metrics.csv")
    keys=["seed","family","target","space"]
    merged=learning[learning.fraction==1].merge(outer[outer.fold==1],on=keys,suffixes=("_learn","_outer"))
    for metric in ["r2","mae","rmse"]:
        np.testing.assert_allclose(merged[metric+"_learn"],merged[metric+"_outer"],rtol=1e-7,atol=1e-7,equal_nan=True)
    final=json.loads((ROOT/"models/calibrated_design/split.json").read_text())
    proper=set(final["proper_train_ids"]); cal=set(final["calibration_ids"])
    assert not proper&cal and len(proper|cal)==196
    models+=check_search(ROOT/"models/calibrated_design",proper,cal)
    models+=check_search(ROOT/"models/full_data",set(range(196)),set())
    bundle=joblib.load(ROOT/"models/calibrated_design/models.joblib")["selected"]
    df=dataset(); ids=np.array(sorted(cal))
    expected=calibrate(bundle,df.iloc[ids][FEATURES].to_numpy(float),df.iloc[ids][TARGETS].to_numpy(float),ids)
    actual=json.loads((ROOT/"models/calibrated_design/calibration.json").read_text())
    np.testing.assert_allclose(actual["absolute_scores"],expected["absolute_scores"],rtol=1e-10,atol=1e-10)
    runs=pd.read_csv(ROOT/"optimization/all_optimizer_runs.csv")
    assert np.equal(runs.Charge,np.round(runs.Charge)).all()
    for feature in FEATURES:
        assert (runs[feature]>=df[feature].min()-1e-9).all() and (runs[feature]<=df[feature].max()+1e-9).all()
    constrained=runs[runs.water_min.notna()&runs.feasible]
    assert (constrained.water_constraint_value>=constrained.water_min-1e-7).all()
    assert (constrained.li_constraint_value>=constrained.li_min-1e-7).all()
    shap=pd.read_csv(ROOT/"interpretation/attribution_values.csv")
    group=shap.groupby(["model","background","fold_id","target","row_id"])
    errors=group.shap_value.sum()+group.base_value.first()-group.prediction.first()
    assert errors.abs().max()<1e-8
    verification={"input_sha256":sha256(ROOT/"data.csv"),"input_unchanged":True,
                  "outer_splits_checked":splits,
                  "saved_model_bundles_checked":models,"metric_groups_recomputed":checks,
                  "no_inner_outer_or_calibration_overlap":True,"learning_full_size_matches_outer_fold":True,
                  "calibration_scores_recomputed":True,"optimizer_integrality_bounds_feasibility_checked":True,
                  "exported_shap_max_additivity_error":float(errors.abs().max()),
                  "source_hashes":{p.name:sha256(p) for p in ROOT.glob("*.py")}}
    dump(ROOT/"provenance/final_verification.json",verification)
    print(json.dumps(verification,indent=2),flush=True)


if __name__=="__main__":
    main()
