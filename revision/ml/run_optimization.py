"""Mean/Pareto and separately calibrated constrained design screening."""
import os
for var in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[var]="1"
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, NonlinearConstraint
from scipy.stats import qmc
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from threadpoolctl import threadpool_limits

from ml_core import (ROOT,FEATURES,TARGETS,dataset,dump,sha256,predict_bundle,joint_lower)

OUT=ROOT/"optimization"
OLD=np.array([.6554,-2.,4.2608])


class DesignEvaluator:
    def __init__(self,bundle,calibration=None,pseudocount=1.):
        self.bundle=bundle
        self.calibration=calibration
        self.pseudocount=pseudocount

    def evaluate(self,x):
        x=np.asarray(x,float).reshape(-1,3)
        p=predict_bundle(self.bundle,x)
        score=np.log((p["count"][:,1]+self.pseudocount)/(p["count"][:,3]+self.pseudocount))
        if self.calibration is None:
            water,li=p["count"][:,0],p["count"][:,1]
        else:
            if self.pseudocount!=1:
                raise ValueError("Calibration targets pseudocount 1 only.")
            lo=joint_lower(p,self.calibration,.95)
            water,li=np.expm1(lo[:,0]),np.expm1(lo[:,1])
            score=lo[:,4]
        return score,water,li,p


def point_record(x,evaluator):
    score,water,li,p=evaluator.evaluate(np.asarray(x)[None,:])
    return {**dict(zip(FEATURES,map(float,x))),
            **{t+"_predicted_mean":float(p["count"][0,i]) for i,t in enumerate(TARGETS)},
            "predicted_log_score":float(p["log"][0,4]),"screening_objective":float(score[0]),
            "water_constraint_value":float(water[0]),"li_constraint_value":float(li[0]),
            "score_scale_log":float(p["scale"][0,4])}


def optimize(evaluator,bounds,seed,label,thresholds=None):
    traces=[]
    def objective(population):
        values=-evaluator.evaluate(np.asarray(population).reshape(3,-1).T)[0]
        return float(values[0]) if np.asarray(population).ndim==1 else values
    def callback(intermediate_result):
        traces.append({"generation":len(traces)+1,"negative_objective":float(intermediate_result.fun),
                       **dict(zip(FEATURES,map(float,intermediate_result.x)))})
    constraints=()
    if thresholds is not None:
        def constraints_value(population):
            _,w,li,_=evaluator.evaluate(np.asarray(population).reshape(3,-1).T)
            values=np.vstack([w,li])
            return values[:,0] if np.asarray(population).ndim==1 else values
        constraints=(NonlinearConstraint(constraints_value,np.asarray(thresholds),np.full(2,np.inf)),)
    start=time.time()
    result=differential_evolution(objective,bounds,rng=np.random.default_rng(seed),
                                  integrality=[False,True,False],maxiter=200,popsize=20,
                                  init="latinhypercube",tol=1e-7,atol=1e-8,
                                  mutation=(.5,1.),recombination=.7,polish=False,
                                  vectorized=True,updating="deferred",constraints=constraints,
                                  callback=callback,x0=OLD)
    record=point_record(result.x,evaluator)
    feasible=thresholds is None or (record["water_constraint_value"]>=thresholds[0]-1e-8 and
                                    record["li_constraint_value"]>=thresholds[1]-1e-8)
    record.update(label=label,seed=seed,feasible=bool(feasible),solver_success=bool(result.success),
                  solver_message=str(result.message),generations=int(result.nit),
                  population_evaluations=int(result.nfev),elapsed_seconds=time.time()-start,
                  water_min=float(thresholds[0]) if thresholds else None,
                  li_min=float(thresholds[1]) if thresholds else None)
    dump(OUT/"runs"/(label+"_seed_"+str(seed)+".json"),record)
    pd.DataFrame(traces).to_csv(OUT/"runs"/(label+"_seed_"+str(seed)+"_trace.csv"),index=False)
    np.savez_compressed(OUT/"runs"/(label+"_seed_"+str(seed)+"_population.npz"),
                        population=result.population,energies=result.population_energies)
    print(f"Optimization {label}, seed {seed}: objective={record['screening_objective']:.4f}, feasible={feasible}",flush=True)
    return record


def lhs_candidates(bounds):
    blocks=[]
    for charge in range(int(bounds[1,0]),int(bounds[1,1])+1):
        design=qmc.LatinHypercube(2,seed=42+charge+9).random(1024)
        hs=qmc.scale(design,bounds[[0,2],0],bounds[[0,2],1])
        blocks.append(np.column_stack([hs[:,0],np.full(len(hs),charge),hs[:,1]]))
    return np.vstack(blocks)


def main():
    OUT.mkdir(exist_ok=True)
    (OUT/"runs").mkdir(exist_ok=True)
    df=dataset()
    x=df[FEATURES].to_numpy(float)
    bounds=np.column_stack([x.min(axis=0),x.max(axis=0)])
    full=joblib.load(ROOT/"models"/"full_data"/"models.joblib")["selected"]
    calibrated=joblib.load(ROOT/"models"/"calibrated_design"/"models.joblib")["selected"]
    calibration=json.loads((ROOT/"models"/"calibrated_design"/"calibration.json").read_text())
    evaluators={"full_mean":DesignEvaluator(full),"calibrated_lcb95":DesignEvaluator(calibrated,calibration)}
    mask=np.isclose(x[:,0],1.)&np.isclose(x[:,1],0.)&np.isclose(x[:,2],2.846421)
    if mask.sum()!=1: raise ValueError("Unique baseline not found.")
    baseline=df.loc[mask].iloc[0]
    dump(OUT/"settings.json",{"input_sha256":sha256(ROOT/"data.csv"),"bounds":bounds,
                              "integer_charge":True,"seeds":[42,43,44,45,46],"maxiter":200,
                              "popsize":20,"initial_population_size":60,"tol":1e-7,"atol":1e-8,
                              "mutation":[.5,1.],"recombination":.7,"polish":False,
                              "lhs_per_charge":1024,"lhs_total":19456,
                              "baseline_observed":baseline.to_dict(),"reference_threshold_fractions":[.5,.75,1.],
                              "constraint_interpretation":"reference scenarios, not application requirements",
                              "calibrated_bound":"joint one-sided 95% lower screen for water, Li and score; no guarantee at an adaptively chosen point"})
    runs=[]
    with threadpool_limits(limits=1):
        for label,evaluator in evaluators.items():
            for seed in range(42,47):
                runs.append(optimize(evaluator,bounds,seed,label+"_unconstrained"))
            for wf in [.5,.75,1.]:
                for lf in [.5,.75,1.]:
                    thresholds=(float(wf*baseline.FWater),float(lf*baseline.FLi))
                    seeds=[42,43,44] if wf==.75 and lf==.75 else [42]
                    for seed in seeds:
                        r=optimize(evaluator,bounds,seed,f"{label}_water{wf}_li{lf}",thresholds)
                        r.update(water_fraction=wf,li_fraction=lf)
                        runs.append(r)
        for pseudo in [.5,2.]:
            r=optimize(DesignEvaluator(full,pseudocount=pseudo),bounds,42,f"full_mean_pseudocount_{pseudo}")
            r["pseudocount"]=pseudo
            runs.append(r)
        pd.DataFrame(runs).to_csv(OUT/"all_optimizer_runs.csv",index=False)
        candidate_x=np.unique(np.vstack([lhs_candidates(bounds),x,OLD,
                                         np.array([[r[c] for c in FEATURES] for r in runs])]),axis=0)
        for label,evaluator in evaluators.items():
            score,water,li,p=evaluator.evaluate(candidate_x)
            table=pd.DataFrame(candidate_x,columns=FEATURES)
            for i,t in enumerate(TARGETS): table[t+"_predicted_mean"]=p["count"][:,i]
            table["predicted_log_score"]=p["log"][:,4]
            table["screening_objective"]=score
            table["water_constraint_value"]=water
            table["li_constraint_value"]=li
            table["score_scale_log"]=p["scale"][:,4]
            front=NonDominatedSorting().do(-np.column_stack([water,li,score]),only_non_dominated_front=True)
            table["pareto"]=False
            table.loc[front,"pareto"]=True
            table.to_csv(OUT/(label+"_candidate_scan.csv"),index=False)
            table[table.pareto].to_csv(OUT/(label+"_pareto.csv"),index=False)
        references=[]
        for model,evaluator in evaluators.items():
            for name,point in [("old_reported",OLD),("observed_baseline",baseline[FEATURES].to_numpy(float))]:
                references.append({"model":model,"candidate":name,**point_record(point,evaluator)})
            for run in runs:
                if run["seed"]==42:
                    point=np.array([run[c] for c in FEATURES])
                    references.append({"model":model,"candidate":run["label"],**point_record(point,evaluator)})
        pd.DataFrame(references).to_csv(OUT/"cross_model_candidate_comparison.csv",index=False)
    dump(OUT/"complete.json",{"runs":len(runs),"candidate_scan_size":len(candidate_x),
                               "data_sha256":sha256(ROOT/"data.csv")})


if __name__=="__main__":
    main()
