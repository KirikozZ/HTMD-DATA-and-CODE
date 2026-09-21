"""Training-only model selection and prediction primitives for the revision."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import norm, poisson
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.linear_model import PoissonRegressor, Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROOT = Path(__file__).resolve().parent
FEATURES = ["Hydrophilicity", "Charge", "Sigma"]
TARGETS = ["FWater", "FLi", "FCl", "FMg"]
OUTPUTS = TARGETS + ["ln_selectivity"]
FAMILIES = ["gpr_mean", "gpr_hetero", "quadratic", "poisson", "gpr_median"]
LEVELS = [.5, .8, .9, .95]


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                              default=lambda v: v.tolist() if isinstance(v, np.ndarray) else float(v)), encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dataset():
    df = pd.read_csv(ROOT / "data.csv")
    if list(df.columns) != FEATURES + TARGETS:
        raise ValueError("Expected the three descriptor columns and four count columns in documented order.")
    values = df.to_numpy(float)
    if not np.isfinite(values).all() or (df[TARGETS].to_numpy() < 0).any():
        raise ValueError("Missing, infinite or negative count data.")
    if not np.equal(df[TARGETS], np.floor(df[TARGETS])).all().all():
        raise ValueError("This count-model protocol requires nonnegative integer observations.")
    if len(df) != 196 or df.duplicated(FEATURES).any():
        raise ValueError("Expected 196 distinct systems.")
    if np.prod([df[c].nunique() for c in FEATURES]) != len(df):
        raise ValueError("Incomplete factorial grid.")
    return df


def candidate_specs():
    anchors = [([.35]*3, 1., .05), ([.7]*3, 1., .05), ([1.2]*3, 1., .05),
               ([2.]*3, 1., .08), ([.7,1.5,.7],1.2,.05), ([1.5,.7,1.5],1.2,.05),
               ([1.,2.5,1.],1.,.1)]
    rng = np.random.default_rng(42)
    gp = [{"lengthscales": l, "signal": s, "noise": n} for l,s,n in anchors]
    for _ in range(140):
        gp.append({"lengthscales": np.exp(rng.uniform(np.log(.25), np.log(5), 3)).tolist(),
                   "signal": float(np.exp(rng.uniform(np.log(.5), np.log(3)))),
                   "noise": float(np.exp(rng.uniform(np.log(.015), np.log(.45))))})
    specs = []
    for family in ["gpr_mean", "gpr_hetero"]:
        specs.extend({"family": family, **p} for p in gp)
    specs.extend({"family": "quadratic", "degree": 2, "alpha": a}
                 for a in [0., .001, .01, .1, 1., 10., 100.])
    specs.extend({"family": "poisson", "degree": d, "alpha": a}
                 for d in [1,2,3] for a in [.001, .01, .1, 1., 10.])
    specs.extend({"family": "gpr_median", **p} for p in gp)
    return [{"id": i, **s} for i,s in enumerate(specs)]


def positive_lognormal_mean(mu, variance):
    """E[max(exp(Z)-1, 0)] for a Gaussian Z, including the mass at zero."""
    sd = np.sqrt(np.maximum(variance, 1e-16))
    return np.maximum(np.exp(np.minimum(mu + variance/2, 700)) * ndtr((mu + variance)/sd)
                      - ndtr(mu/sd), 0.)


class FitContext:
    def __init__(self, x, y):
        self.x = np.asarray(x, float)
        self.y = np.asarray(y, float)
        self.scaler = StandardScaler().fit(self.x)
        self.zx = self.scaler.transform(self.x)
        self.yl = np.log1p(y)
        self.ym = float(self.yl.mean())
        self.ys = max(float(self.yl.std()), 1e-10)
        self.zy = (self.yl-self.ym)/self.ys
        self.pilot = make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(),
                                   PoissonRegressor(alpha=1., max_iter=1500, tol=1e-7))
        self.pilot.fit(self.zx, self.y)
        m = np.maximum(self.pilot.predict(self.zx), 1e-8)
        self.noise_normalizer = float(np.mean(m/(1+m)**2))

    def noise_weights(self, x):
        m = np.maximum(self.pilot.predict(self.scaler.transform(x)), 1e-8)
        return np.clip(m/(1+m)**2 / max(self.noise_normalizer, 1e-12), .05, 20.)


@dataclass
class CountModel:
    spec: dict
    context: FitContext
    estimator: object
    residual_variance: float = 0.

    def predict(self, x, levels=()):
        x = np.asarray(x, float).reshape(-1,3)
        zx = self.context.scaler.transform(x)
        family = self.spec["family"]
        if family.startswith("gpr"):
            pred, latent_std = self.estimator.predict(zx, return_std=True)
            mu = pred*self.context.ys+self.context.ym
            latent_var = (latent_std*self.context.ys)**2
            weights = self.context.noise_weights(x) if family == "gpr_hetero" else np.ones(len(x))
            variance = latent_var+(self.spec["noise"]*self.context.ys)**2*weights
            mean = np.maximum(np.expm1(mu),0.) if family == "gpr_median" else positive_lognormal_mean(mu,variance)
        elif family == "quadratic":
            mu = self.estimator.predict(zx)
            variance = np.full(len(x), self.residual_variance)
            latent_var = np.zeros(len(x))
            mean = positive_lognormal_mean(mu,variance)
        else:
            mean = np.maximum(self.estimator.predict(zx), 1e-12)
            mu = np.log1p(mean)
            variance = mean/(1+mean)**2
            latent_var = np.zeros(len(x))
        sd = np.sqrt(np.maximum(variance,1e-12))
        result = {"count":mean, "log":np.log1p(mean), "mu":mu,
                  "scale":np.maximum(sd,1e-5), "latent_scale":np.sqrt(latent_var)}
        for level in levels:
            a = (1-level)/2
            if family == "poisson":
                low, high = np.log1p(poisson.ppf(a,mean)), np.log1p(poisson.ppf(1-a,mean))
            else:
                z = norm.ppf(1-a)
                low, high = np.maximum(mu-z*sd,0), np.maximum(mu+z*sd,0)
            result[f"lower_{level}"] = low
            result[f"upper_{level}"] = high
        return result


def fit_model(context, spec):
    family = spec["family"]
    if family.startswith("gpr"):
        weights = context.noise_weights(context.x) if family == "gpr_hetero" else np.ones(len(context.x))
        kernel = ConstantKernel(spec["signal"]**2, constant_value_bounds="fixed") * RBF(
            spec["lengthscales"], length_scale_bounds="fixed")
        est = GaussianProcessRegressor(kernel=kernel, alpha=spec["noise"]**2*weights+1e-8,
                                       optimizer=None, normalize_y=False)
        est.fit(context.zx, context.zy)
        return CountModel(spec,context,est)
    if family == "quadratic":
        est = make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(),
                            Ridge(alpha=spec["alpha"]))
        est.fit(context.zx,context.yl)
        z = est[:-1].transform(context.zx)
        singular = np.linalg.svd(z-z.mean(axis=0),compute_uv=False)**2
        df = float(np.sum(singular/(singular+max(spec["alpha"],1e-12))))+1
        variance = float(np.sum((context.yl-est.predict(context.zx))**2)/max(len(z)-df,1))
        return CountModel(spec,context,est,variance)
    est = make_pipeline(PolynomialFeatures(spec["degree"],include_bias=False), StandardScaler(),
                        PoissonRegressor(alpha=spec["alpha"],max_iter=2000,tol=1e-7))
    est.fit(context.zx,context.y)
    return CountModel(spec,context,est)


def metrics(y, yp):
    error = np.asarray(y)-np.asarray(yp)
    denominator = np.sum((np.asarray(y)-np.mean(y))**2)
    return {"r2": float(1-np.sum(error**2)/denominator) if denominator>0 else np.nan,
            "mae":float(np.mean(np.abs(error))), "rmse":float(np.sqrt(np.mean(error**2)))}


def truth_logs(raw):
    g = np.log1p(raw)
    return np.column_stack([g,g[:,1]-g[:,3]])


def predict_bundle(bundle, x, levels=()):
    details = [bundle[t].predict(x, levels) for t in TARGETS]
    log = np.column_stack([d["log"] for d in details])
    scales = np.column_stack([d["scale"] for d in details])
    result = {"log":np.column_stack([log,log[:,1]-log[:,3]]),
              "count":np.column_stack([d["count"] for d in details]),
              "scale":np.column_stack([scales,np.sqrt(scales[:,1]**2+scales[:,3]**2)])}
    for level in levels:
        lo=np.column_stack([d[f"lower_{level}"] for d in details])
        hi=np.column_stack([d[f"upper_{level}"] for d in details])
        # Bonferroni bounds avoid assuming independent Li/Mg errors.
        marginal_level = (1+level)/2
        li = bundle["FLi"].predict(x,[marginal_level])
        mg = bundle["FMg"].predict(x,[marginal_level])
        result[f"lower_{level}"]=np.column_stack([lo,li[f"lower_{marginal_level}"]-mg[f"upper_{marginal_level}"]])
        result[f"upper_{level}"]=np.column_stack([hi,li[f"upper_{marginal_level}"]-mg[f"lower_{marginal_level}"]])
    return result


def search_and_fit(x, raw, row_ids, directory, all_families=True):
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    specs=candidate_specs()
    folds=list(KFold(5,shuffle=True,random_state=42).split(x))
    pred=np.full((len(specs),len(x),4),np.nan)
    logs=[]
    fold_audit=[]
    for fi,(tr,va) in enumerate(folds):
        fold_audit.append({"train_rows":np.asarray(row_ids)[tr].tolist(),
                           "validation_rows":np.asarray(row_ids)[va].tolist(),
                           "x_mean":x[tr].mean(axis=0).tolist(),"x_std":x[tr].std(axis=0).tolist(),
                           "y_log_mean":np.log1p(raw[tr]).mean(axis=0).tolist(),
                           "y_log_std":np.log1p(raw[tr]).std(axis=0).tolist()})
        for ti,target in enumerate(TARGETS):
            context=FitContext(x[tr],raw[tr,ti])
            gp_models={}
            for ci,spec in enumerate(specs):
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always",ConvergenceWarning)
                        if spec["family"]=="gpr_median":
                            base=gp_models[ci-(len(specs)-147)]
                            model=replace(base,spec=spec)
                        else:
                            model=fit_model(context,spec)
                            if spec["family"]=="gpr_mean":
                                gp_models[ci]=model
                        pred[ci,va,ti]=model.predict(x[va])["log"]
                    for warning in caught:
                        logs.append({"fold":fi,"target":target,"candidate":ci,"warning":str(warning.message)})
                except (ValueError,np.linalg.LinAlgError,FloatingPointError) as exc:
                    logs.append({"fold":fi,"target":target,"candidate":ci,"failed":str(exc)})
    valid=np.isfinite(pred).all(axis=(1,2))
    if not valid.any():
        raise RuntimeError("No valid candidate remained in inner selection.")
    errors=pred-np.log1p(raw)[None,:,:]
    mse=np.mean(errors**2,axis=1)
    mse[~valid]=np.inf
    li=errors[:,:,1]
    mg=errors[:,:,3]
    joint=mse[:,1,None]+mse[None,:,3]-2*(np.nan_to_num(li)@np.nan_to_num(mg).T)/len(x)
    joint=np.maximum(joint,0)
    joint[~valid,:]=np.inf
    joint[:,~valid]=np.inf
    choices={}
    for family in ["selected",*FAMILIES] if all_families else ["selected"]:
        ids=np.array([s["id"] for s in specs if valid[s["id"]] and
                      (s["family"]!="gpr_median" if family=="selected" else s["family"]==family)],int)
        if not len(ids):
            continue
        best=ids[np.argmin(mse[ids],axis=0)]
        a,b=np.unravel_index(np.argmin(joint[np.ix_(ids,ids)]),(len(ids),len(ids)))
        best[1],best[3]=ids[a],ids[b]
        choices[family]=best.tolist()
    np.savez_compressed(directory/"inner_search.npz",oof_log_predictions=pred,
                        log_rmse=np.sqrt(mse),joint_score_rmse=np.sqrt(joint),row_ids=row_ids)
    dump(directory/"search.json",{"folds":fold_audit,"specs":specs,"choices":choices,"warnings":logs})
    rows=[]
    for spec in specs:
        for ti,target in enumerate(TARGETS):
            rows.append({"candidate":spec["id"],"family":spec["family"],"target":target,
                         "inner_log_rmse":float(np.sqrt(mse[spec["id"],ti])),"valid":bool(valid[spec["id"]])})
    pd.DataFrame(rows).to_csv(directory/"candidate_scores.csv",index=False)
    contexts=[FitContext(x,raw[:,i]) for i in range(4)]
    fitted={}
    memo={}
    for family,ids in choices.items():
        fitted[family]={}
        for i,target in enumerate(TARGETS):
            key=(i,ids[i])
            if key not in memo:
                memo[key]=fit_model(contexts[i],specs[ids[i]])
            fitted[family][target]=memo[key]
    joblib.dump(fitted,directory/"models.joblib",compress=3)
    return fitted


def conformal_quantile(scores, coverage):
    values=np.sort(np.asarray(scores,float),axis=0)
    rank=int(np.ceil((len(values)+1)*coverage))
    return np.full(values.shape[1:],np.inf) if rank>len(values) else values[rank-1]


def calibrate(bundle,x,raw,row_ids):
    p=predict_bundle(bundle,x)
    signed=(p["log"]-truth_logs(raw))/p["scale"]
    q={str(level):conformal_quantile(abs(signed),level).tolist() for level in LEVELS}
    joint=signed[:,[0,1,4]].max(axis=1)
    jq={str(level):max(float(conformal_quantile(joint,level)),0.) for level in LEVELS}
    return {"rows":np.asarray(row_ids).tolist(),"absolute_scores":abs(signed).tolist(),
            "signed_scores":signed.tolist(),"quantiles":q,"joint_lower_quantiles":jq}


def calibrated_intervals(p,calibration,level):
    q=np.array(calibration["quantiles"][str(level)])
    lower=p["log"]-q*p["scale"]
    upper=p["log"]+q*p["scale"]
    lower[:,:4]=np.maximum(lower[:,:4],0)
    upper[:,:4]=np.maximum(upper[:,:4],0)
    return lower,upper


def joint_lower(p,calibration,level=.95):
    lower=p["log"]-calibration["joint_lower_quantiles"][str(level)]*p["scale"]
    lower[:,:4]=np.maximum(lower[:,:4],0)
    return lower
