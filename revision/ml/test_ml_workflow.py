"""Numerical and contract tests; synthetic fixtures are never analysis data."""
import os
for var in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[var]="1"
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

import ml_core as core
import run_interpretation as interpretation
import run_optimization as optimization


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df=core.dataset()
        cls.x=cls.df[core.FEATURES].to_numpy(float)
        cls.y=cls.df.FLi.to_numpy(float)

    def test_positive_lognormal_mean(self):
        for mu in [-1.,0.,1.,5.]:
            for variance in [.1,1.]:
                sd=np.sqrt(variance)
                expected=quad(lambda z:np.expm1(z)*norm.pdf(z,mu,sd),0,max(mu+12*sd,12*sd))[0]
                actual=core.positive_lognormal_mean(np.array([mu]),np.array([variance]))[0]
                self.assertAlmostEqual(actual,expected,places=8)
                self.assertGreaterEqual(actual+1e-12,max(np.expm1(mu),0))

    def test_gp_matches_original_numpy_implementation(self):
        from reference import train_gpr_selectivity as legacy
        context=core.FitContext(self.x[:120],self.y[:120])
        spec=core.candidate_specs()[316]
        model=core.fit_model(context,spec)
        hp=legacy.HyperParams(spec["lengthscales"],spec["signal"],spec["noise"])
        old=legacy.fit_gpr(self.x[:120],np.log1p(self.y[:120]),"FLi",hp)
        expected=np.maximum(np.expm1(legacy.predict_log(old,self.x[120:])),0)
        actual=model.predict(self.x[120:])
        np.testing.assert_allclose(actual["count"],expected,rtol=1e-9,atol=1e-9)
        np.testing.assert_allclose(actual["latent_scale"],legacy.prediction_std_log(old,self.x[120:]),rtol=1e-8,atol=1e-8)

    def test_conformal_order_statistic(self):
        scores=np.arange(1,5,dtype=float)
        self.assertEqual(core.conformal_quantile(scores,.8),4)
        self.assertTrue(np.isinf(core.conformal_quantile(scores,.95)))
        np.testing.assert_array_equal(core.conformal_quantile(np.column_stack([scores,2*scores]),.8),[4,8])

    def test_all_model_families_return_valid_count_intervals(self):
        context=core.FitContext(self.x[:120],self.y[:120])
        specs=core.candidate_specs()
        for idx in [0,147,294,301,316]:
            model=core.fit_model(context,specs[idx])
            p=model.predict(self.x[120:],[.8,.95])
            self.assertTrue(np.isfinite(p["count"]).all())
            self.assertTrue((p["count"]>=0).all())
            self.assertTrue((p["lower_0.95"]<=p["upper_0.95"]).all())
            self.assertTrue((p["lower_0.95"]<=p["lower_0.8"]).all())
            self.assertTrue((p["upper_0.95"]>=p["upper_0.8"]).all())

    def test_exact_shap_and_interactions_on_additive_function(self):
        coefficients=np.arange(1,16).reshape(3,5)/10
        x=np.array([[.5,-3,2.],[1.,6,4.]])
        background=np.array([[.8,0,3.],[1.2,9,1.]])
        def predict(_bundle,z): return {"log":z@coefficients+2}
        with patch.object(interpretation,"predict_bundle",predict):
            phi,inter,base,pred=interpretation.exact_shap({},x,background)
        expected=np.einsum("nj,jt->ntj",x-background.mean(axis=0),coefficients)
        np.testing.assert_allclose(phi,expected,atol=1e-12)
        for a in range(3):
            for b in range(3):
                if a!=b: np.testing.assert_allclose(inter[:,:,a,b],0,atol=1e-12)
        np.testing.assert_allclose(base+phi.sum(axis=2),pred,atol=1e-12)

    def test_integer_optimizer_scalar_vector_constraints(self):
        class Fixture:
            def evaluate(self,x):
                x=np.asarray(x).reshape(-1,3)
                score=-((x-np.array([.7,-2.,3.]))**2).sum(axis=1)
                count=np.column_stack([100+x[:,0],20+x[:,2],np.ones(len(x)),np.ones(len(x))])
                log=np.column_stack([np.log1p(count),score])
                return score,count[:,0],count[:,1],{"count":count,"log":log,"scale":np.ones_like(log)}
        with tempfile.TemporaryDirectory(dir=core.ROOT) as directory:
            out=Path(directory); (out/"runs").mkdir()
            with patch.object(optimization,"OUT",out):
                r=optimization.optimize(Fixture(),np.array([[.5,1.2],[-9,9],[1.346421,4.346421]]),42,
                                        "synthetic_contract_test",thresholds=(100.,22.))
        self.assertTrue(r["feasible"])
        self.assertEqual(r["Charge"],-2.)
        self.assertAlmostEqual(r["Hydrophilicity"],.7,places=3)
        self.assertAlmostEqual(r["Sigma"],3.,places=3)


if __name__=="__main__":
    unittest.main(verbosity=2)
