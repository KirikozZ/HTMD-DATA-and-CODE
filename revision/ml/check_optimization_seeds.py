"""Additional optimizer-seed sensitivity check; does not refit or replace models."""
import run_optimization as opt
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from ml_core import ROOT, FEATURES, TARGETS, dataset, dump, sha256


def main():
    out = ROOT / 'optimization' / 'seed_sensitivity_1_100_1000_10000_100000'
    (out / 'runs').mkdir(parents=True, exist_ok=True)
    opt.OUT = out
    model_path = ROOT / 'models' / 'full_data' / 'models.joblib'
    original_path = ROOT / 'optimization' / 'all_optimizer_runs.csv'
    protected = [model_path, original_path, ROOT / 'interpretation' / 'local_candidates.csv']
    before = {str(p): sha256(p) for p in protected}
    x = dataset()[FEATURES].to_numpy(float)
    bounds = np.column_stack([x.min(axis=0), x.max(axis=0)])
    evaluator = opt.DesignEvaluator(joblib.load(model_path)['selected'])
    seeds = [1, 100, 1000, 10000, 100000]
    with threadpool_limits(limits=1):
        rows = [opt.optimize(evaluator, bounds, s, 'full_mean_unconstrained') for s in seeds]
    new = pd.DataFrame(rows)
    new.to_csv(out / 'additional_seed_results.csv', index=False)
    old = pd.read_csv(original_path)
    old = old[old.label == 'full_mean_unconstrained'].copy()
    combined = pd.concat([old.assign(seed_group='original'), new.assign(seed_group='additional')], ignore_index=True)
    combined.to_csv(out / 'all_ten_seed_results.csv', index=False)
    columns = FEATURES + [t + '_predicted_mean' for t in TARGETS] + ['predicted_log_score']
    stats = []
    for name, frame in [('original', old), ('additional', new), ('all_ten', combined)]:
        for col in columns:
            values = frame[col]
            spread = float(values.max() - values.min())
            stats.append(dict(group=name, quantity=col, minimum=values.min(), maximum=values.max(),
                              spread=spread, relative_spread_percent=100*spread/abs(values.mean())))
    pd.DataFrame(stats).to_csv(out / 'seed_spread_summary.csv', index=False)
    after = {str(p): sha256(p) for p in protected}
    assert before == after, 'An existing result or model changed.'
    assert new.solver_success.all() and new.feasible.all(), 'Inspect unconverged optimizer runs.'
    dump(out / 'audit.json', {'seeds': seeds, 'bounds': bounds, 'protected_sha256': after,
                             'existing_files_unchanged': before == after,
                             'optimizer_script_sha256': sha256(ROOT / 'run_optimization.py'),
                             'data_sha256': sha256(ROOT / 'data.csv'),
                             'protocol': 'Same frozen full-data selected model and original optimizer settings, including fixed OLD x0; only optimizer RNG seed changes. No replacement of manuscript candidate.'})
    print(new[['seed'] + columns].to_string(index=False))
    print(pd.DataFrame(stats).query("group == 'additional'").to_string(index=False))
    print('Saved:', out)


if __name__ == '__main__':
    main()
