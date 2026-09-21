"""Independently verify published interval, cumulative and block tables."""
import argparse, csv, hashlib, json, math, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPECIES = ['water', 'Li', 'Cl', 'Mg']
TARGETS = ['FWater', 'FLi', 'FCl', 'FMg']
METHODS = ['legacy', 'plane', 'transit']

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--follow', action='store_true', help='Validate completed cases while export is running')
    args = parser.parse_args()
    def cases():
        if not args.follow:
            manifest = json.loads((ROOT/'recount_manifest.json').read_text(encoding='utf-8'))
            assert manifest['full_196_exported'], 'Incomplete source processing'
            yield from manifest['cases']
            return
        for i in range(1, 197):
            audit = ROOT/'case_audits'/f'S{i:03d}.json'
            while not audit.exists():
                time.sleep(2)
            case = json.loads(audit.read_text(encoding='utf-8'))
            assert case['processing_status'] == 'complete', case
            yield case
    with (ROOT/'training/paper_training_data.csv').open(encoding='utf-8-sig') as f:
        training = list(csv.DictReader(f))
    assert digest(ROOT/'training/paper_training_data.csv') == '2cf6f24d0ae341383ecf7c0a68108d2672417f344bec6a9d7aad3efb4ac0e659'
    verified = []; differences = []
    for case, original in zip(cases(), training, strict=True):
        key = case['case_id']; totals = {(s,m,d): 0 for s in SPECIES for m in METHODS for d in ['forward','reverse']}
        prev = None; n = 0; first = None; duration = 0
        for output in case['outputs']:
            assert digest(ROOT/output['path']) == output['sha256'], output['path']
        with (ROOT/'events'/f'{key}.csv').open(newline='') as f:
            for row in csv.DictReader(f):
                frame = int(row['frame_index']); step = int(row['step']); t = float(row['time_ns'])
                assert frame == n and step == (n+1)*10000
                assert math.isclose(t, step*1e-6, abs_tol=1e-10)
                assert int(row['interval_observed']) == int(n > 0)
                interval = float(row['interval_duration_ns'])
                assert math.isclose(interval, .01 if n else 0, abs_tol=1e-10)
                assert math.isclose(float(row['interval_end_ns'])-float(row['interval_start_ns']), interval, abs_tol=1e-10)
                duration += interval
                if first is None: first = t
                for s in SPECIES:
                    assert sum(int(row[s+'_'+r]) for r in ['feed','membrane','permeate']) == case['initial_species_counts'][s]
                    for m in METHODS:
                        fwd, rev = (int(row[f'{s}_{m}_{d}']) for d in ['forward','reverse'])
                        assert fwd >= 0 and rev >= 0
                        assert n or (fwd == rev == 0)
                        totals[s,m,'forward'] += fwd; totals[s,m,'reverse'] += rev
                        assert int(row[f'{s}_{m}_net']) == fwd-rev
                        assert int(row[f'{s}_{m}_cumulative']) == totals[s,m,'forward']-totals[s,m,'reverse']
                    if prev is not None:
                        assert int(row[s+'_permeate'])-int(prev[s+'_permeate']) == int(row[s+'_plane_net'])-int(row[s+'_wrap_net'])
                n += 1; prev = row
        assert n == case['frames'] == 12000 and first == .01 and t == 120
        assert math.isclose(duration, 119.99, abs_tol=1e-6)
        with (ROOT/'blocks'/f'{key}.csv').open(encoding='utf-8-sig') as f:
            blocks = list(csv.DictReader(f))
        assert len(blocks) == 12
        assert math.isclose(sum(float(b['duration_ns']) for b in blocks), duration, abs_tol=1e-6)
        for s in SPECIES:
            for m in METHODS:
                for d in ['forward','reverse']:
                    assert sum(int(b[f'{s}_{m}_{d}']) for b in blocks) == totals[s,m,d]
                for b in blocks:
                    assert int(b[f'{s}_{m}_net']) == int(b[f'{s}_{m}_forward'])-int(b[f'{s}_{m}_reverse'])
                assert totals[s,m,'forward']-totals[s,m,'reverse'] == case['recounted_counts'][s][m]
        paper = [int(original[x]) for x in TARGETS]
        assert paper == [case['training_counts'][s] for s in SPECIES]
        if not case['matches_training']: differences.append(key)
        verified.append({'case_id':key,'frames':n,'intervals':n-1,'species_checks':4,'mass_balance_max_error':0})
        if len(verified)%28 == 0: print(f'Validated {len(verified)}/196', flush=True)
    assert len(verified) == 196
    result = {'passed':True,'cases':196,'frames':sum(v['frames'] for v in verified),
              'observed_intervals':sum(v['intervals'] for v in verified),
              'paper_training_sha256':digest(ROOT/'training/paper_training_data.csv'),
              'training_unchanged':True,'label_difference_case_ids':differences,
              'checks':['output SHA256','all cumulative counts','forward-minus-reverse','all species populations',
                        'every interval periodic mass balance','all 10 ns block sums','saved window','paper labels'],
              'per_case':verified}
    (ROOT/'validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='per_case'}),flush=True)

if __name__ == '__main__': main()
