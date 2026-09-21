"""Recount the seven optimized-condition trajectories using the shared plane rules."""
from pathlib import Path
import argparse,hashlib,json,subprocess
ROOT=Path(__file__).resolve().parent
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--raw-root',type=Path,required=True);a=ap.parse_args()
    cases=json.loads((ROOT/'cases.json').read_text());sources=[a.raw_root/c['case']/'equilibrium.lammpstrJ' for c in cases]
    if not all(p.is_file() for p in sources):raise SystemExit('Provide the directory containing all seven complete source trajectories.')
    out=ROOT/'recounted';out.mkdir(exist_ok=True)
    for c,p in zip(cases,sources):
        before=p.stat();dest=out/(c['case']+'.csv')
        result=subprocess.run([str(ROOT/'recount_transport.exe'),str(p),str(dest)],capture_output=True,text=True,check=True)
        h=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
        assert (before.st_size,before.st_mtime_ns)==(p.stat().st_size,p.stat().st_mtime_ns)
        audit={'source_relative':c['case']+'/equilibrium.lammpstrJ','source_sha256':h.hexdigest(),'bytes':before.st_size,'counter_result':result.stdout.strip()}
        (out/(c['case']+'.json')).write_text(json.dumps(audit,indent=2),encoding='utf-8')
        print(c['case'],result.stdout.strip(),flush=True)
if __name__=='__main__':main()
