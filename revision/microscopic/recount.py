from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import os,subprocess,json,time,hashlib
ROOT=Path(__file__).resolve().parent
SOURCE=Path(os.environ['HTMD_OPTIMIZED_RAW']) if os.environ.get('HTMD_OPTIMIZED_RAW') else ROOT.parent/'simulation'/'optimized'
def run(p):
 out=ROOT/'raw'/p.name
 t=time.time()
 original=p/'equilibrium.lammpstrJ';st=original.stat()
 cases=json.loads((ROOT.parent/'optimized_transport/cases.json').read_text())
 meta=next(c for c in cases if c['case']==p.name)
 cmd=[str(ROOT/'micro_recount.exe'),str(original),str(out/'water_bonds.txt'),str(out)]
 with (out/'progress.log').open('w') as log:
  result=subprocess.run(cmd,stdout=log,stderr=log)
 if result.returncode:raise RuntimeError(p.name+' '+(out/'progress.log').read_text())
 assert st.st_size==original.stat().st_size and st.st_mtime_ns==original.stat().st_mtime_ns
 (out/'source_audit.json').write_text(json.dumps({'source':p.name+'/equilibrium.lammpstrJ','size':st.st_size,'mtime_ns':st.st_mtime_ns,'parser_sha256':hashlib.sha256((ROOT/'micro_recount.cpp').read_bytes()).hexdigest(),'parameters':meta},indent=2))
 print(p.name,'COMPLETE',round(time.time()-t,1),'s',flush=True)
if __name__=='__main__':
 with ThreadPoolExecutor(max_workers=3) as pool:
  for f in as_completed([pool.submit(run,p) for p in sorted(SOURCE.iterdir()) if p.is_dir()]):f.result()
 print('ALL SEVEN MICROSCOPIC RECOUNTS COMPLETE',flush=True)
