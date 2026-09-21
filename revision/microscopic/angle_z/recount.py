from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import os,subprocess,time,json,hashlib
R=Path(__file__).resolve().parent
SOURCE=Path(os.environ['HTMD_OPTIMIZED_RAW']) if os.environ.get('HTMD_OPTIMIZED_RAW') else R.parents[1]/'simulation'/'optimized'
def run(p):
 dest=R/p.name;dest.mkdir(exist_ok=True);t=time.time();traj=p/'equilibrium.lammpstrJ';st=traj.stat()
 with (dest/'progress.log').open('w') as log:
  ret=subprocess.run([str(R/'angle_recount.exe'),str(traj),str(R.parent/'raw'/p.name/'water_bonds.txt'),str(dest)],stdout=log,stderr=log)
 if ret.returncode:raise RuntimeError(p.name+' '+(dest/'progress.log').read_text())
 assert (st.st_size,st.st_mtime_ns)==(traj.stat().st_size,traj.stat().st_mtime_ns)
 (dest/'source_audit.json').write_text(json.dumps(dict(source=p.name+'/equilibrium.lammpstrJ',size=st.st_size,mtime_ns=st.st_mtime_ns,parser_sha256=hashlib.sha256((R/'angle_recount.cpp').read_bytes()).hexdigest()),indent=2))
 print(p.name,'COMPLETE',round(time.time()-t,1),'s',flush=True)
with ThreadPoolExecutor(max_workers=3) as pool:
 for f in as_completed([pool.submit(run,p) for p in sorted(SOURCE.iterdir()) if p.is_dir()]):f.result()
print('ALL SEVEN ANGLE PROFILES COMPLETE',flush=True)
