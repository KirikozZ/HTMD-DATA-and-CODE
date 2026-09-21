"""Run the main scientific workflow in an explicitly prepared working copy."""
from pathlib import Path
import argparse,os,shutil,subprocess,sys,json
ROOT=Path(__file__).resolve().parent
STAGES={
 'ml-check':[('ml/test_ml_workflow.py',[]),('ml/verify_results.py',[])],
 'ml-fit':[('ml/run_validation.py',['--stage','all']),('ml/run_diagnostics.py',[]),('ml/run_optimization.py',[]),('ml/run_interpretation.py',[]),('ml/check_optimization_seeds.py',[]),('ml/export_publication_tables.py',[])],
 'publication-tables':[('ml/export_publication_tables.py',[])],
 'transport-tables':[('optimized_transport/summarize.py',[])],
 'topology':[('microscopic/prepare_topology.py',[])],
 'microscopic-tables':[('microscopic/summarize.py',[]),('microscopic/export_profiles.py',[])],
 'screening-check':[('screening/validate_export.py',[])],
 'micro-recount':[('microscopic/prepare_topology.py',[]),('microscopic/recount.py',[]),('microscopic/angle_z/recount.py',[])]}
CPP=['screening/recount_screening.cpp','optimized_transport/recount_transport.cpp','microscopic/micro_recount.cpp','microscopic/angle_z/angle_recount.cpp']
def main():
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='action',required=True)
    p=sub.add_parser('prepare');p.add_argument('--dest',type=Path,required=True)
    p=sub.add_parser('stage');p.add_argument('stage',choices=STAGES);p.add_argument('--work',type=Path,required=True);p.add_argument('--python',default=sys.executable);p.add_argument('--optimized-raw',type=Path)
    p=sub.add_parser('compile');p.add_argument('--work',type=Path,required=True);p.add_argument('--compiler',default='g++')
    a=ap.parse_args()
    if a.action=='prepare':
        dest=a.dest.resolve()
        if dest.exists():raise SystemExit('Choose a new working directory.')
        required = [ROOT/'screening/recount_summary.csv', ROOT/'optimized_transport/transport_counts.csv']
        required += [p.with_suffix('.data') for p in (ROOT/'simulation').rglob('TbHz.in')]
        required += [p.with_suffix('.colvars') for p in (ROOT/'simulation/pmf').rglob('TbHz.in')]
        if not required or any(not p.is_file() for p in required):
            raise SystemExit('Extract the matching Release ZIP at the repository root first; simulation inputs or analysis data are missing.')
        for folder in ['simulation','screening','ml','optimized_transport','microscopic']:
            shutil.copytree(ROOT/folder,dest/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        (dest/'WORKING_COPY.json').write_text(json.dumps({'workflow':'main','source_version':'2026-09-21'}))
        print('Prepared',dest);return
    work=a.work.resolve()
    if not (work/'WORKING_COPY.json').exists():raise SystemExit('Run prepare first.')
    if a.action=='compile':
        for name in CPP:
            p=work/name;subprocess.run([a.compiler,'-std=c++17','-O3',str(p),'-o',str(p.with_suffix('.exe'))],check=True)
        return
    env=dict(os.environ,PYTHONUTF8='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    if a.optimized_raw:env['HTMD_OPTIMIZED_RAW']=str(a.optimized_raw.resolve())
    for name,args in STAGES[a.stage]:
        subprocess.run([a.python,'-X','utf8',str(work/name),*args],cwd=work,env=env,check=True)
if __name__=='__main__':main()
