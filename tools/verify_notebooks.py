"""Verify bundled evidence, then execute exactly three notebooks in fresh CPU kernels."""
from pathlib import Path
import json,os,sys,time,hashlib
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R))
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYDEVD_DISABLE_FILE_VALIDATION']='1'
for key,sub in [('IPYTHONDIR','ipython'),('JUPYTER_CONFIG_DIR','jupyter_config'),('JUPYTER_DATA_DIR','jupyter_data'),('JUPYTER_RUNTIME_DIR','jupyter_runtime'),('MPLCONFIGDIR','matplotlib')]:
    p=R/'outputs/_runtime'/sub;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
import nbformat
from nbclient import NotebookClient
from briefing_utils import verify_assets,outcomes,data

def main():
    count=verify_assets()
    gain,native,repair=[outcomes(k) for k in ['gain','native','repair']]
    assert gain.sum().to_dict()==data('gain_summary.json')['final_successes']
    assert native.sum().to_dict()==data('native_summary.json')['successes_per_100']
    assert repair.sum().to_dict()=={r['policy']:r['successes'] for r in data('repair_results.json')['policies']}
    assert not(set(gain.index)&set(native.index) or set(gain.index)&set(repair.index) or set(native.index)&set(repair.index))
    for r in data('repair_results.json')['paired'].values():
        d=repair[r['a']]-repair[r['b']]
        assert (round(d.mean()*100),int((d==1).sum()),int((d==-1).sum()))==(round(r['delta_pp']),r['wins'],r['harms'])
    for k,r in data('native_summary.json')['paired_comparisons'].items():
        a,b=k.split('_minus_');d=native[a]-native[b]
        assert (round(d.mean()*100),int((d==1).sum()),int((d==-1).sum()))==(round(r['paired_gain_pp']),r['improvements'],r['new_failures'])
    files=sorted(R.glob('*.ipynb'));assert len(files)==3
    result=[]
    for p in files:
        nb=nbformat.read(p,as_version=4);t=time.perf_counter()
        NotebookClient(nb,timeout=180,kernel_name='python3',allow_errors=False,resources={'metadata':{'path':str(R)}}).execute()
        cells=[x for x in nb.cells if x.cell_type=='code']
        assert all(x.execution_count for x in cells)
        assert not any(o.output_type=='error' for x in cells for o in x.outputs)
        nbformat.validate(nb)
        # Keep published notebook bytes stable; save this execution separately.
        nbformat.write(nb,R/'outputs'/p.name)
        result.append(dict(notebook=p.name,code_cells=len(cells),seconds=time.perf_counter()-t,status='PASS'))
        print(p.name+': PASS',flush=True)
    report=dict(status='PASS',assets_checked=count,fresh_kernel_per_notebook=True,notebooks=result,
                scope='Archived outcome arithmetic and fresh notebook execution; no new policy inference.')
    (R/'outputs/NOTEBOOK_CHECK.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
if __name__=='__main__':main()
