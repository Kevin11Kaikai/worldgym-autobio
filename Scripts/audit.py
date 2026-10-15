"""CPU NumPy checks of saved rollout likelihoods, real optimizer probes, and official MuJoCo labels."""
import argparse,json,math,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/'runtime'))
from control import read,write,sha,now

def audit(run):
    reports=[]
    for p in sorted(run.rglob('trajectory.json')):
        row=read(p);trace=p.parent/'policy_trace.npz'
        assert sha(trace)==row['trace_sha256']
        z=np.load(trace);lp=-.5*((z['sample'].astype(float)-z['mean'])/row['std'])**2-math.log(row['std'])-.5*math.log(2*math.pi)
        err=float(np.max(np.abs(lp-z['old_logprob'])));assert err<3e-4,err
        assert len(z['controls'])==1130 and len(z['sample'])==23
        if row['native']:
            assert 'success' not in row and not (p.parent/'official_audit.json').exists()
        else:
            a=read(p.parent/'official_audit.json');pair={a['finger_geom'],a['cap_geom']}
            hits=[h>.925 and any(set(c)==pair for c in contacts) for h,contacts in zip(a['heights'],a['contacts'])]
            assert hits==a['official_hits']
            assert bool(any(hits))==row['success']
            first=(hits.index(True)+1)*.2 if any(hits) else None
            assert first is None and row['first_success'] is None or first is not None and abs(first-row['first_success'])<1e-8
        reports.append(dict(kind='trajectory',path=p.relative_to(run).as_posix(),max_logprob_error=err))
    for p in sorted(run.rglob('likelihood_epoch*.npz')):
        z=np.load(p);std=float(z['std']);mu=z['new_mean'].astype(float);oldmu=z['old_mean'].astype(float);acts=z['actions'].astype(float)
        lp=lambda m:-.5*((acts-m)/std)**2-math.log(std)-.5*math.log(2*math.pi)
        ratio=np.exp(np.clip(lp(mu)-z['old_lp'],-20,20));adv=float(z['advantage']);mask=z['valid'][:,None]
        pg=(-np.minimum(ratio*adv,np.clip(ratio,1-float(z['clip_low']),1+float(z['clip_high']))*adv)*mask).sum()/(mask.sum()*7)
        kl=(.5*((mu-z['ref_mean'])/std)**2*mask).sum()/(mask.sum()*7)
        assert abs(pg-float(z['pg']))<3e-4 and abs(kl-float(z['kl']))<3e-4
        reports.append(dict(kind='likelihood_policy_gradient_kl',path=p.relative_to(run).as_posix(),pg_error=abs(pg-float(z['pg'])),kl_error=abs(kl-float(z['kl']))))
    for p in sorted(run.rglob('adam*.json')):
        meta=read(p);z=np.load(p.with_suffix('.npz'));errs=[]
        for i,pr in enumerate(meta['parameters']):
            b,g,m,v,after=[z[f'{i}_{key}'].astype(float) for key in ['before','gradient','m','v','after']]
            b1,b2=meta['betas'];step=pr['step'];m=b1*m+(1-b1)*g;v=b2*v+(1-b2)*g*g
            expected=b-meta['lr']*(m/(1-b1**step))/(np.sqrt(v/(1-b2**step))+meta['eps'])
            err=float(np.max(np.abs(expected-after)));assert err<2e-6,err;errs.append(err)
        reports.append(dict(kind='actual_Adam_coordinate_probes',path=p.relative_to(run).as_posix(),max_error=max(errs)))
    if not reports:raise ValueError('No saved trajectories or GRPO probes found; nothing was audited.')
    result=dict(status='PASS',utc=now(),checks=reports,scope='Independent NumPy formula recomputation from saved tensors; not a second training run or a performance replication.')
    write(run/'AUDIT.json',result);return dict(status='PASS',checks=len(reports))
if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run',required=True,type=Path)
    print(json.dumps(audit(ap.parse_args().run),indent=2))
