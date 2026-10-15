"""Matched 3000-step SmolVLA continuation; precision and learning rate are explicit."""
from control import H,R,read,write,sha,append,state,budget,now
from policy_runtime import Actor,autocast,fingerprint
from data_runtime import Data
from train_sft import validate
import torch,numpy as np,math,time,gc,json
def build(precision,checkpoint):
 a=Actor(checkpoint)
 if precision=="fp32":
  for q in a.parameters():q.data=q.data.float()
  ck=torch.load(checkpoint,map_location="cpu",weights_only=False)
  a.model.load_state_dict(ck["adapter"],strict=False)
 a.rl_mode=False
 return a

def train(precision):
 p=read(H/"SFT_REPAIR_PROTOCOL.json");out=H/"sft_repair"/precision;out.mkdir(parents=True,exist_ok=True)
 if (out/"COMPLETE.json").exists():print(read(out/"COMPLETE.json"));return
 budget(7200);last=out/"last.pt";d=Data(40);a=build(precision,last if last.exists() else R/p["base"])
 torch.manual_seed(p["seed"]);np.random.seed(p["seed"])
 opt=torch.optim.AdamW(a.parameters(),lr=p["lr"],weight_decay=0);begin=0;elapsed=0
 if last.exists():
  ck=torch.load(last,map_location="cpu",weights_only=False);assert ck["protocol_sha256"]==sha(H/"SFT_REPAIR_PROTOCOL.json")
  opt.load_state_dict(ck["optimizer"]);begin=ck["step"];elapsed=ck["active_seconds"]
  torch.set_rng_state(ck["torch_rng"]);torch.cuda.set_rng_state_all(ck["cuda_rng"])
 first=fingerprint(a.adapter());t=time.time()
 for step in range(begin+1,p["steps"]+1):
  budget(7200);s=time.time();a.policy.train();opt.zero_grad(set_to_none=True)
  b=d.batch(step,p["seed"],p["batch"])
  with autocast():loss=a.sft_loss(b)
  assert torch.isfinite(loss);loss.backward();gn=torch.nn.utils.clip_grad_norm_(a.parameters(),10)
  assert torch.isfinite(gn)
  probes=[]
  if precision=="fp32" and step in [1,p["steps"]]:
   for n,q in a.model.named_parameters():
    if not q.requires_grad or q.grad is None or len(probes)>=12:continue
    idx=torch.linspace(0,q.numel()-1,32).long().cuda();st=opt.state.get(q,{})
    probes.append(dict(name=n,indices=idx.cpu().numpy(),step=int(st.get("step",0))+1,
      before=q.detach().flatten()[idx].cpu().numpy().copy(),gradient=q.grad.detach().flatten()[idx].cpu().numpy().copy(),
      m=st["exp_avg"].flatten()[idx].cpu().numpy().copy() if st else np.zeros(32,np.float32),
      v=st["exp_avg_sq"].flatten()[idx].cpu().numpy().copy() if st else np.zeros(32,np.float32)))
  opt.step();torch.cuda.synchronize();elapsed+=time.time()-s
  if probes:
   named=dict(a.model.named_parameters());z={}
   for i,r in enumerate(probes):
    r["after"]=named[r["name"]].detach().flatten()[torch.from_numpy(r["indices"]).cuda()].cpu().numpy()
    for k in ["indices","before","gradient","m","v","after"]:z[f"{i}_{k}"]=r[k]
   np.savez(out/f"adam_{step:04d}.npz",**z)
   write(out/f"adam_{step:04d}.json",dict(parameters=[dict(name=x["name"],step=x["step"]) for x in probes],lr=p["lr"],betas=[.9,.999],eps=1e-8))
  if step%100==0:
   row=dict(step=step,precision=precision,loss=float(loss),gradient_norm=float(gn),active_seconds=elapsed)
   append(out/"training.jsonl",row);state("SFT_precision_repair",**row);print(json.dumps(row),flush=True)
  if step%500==0 or step==p["steps"]:
   val=validate(a,d);a.save(last,step=step,optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),
    cuda_rng=torch.cuda.get_rng_state_all(),active_seconds=elapsed,protocol_sha256=sha(H/"SFT_REPAIR_PROTOCOL.json"),validation=val)
 endpoint=out/"endpoint.pt";sh=a.save(endpoint,step=p["steps"],validation=val,protocol_sha256=sha(H/"SFT_REPAIR_PROTOCOL.json"))
 write(out/"COMPLETE.json",dict(status="COMPLETE",precision=precision,steps=p["steps"],active_seconds=elapsed,wall_seconds=time.time()-t,
  validation=val,sha256=sh,initial_weights=first,endpoint_weights=fingerprint(a.adapter()),utc=now()))
