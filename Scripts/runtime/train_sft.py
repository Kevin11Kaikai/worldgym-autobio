"""Matched supervised initial VLA; whole pretrained action expert, fit40 only."""
import time,random,json,gc,math
from control import H,R,read,write,sha,append,state,budget,now,log
from data_runtime import Data
from policy_runtime import Actor,autocast,fingerprint
import numpy as np,torch
def validate(actor,data):
    cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state_all()
    torch.manual_seed(70204);actor.policy.eval();vals=[]
    with torch.inference_mode(),autocast():
        for i in range(16):vals.append(float(actor.sft_loss(data.batch(i,70205,4,True))))
    torch.set_rng_state(cpu);torch.cuda.set_rng_state_all(gpu)
    return float(np.mean(vals))
def train():
    p=read(H/"SFT_PROTOCOL.json");budget(1800);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    seed=p["seed"];torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    d=Data(40);folder=H/"sft";folder.mkdir(exist_ok=True);last=folder/"last.pt";step0=0;elapsed=0.
    if (folder/"COMPLETE.json").exists():return read(folder/"COMPLETE.json")
    a=Actor(last if last.exists() else None);opt=torch.optim.AdamW(a.parameters(),lr=p["lr"],weight_decay=1e-10)
    if last.exists():
        c=torch.load(last,map_location="cpu",weights_only=False);assert c["protocol_sha256"]==sha(H/"SFT_PROTOCOL.json")
        opt.load_state_dict(c["optimizer"]);step0=c["step"];elapsed=c["active_seconds"]
        torch.set_rng_state(c["torch_rng"]);torch.cuda.set_rng_state_all(c["cuda_rng"])
    for step in range(step0+1,p["steps"]+1):
        budget(1800);s=time.time();a.policy.train();opt.zero_grad(set_to_none=True)
        lr=p["lr"]*min(1.,step/200)*(.1+.9*.5*(1+math.cos(math.pi*step/p["steps"])))
        for g in opt.param_groups:g["lr"]=lr
        b=d.batch(step,seed,p["batch"])
        with autocast():loss=a.sft_loss(b)
        assert torch.isfinite(loss);loss.backward();grad=torch.nn.utils.clip_grad_norm_(a.parameters(),10)
        assert torch.isfinite(grad);opt.step();torch.cuda.synchronize();elapsed+=time.time()-s
        if step%50==0:
            row=dict(step=step,loss=float(loss),grad=float(grad),lr=lr,active_seconds=elapsed,step_seconds=time.time()-s)
            append(folder/"training.jsonl",row);state("SFT",**row,total=p["steps"])
        if step in p["checkpoint_steps"] or step%1000==0:
            val=validate(a,d)
            a.save(last,step=step,optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                active_seconds=elapsed,protocol_sha256=sha(H/"SFT_PROTOCOL.json"),validation=val)
            if step in p["checkpoint_steps"]:
                cp=folder/f"step_{step:06d}.pt";cp_sha=a.save(cp,step=step,validation=val,protocol_sha256=sha(H/"SFT_PROTOCOL.json"))
                append(folder/"checkpoints.jsonl",dict(step=step,validation=val,sha256=cp_sha,path=str(cp.relative_to(H))))
            print(json.dumps(dict(SFT_step=step,validation=val,active_seconds=elapsed)),flush=True)
    r=dict(status="COMPLETE",steps=p["steps"],active_seconds=elapsed,utc=now(),last_sha256=sha(last),
        total_parameters=sum(p.numel() for p in a.model.parameters()),trainable_parameters=sum(p.numel() for p in a.parameters()),
        peak_cuda_bytes=torch.cuda.max_memory_allocated())
    write(folder/"COMPLETE.json",r);log("Full SmolVLA action expert SFT complete; development selection is a separate step.")
    return r
if __name__=="__main__":print(json.dumps(train()),flush=True)
