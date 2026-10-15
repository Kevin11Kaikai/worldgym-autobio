"""Native WorldGym -> real VLM reward -> differentiable Gaussian GRPO. Includes a matched MuJoCo arm."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'runtime'))
from cli import common,begin,configure,resolve_checkpoint,gpu_lock,finish
from control import write,sha
def main():
    ap=argparse.ArgumentParser(description=__doc__);common(ap)
    ap.add_argument('--checkpoint',required=True,help='Starting SFT adapter, or verified asset name task_sft.')
    ap.add_argument('--stats',required=True,help='Matching DATA.json, or verified asset name normalization.')
    ap.add_argument('--environment',choices=['worldgym','mujoco'],default='worldgym')
    ap.add_argument('--smoke',action='store_true',help='One group of four full trajectories and one optimizer epoch; not a performance test.')
    args=ap.parse_args();configure(args);cp=resolve_checkpoint(args.checkpoint)
    with gpu_lock():
        cfg=begin(args,'rl' if args.environment=='worldgym' else 'mujoco_rl',dict(command='grpo',environment=args.environment,
            checkpoint_sha256=sha(cp),purpose='SMOKE' if args.smoke else 'REPRODUCTION'),args.stats)
        cfg['reference_checkpoint']=str(cp);cfg['rl']=dict(cfg['rl'])
        if args.smoke:cfg['rl'].update(groups=1,epochs_per_group=1,observations_per_trajectory=2)
        target=args.run/'protocol.json'
        if not target.exists():write(target,cfg)
        from train_grpo import train
        result=train('native_GRPO' if args.environment=='worldgym' else 'MuJoCo_GRPO',cp,args.environment=='worldgym',cfg['rl']['groups'])
        finish(args.run,result)
if __name__=='__main__':main()
