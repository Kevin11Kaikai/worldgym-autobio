"""Train the complete pretrained SmolVLA action expert on 40 AutoBio demonstrations."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'runtime'))
from cli import common,begin,configure,resolve_checkpoint,gpu_lock,finish
from control import write,read,sha
def main():
    ap=argparse.ArgumentParser(description=__doc__);common(ap)
    ap.add_argument('--smoke',action='store_true',help='Two real training steps; engineering check, not a trained replacement.')
    ap.add_argument('--continue-from',help='Continue a selected SFT adapter for 3000 steps, e.g. task_sft.')
    ap.add_argument('--precision',choices=['mixed_original','fp32'],default='mixed_original')
    ap.add_argument('--learning-rate',choices=['original','scaled'],default='original',help='Continuation: 1e-5 or the predeclared 4.601757083066472e-6.')
    args=ap.parse_args();configure(args)
    cp=resolve_checkpoint(args.continue_from) if args.continue_from else None
    if not cp and (args.precision!='mixed_original' or args.learning_rate!='original'):
        ap.error('--precision/--learning-rate refer to --continue-from, not initial SFT.')
    with gpu_lock():
        cfg=begin(args,'sft',dict(command='sft',purpose='SMOKE' if args.smoke else 'REPRODUCTION',
            continue_sha256=sha(cp) if cp else None,precision=args.precision,learning_rate=args.learning_rate))
        if cp:
            p=dict(base=str(cp),steps=2 if args.smoke else 3000,seed=801511,batch=4,
                   lr=1e-5 if args.learning_rate=='original' else 4.601757083066472e-6)
            target=args.run/'SFT_REPAIR_PROTOCOL.json'
            if not target.exists():write(target,p)
            import torch
            torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
            from continue_sft import train
            train(args.precision)
            finish(args.run,read(args.run/'sft_repair'/args.precision/'COMPLETE.json'));return
        p=cfg['sft'].copy()
        if args.smoke:p.update(steps=2,checkpoint_steps=[2])
        target=args.run/'SFT_PROTOCOL.json'
        if not target.exists():write(target,p)
        from train_sft import train
        finish(args.run,train())
if __name__=='__main__':main()
