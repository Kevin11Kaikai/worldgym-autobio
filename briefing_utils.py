"""Small, CPU-only helpers for the three briefing notebooks."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from IPython.display import display, HTML

ROOT = Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
                     'axes.spines.top':False,'axes.spines.right':False,
                     'figure.dpi':120,'savefig.bbox':'tight'})
TEAL, NAVY, ORANGE, GRAY = '#168b83', '#183b52', '#cc6b42', '#879ba8'

def data(name):
    return json.loads((ROOT / 'evidence' / name).read_text(encoding='utf-8-sig'))

def outcomes(round_name):
    df = pd.read_csv(ROOT / 'evidence' / f'{round_name}_outcomes.csv')
    if round_name == 'gain':
        wide = df.set_index('seed')
    else:
        assert not df.duplicated(['policy','seed']).any()
        wide = df.pivot(index='seed',columns='policy',values='success')
    assert len(wide) == 100 and not wide.isna().any().any()
    assert set(np.unique(wide.values)).issubset({0,1})
    return wide.astype(int)

def verify_assets():
    manifest = json.loads((ROOT / 'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    for row in manifest['files']:
        assert hashlib.sha256((ROOT / row['path']).read_bytes()).hexdigest() == row['sha256'], row['path']
    return len(manifest['files'])

def table(rows, columns=None):
    df = rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows,columns=columns)
    display(HTML(df.to_html(index=False,escape=True,border=0)))

def pipeline():
    fig, ax = plt.subplots(figsize=(11.8,5.2))
    ax.set(xlim=(0,12),ylim=(0,6)); ax.axis('off')
    def box(x,y,w,title,sub,color=TEAL):
        ax.add_patch(FancyBboxPatch((x,y),w,1.0,boxstyle='round,pad=0.05,rounding_size=0.12',
                                   facecolor=color,edgecolor='none'))
        ax.text(x+w/2,y+.65,title,color='white',weight='bold',ha='center',va='center',fontsize=12)
        ax.text(x+w/2,y+.27,sub,color='white',ha='center',va='center',fontsize=9.5)
    def arrow(a,b,color=NAVY,rad=0):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=15,lw=1.8,
                                    color=color,connectionstyle=f'arc3,rad={rad}'))
    box(.2,4.7,2.6,'Demonstrations','images + joint targets',GRAY)
    box(3.6,4.7,2.6,'SFT','imitate expert actions',NAVY)
    box(7,4.7,4.5,'Initial SmolVLA policy','the native world-model RL branch',NAVY)
    arrow((2.85,5.2),(3.5,5.2)); arrow((6.25,5.2),(6.9,5.2))
    box(.2,2.45,2.4,'SmolVLA policy','observation -> actions')
    box(3.15,2.45,2.4,'WorldGym','actions -> future views')
    box(6.1,2.45,2.4,'VLM reward','views + task -> score')
    box(9.05,2.45,2.4,'GRPO','update policy weights')
    for a,b in [(2.65,3.05),(5.6,6),(8.55,8.95)]: arrow((a,2.95),(b,2.95))
    arrow((4.35,2.35),(1.4,2.35),rad=-.3)
    ax.text(2.9,1.65,'Generated observations feed the next action',ha='center',fontsize=9.5,color=NAVY)
    ax.plot([10.25,10.25,1.4,1.4],[3.55,4.05,4.05,3.55],color=TEAL,lw=1.5)
    arrow((1.4,3.85),(1.4,3.5),color=TEAL)
    ax.text(6.2,4.16,'repeat rollout -> score -> update',ha='center',fontsize=10,color=TEAL)
    box(3.25,.15,5.45,'Separate MuJoCo evaluation','official task success; frozen, unseen scenes',NAVY)
    ax.text(.2,.5,'Final policy',weight='bold',color=NAVY)
    arrow((1.95,.65),(3.1,.65))
    ax.text(9.05,.65,'Simulation evidence\nNo real robot trial',ha='left',va='center',color=ORANGE,fontsize=10)
    fig.tight_layout(); plt.show()

def input_views():
    with np.load(ROOT/'evidence/observation_example.npz') as z:
        fig, axes = plt.subplots(1,2,figsize=(7.4,3.7))
        for ax,key,title in zip(axes,['front','wrist'],['Front camera','Wrist camera']):
            ax.imshow(z[key]);ax.set_title(title);ax.axis('off')
        fig.suptitle('Saved initial observation | fixed ACT40 demo rollout',fontsize=12)
        fig.tight_layout(); plt.show()
        assert z['state'].shape==(7,) and z['action_chunk'].shape==(50,7)

def result_panels():
    panels = [
      ('gain','R1 | Policy selection',[
        ('original_ACT','Original ACT'),('previous_MVP_selected','Previous MVP'),
        ('FT40_s5101_step10000','WG-selected / fixed FT40'),('FT20_s5102_step5000','No-WG visual selection'),
        ('MuJoCo_s5102_g16','Initial-action selection'),('FT20_s5101_step5000','MuJoCo dev selection'),
        ('reference_ACT80','ACT80 reference')]),
      ('native','R2 | Native world-model RL',[
        ('SFT','SmolVLA SFT'),('MuJoCo_GRPO','MuJoCo GRPO'),('native_GRPO','Native WG GRPO'),
        ('native_continued','Native continued'),('Dyna_GRPO','Dyna GRPO')]),
      ('repair','R3 | Diagnosis + SFT',[
        ('SFT_original','SmolVLA starting SFT'),('SFT_mixed_original_lr','Mixed / original LR'),
        ('SFT_fp32_original_lr','FP32 / original LR'),('SFT_mixed_scaled_lr','Mixed / scaled LR [selected]'),
        ('SFT_fp32_scaled_lr','FP32 / scaled LR'),('ACT40','ACT40 strong reference')])]
    fig, axes = plt.subplots(1,3,figsize=(15,5.1),layout='constrained')
    for ax,(round_name,title,items) in zip(axes,panels):
        wide=outcomes(round_name); labels=[x[1] for x in items]; vals=[int(wide[x[0]].sum()) for x in items]
        colors=[TEAL if ('selected' in x.lower() or x=='Native WG GRPO') else NAVY if 'ACT' in x else GRAY for x in labels]
        bars=ax.barh(np.arange(len(vals)),vals,color=colors,height=.62)
        ax.set_yticks(np.arange(len(vals)),labels,fontsize=9.5);ax.invert_yaxis()
        ax.set_xlim(0,112);ax.set_xticks([0,25,50,75,100]);ax.set_xlabel('Successes / 100 scenes')
        ax.set_title(title,weight='bold',fontsize=11,pad=16)
        ax.bar_label(bars,padding=4,fontsize=10);ax.grid(axis='x',alpha=.12);ax.set_axisbelow(True)
    plt.show()

def comparison_table():
    gain=data('gain_results.json')['comparisons']; native=data('native_summary.json'); repair=data('repair_results.json')
    specs=[
      ('R1','WG-selected - original ACT','gain','FT40_s5101_step10000','original_ACT',[100*x for x in gain['WorldGym_vs_original']['paired_bootstrap95']],20,4),
      ('R1','WG-selected - no-WG visual selector','gain','FT40_s5101_step10000','FT20_s5102_step5000',[100*x for x in gain['WorldGym_vs_noWorldGym_visual']['paired_bootstrap95']],6,5),
      ('R2','Native WG RL - SFT','native','native_GRPO','SFT',native['paired_comparisons']['native_GRPO_minus_SFT']['bootstrap95_pp'],6,12),
      ('R2','Dyna RL - matched continuation','native','Dyna_GRPO','native_continued',native['paired_comparisons']['Dyna_GRPO_minus_native_continued']['bootstrap95_pp'],11,16),
      ('R3','Selected SFT - starting SFT','repair','SFT_mixed_scaled_lr','SFT_original',repair['paired']['selected_SFT_vs_original']['ci95_pp'],44,10),
      ('R3','Selected SFT - ACT40','repair','SFT_mixed_scaled_lr','ACT40',repair['paired']['selected_SFT_vs_ACT40']['ci95_pp'],2,40),
      ('R3','Scaled LR - original LR (mixed)','repair','SFT_mixed_scaled_lr','SFT_mixed_original_lr',repair['paired']['rate_change_mixed']['ci95_pp'],7,6)]
    rows=[]
    for stage,label,r,a,b,ci,win,harm in specs:
        wide=outcomes(r);d=wide[a]-wide[b]; w=int((d==1).sum());h=int((d==-1).sum())
        if win is not None:assert (w,h)==(win,harm)
        rows.append([stage,label,f'{int(d.sum()):+d}',f'[{ci[0]:.0f}, {ci[1]:.0f}]',w,h])
    table(rows,['Round','Paired comparison','Change (pp)','Archived 95% interval','Gained scenes','New failures'])
    return rows

def diagnostic_tables():
    d=data('repair_diagnostics.json')
    wm=[]
    for key,label in [('hold_initial','Hold still'),('policy_replay','Move'),('reverse_gripper','Reverse gripper')]:
        a=d['worldmodel_means']['replay'][key]['prediction_MAE'];b=d['worldmodel_means']['holdmix'][key]['prediction_MAE']
        wm.append([label,f'{a:.4f}',f'{b:.4f}',f'{100*(b/a-1):+.1f}%'])
    table(wm,['Matched initial state and action','Replay-only MAE','Hold-mix MAE','Relative error change (lower is better)'])
    rows=[]
    for k,label in [('original','Original reward'),('prompt_only','Prompt only'),('repaired','Prompt + weight update')]:
        m=d['reward'][k];rows.append([label,f"{100*m['balanced_accuracy']:.1f}%",m['false_positive'],m['false_negative']])
    table(rows,['VLM reward (192 frames)','Balanced accuracy (higher is better)','False positives','False negatives'])

def model_action_plot():
    d=data('repair_diagnostics.json');actions=['hold_initial','policy_replay','reverse_gripper']
    fig,ax=plt.subplots(figsize=(8.2,3.7))
    x=np.arange(3);w=.24
    for i,(name,label,col) in enumerate([('base','Base WG',GRAY),('replay','Replay-only',NAVY),('holdmix','Hold-mix',TEAL)]):
        vals=[d['worldmodel_means'][name][a]['prediction_MAE'] for a in actions]
        ax.bar(x+(i-1)*w,vals,w,label=label,color=col)
    ax.set_xticks(x,['Hold still','Move','Reverse gripper']);ax.set_ylabel('Prediction MAE (lower is better)')
    ax.set_ylim(0,.11);ax.legend(ncol=3,frameon=False,loc='upper left',fontsize=10)
    ax.set_title('A static-scene repair is not a general dynamics repair',fontsize=12,pad=12)
    fig.tight_layout();plt.show()
