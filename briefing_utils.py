"""Small CPU-only evidence readers and plots for the three stage lessons."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import HTML, display

ROOT = Path(__file__).resolve().parent
TEAL, NAVY, ORANGE, GRAY = '#168b83', '#183b52', '#cc6b42', '#879ba8'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 120, 'savefig.bbox': 'tight'})


def data(name):
    return json.loads((ROOT / 'evidence' / name).read_text(encoding='utf-8-sig'))


def table(rows, columns=None):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows, columns=columns)
    display(HTML(frame.to_html(index=False, escape=True, border=0)))


def verify_assets():
    manifest = json.loads((ROOT / 'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    assert len({row['path'] for row in manifest['files']}) == len(manifest['files'])
    for row in manifest['files']:
        assert hashlib.sha256((ROOT / row['path']).read_bytes()).hexdigest() == row['sha256'], row['path']
    return len(manifest['files'])


def outcomes(round_name):
    frame = pd.read_csv(ROOT / 'evidence' / f'{round_name}_outcomes.csv')
    if round_name == 'gain':
        wide = frame.set_index('seed')
    else:
        assert not frame.duplicated(['policy', 'seed']).any()
        wide = frame.pivot(index='seed', columns='policy', values='success')
    assert len(wide) == 100 and not wide.isna().any().any()
    assert set(np.unique(wide.values)).issubset({0, 1})
    return wide.astype(int)


def check_stage_evidence():
    p1 = pd.read_csv(ROOT / 'evidence/phase1_outcomes.csv')
    assert len(p1) == 100 and p1['valid'].eq(1).all()
    assert not p1.duplicated(['condition', 'seed']).any()
    one = p1.pivot(index='seed', columns='condition', values='success')
    assert one.shape == (50, 2) and not one.isna().any().any()
    assert one.sum().to_dict() == {10: 31, 80: 50}
    p2 = pd.read_csv(ROOT / 'evidence/phase2_outcomes.csv')
    assert len(p2) == 140 and not p2.duplicated(['model', 'seed']).any()
    assert p2.pivot(index='seed', columns='model', values='success').shape == (20, 7)
    for column, key in [('worldgym_probability', 'main'), ('no_worldgym_probability', 'matched_no_worldgym_visual')]:
        saved = data('phase2_summary.json')[key]
        actual = p2.groupby('model')['success'].mean()
        predicted = p2.groupby('model')[column].mean()
        assert np.isclose((p2[column].ge(.5) == p2['success'].astype(bool)).mean(), saved['classification']['accuracy'])
        assert np.isclose(np.mean((p2[column] - p2['success']) ** 2), saved['classification']['Brier'])
        correct = total = 0
        for i, a in enumerate(actual.index):
            for b in actual.index[i + 1:]:
                if actual[a] != actual[b]:
                    total += 1
                    correct += (actual[a] - actual[b]) * (predicted[a] - predicted[b]) > 0
        assert (correct, total) == (saved['correct_pairs'], saved['non_tied_pairs'])
        assert np.isclose(actual.rank().corr(predicted.rank()), saved['spearman'])
    return {'phase1_paired_scenes': len(one), 'phase2_paired_scenes': 20,
            'status': 'PASS_SAVED_OUTCOME_ARITHMETIC', 'new_inference': False}


def input_views():
    with np.load(ROOT / 'evidence/observation_example.npz') as sample:
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5))
        for ax, key, title in zip(axes, ['front', 'wrist'], ['Front camera', 'Wrist camera']):
            ax.imshow(sample[key]); ax.set_title(title); ax.axis('off')
        fig.suptitle('Actual saved observation | later ACT40 demonstration', fontsize=11)
        fig.tight_layout(); plt.show()
        assert sample['state'].shape == (7,) and sample['action_chunk'].shape == (50, 7)
        print('Joint-state shape:', sample['state'].shape)
        print('Action-chunk shape:', sample['action_chunk'].shape)


def phase1_results():
    frame = pd.read_csv(ROOT / 'evidence/phase1_outcomes.csv')
    rows = []
    for condition in [10, 80]:
        group = frame[frame.condition.eq(condition)]
        rows.append([f'{condition}-demo ACT', int(group.success.sum()), len(group), f'{100 * group.success.mean():.1f}%'])
    table(rows, ['Repaired policy', 'Successes', 'Paired scenes', 'Observed success rate'])


def phase2_results():
    frame = pd.read_csv(ROOT / 'evidence/phase2_outcomes.csv')
    actual = frame.groupby('model').success.agg(['sum', 'mean'])
    predicted = frame.groupby('model').worldgym_probability.mean()
    labels = {'86191aa6': '10-demo original', 'f12ea971': '80-demo original',
              'c24fbea8': '80-demo repaired', 'ea4310c7': '10-demo repaired',
              'c6ebfff0': '10-demo candidate A', 'ef31caa8': '10-demo candidate B', 'fbd1aed0': '10-demo candidate C'}
    order = ['86191aa6', 'f12ea971', 'ea4310c7', 'c6ebfff0', 'ef31caa8', 'fbd1aed0', 'c24fbea8']
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    y = np.arange(len(order))
    ax.barh(y - .17, [100 * actual.loc[k, 'mean'] for k in order], .32, label='MuJoCo measured success rate', color=NAVY)
    ax.barh(y + .17, [100 * predicted[k] for k in order], .32, label='Hybrid mean predicted probability', color=TEAL)
    ax.set_yticks(y, [labels[k] for k in order]); ax.invert_yaxis()
    ax.set_xlim(0, 110); ax.set_xlabel('Percent | 20 common initial states per policy')
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.01), frameon=False, fontsize=9)
    fig.tight_layout(); plt.show()
    table([[labels[k], f"{int(actual.loc[k, 'sum'])}/20", f'{100 * predicted[k]:.1f}%'] for k in order],
          ['Frozen ACT policy', 'MuJoCo successes', 'Hybrid predicted probability'])


def native_results():
    wide = outcomes('native')
    names = [('SFT', 'SFT start'), ('native_GRPO', 'Native WorldGym GRPO'), ('MuJoCo_GRPO', 'Matched MuJoCo GRPO'),
             ('native_continued', 'Native continued'), ('Dyna_GRPO', 'Dyna GRPO')]
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    bars = ax.barh([name for _, name in names], [wide[key].sum() for key, _ in names], color=[NAVY, TEAL, GRAY, TEAL, ORANGE])
    ax.invert_yaxis(); ax.set_xlim(0, 100); ax.set_xlabel('Official MuJoCo successes / 100 common new scenes')
    ax.bar_label(bars, padding=4); fig.tight_layout(); plt.show()
    rows = []
    for comparison, a, b in [('native_GRPO_minus_SFT', 'native_GRPO', 'SFT'), ('Dyna_GRPO_minus_native_continued', 'Dyna_GRPO', 'native_continued')]:
        diff = wide[a] - wide[b]
        saved = data('native_summary.json')['paired_comparisons'][comparison]
        rows.append([f'{a} - {b}', int(diff.sum()), int(diff.eq(1).sum()), int(diff.eq(-1).sum()),
                     '[' + ', '.join(f'{x:.0f}' for x in saved['bootstrap95_pp']) + ']'])
    table(rows, ['Paired comparison', 'Change (pp)', 'Improved scenes', 'New failures', 'Archived 95% interval (pp)'])


def model_action_plot():
    result = data('repair_diagnostics.json')
    actions = ['hold_initial', 'policy_replay', 'reverse_gripper']
    fig, ax = plt.subplots(figsize=(8.2, 3.7)); x = np.arange(3); width = .24
    for i, (name, label, color) in enumerate([('base', 'Base WorldGym', GRAY), ('replay', 'Replay-only', NAVY), ('holdmix', 'Hold-mix', TEAL)]):
        values = [result['worldmodel_means'][name][a]['prediction_MAE'] for a in actions]
        ax.bar(x + (i - 1) * width, values, width, label=label, color=color)
    ax.set_xticks(x, ['Hold still', 'Move', 'Reverse gripper']); ax.set_ylabel('Prediction MAE (lower is better)')
    ax.set_ylim(0, .11); ax.legend(ncol=3, frameon=False, loc='upper left', fontsize=9)
    fig.tight_layout(); plt.show()
