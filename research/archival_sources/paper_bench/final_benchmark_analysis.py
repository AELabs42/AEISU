"""CPU-only, post-hoc paired analysis of the frozen QRISP quality scores."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


CONDITIONS = ('ssaa_area', 'bicubic_aa', 'fxaa', 'smaa', 'native')
METRICS = ('y_psnr_shave2_db', 'y_ssim_shave2', 'lpips_alex_v01')
LABELS = {'aif_mixed': 'Adaptive mixed', 'bilinear': 'Bilinear', 'bicubic': 'Bicubic',
          'fsr1_easu': 'EASU', 'fsr1_rcas_02': 'EASU+RCAS', 'nis_0': 'NIS', 'sgsr1': 'SGSR1',
          'repnet_x2': 'RepNet', 'rt4ksr_x2': 'RT4KSR', 'quicksrnet_small_x2': 'QuickSRNet Small',
          'quicksrnet_medium_x2': 'QuickSRNet Medium', 'span_x2': 'SPAN', 'swinir_light_x2': 'SwinIR-Light'}
GROUPS = ('most_similar', 'middle', 'least_similar')
RELATIONSHIPS = (
    ('fsr1_easu', 'bilinear'), ('fsr1_rcas_02', 'fsr1_easu'),
    ('nis_0', 'fsr1_easu'), ('sgsr1', 'fsr1_easu'),
    ('rt4ksr_x2', 'repnet_x2'), ('quicksrnet_medium_x2', 'quicksrnet_small_x2'),
    ('span_x2', 'quicksrnet_medium_x2'), ('swinir_light_x2', 'span_x2'),
)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def records(path):
    # These score files are immutable once inference has finished. No image arrays are opened.
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sample_weights(pairs, draws, seed):
    """Paired two-stage scene/sequence bootstrap, shared by every contrast."""
    rng = np.random.default_rng(seed)
    scenes = sorted({p['scene'] for p in pairs})
    scene_counts = rng.multinomial(len(scenes), np.ones(len(scenes)) / len(scenes), size=draws)
    weights = {'all': np.zeros((draws, len(pairs)), dtype=np.float64)}
    for scene_index, scene in enumerate(scenes):
        indices = np.array([i for i, p in enumerate(pairs) if p['scene'] == scene])
        segments = sorted({pairs[i]['segment'] for i in indices})
        segment_index = np.array([segments.index(pairs[i]['segment']) for i in indices])
        segment_counts = rng.multinomial(len(segments), np.ones(len(segments)) / len(segments), size=draws)
        raw = segment_counts[:, segment_index].astype(np.float64)
        for group, output in weights.items():
            mask = np.array([group == 'all' or pairs[i]['group'] == group for i in indices])
            chosen = raw * mask
            denom = chosen.sum(axis=1)
            if np.any(denom == 0):
                raise ValueError(f'Bootstrap has an empty {scene}/{group} resample; revise the sampling implementation.')
            output[:, indices] = chosen / denom[:, None] * (scene_counts[:, scene_index] / len(scenes))[:, None]
    for output in weights.values():
        np.testing.assert_allclose(output.sum(axis=1), 1, atol=1e-12)
    return scenes, weights


def point_weights(pairs, group='all', omit=None):
    scenes = sorted({p['scene'] for p in pairs if p['scene'] != omit})
    weight = np.zeros(len(pairs))
    for scene in scenes:
        indices = [i for i, p in enumerate(pairs) if p['scene'] == scene and (group == 'all' or p['group'] == group)]
        weight[indices] = 1 / len(indices) / len(scenes)
    return weight


def add_contrast(target, pairs, identifier, values, group='all', other_values=None, other_group=None):
    if other_values is None:
        other_values = np.zeros_like(values)
    if other_group is None:
        other_group = group
    scenes = sorted({p['scene'] for p in pairs})
    point = point_weights(pairs, group) @ values - point_weights(pairs, other_group) @ other_values
    scene_effects = {}
    for scene in scenes:
        indices = [i for i, p in enumerate(pairs) if p['scene'] == scene and (group == 'all' or p['group'] == group)]
        other = [i for i, p in enumerate(pairs) if p['scene'] == scene and (other_group == 'all' or p['group'] == other_group)]
        scene_effects[scene] = (values[indices].mean(axis=0) - other_values[other].mean(axis=0)).tolist()
    leave_one_out = [point_weights(pairs, group, scene) @ values - point_weights(pairs, other_group, scene) @ other_values
                     for scene in scenes]
    for metric_index, metric in enumerate(METRICS):
        target.append({'id': identifier, 'metric': metric, 'point': float(point[metric_index]),
                       'scene_effects': {s: float(v[metric_index]) for s, v in scene_effects.items()},
                       'leave_one_scene_out_range': [float(np.min(np.array(leave_one_out)[:, metric_index])),
                                                    float(np.max(np.array(leave_one_out)[:, metric_index]))],
                       '_values': values[:, metric_index], '_other': other_values[:, metric_index],
                       '_group': group, '_other_group': other_group})


def summarize_family(contrasts, weights):
    """Nominal percentile intervals and simultaneous standardized-deviation bands."""
    samples = np.column_stack([weights[c['_group']] @ c['_values'] - weights[c['_other_group']] @ c['_other']
                               for c in contrasts])
    points = np.array([c['point'] for c in contrasts])
    standard_errors = samples.std(axis=0, ddof=1)
    safe = np.where(standard_errors > 1e-15, standard_errors, 1)
    maximum = np.max(np.abs((samples - points) / safe), axis=1)
    critical = float(np.quantile(maximum, 0.95))
    nominal = np.quantile(samples, [0.025, 0.975], axis=0)
    for i, contrast in enumerate(contrasts):
        contrast['nominal_95_percent_interval'] = nominal[:, i].tolist()
        contrast['simultaneous_95_percent_band'] = [float(points[i] - critical * standard_errors[i]),
                                                  float(points[i] + critical * standard_errors[i])]
        contrast['nominal_excludes_zero'] = bool(nominal[0, i] > 0 or nominal[1, i] < 0)
        lo, hi = contrast['simultaneous_95_percent_band']
        contrast['simultaneous_excludes_zero'] = bool(lo > 0 or hi < 0)
        for key in [key for key in contrast if key.startswith('_')]:
            del contrast[key]
    return {'contrasts': len(contrasts), 'simultaneous_critical_value': critical, 'rows': contrasts}


def format_effect(row, band='nominal_95_percent_interval'):
    precision = 3 if row['metric'] == METRICS[0] else 5
    lo, hi = row[band]
    return f"{row['point']:+.{precision}f} [{lo:+.{precision}f}, {hi:+.{precision}f}]"


def render_evidence(output, result):
    lines = ['# Paired statistical evidence for the QRISP quality analysis', '',
             'Post-hoc analysis of saved scores only; no inference, new image selection, or GPU work.', '',
             'Each difference is left minus right. Positive PSNR/SSIM and negative LPIPS favor the left method/condition.',
             'Intervals resample environments and then source-sequence clusters, preserving pairing across methods and conditions.',
             'There are five environments and 55 source-sequence clusters. These are exploratory uncertainty estimates, not evidence',
             'of representativeness across games. Sequence clusters do not guarantee complete temporal independence.', '',
             'Nominal 95% intervals are shown alongside family-wise simultaneous 95% standardized-bootstrap bands.',
             'The latter account for contrasts in each named family, not every possible post-hoc analysis.',
             'A zero-excluding band is described as bootstrap-supported under this design, not a universal or causal finding.',
             'The bands use a max-standardized-deviation bootstrap; they are not exact small-sample tests.',
             'With only five independent environments, even an exact two-sided sign-flip test has a minimum possible p-value',
             'of 2/32 = 0.0625. Broad claims of conventional scene-level statistical significance are therefore inappropriate.', '',
             'The source selection balances consistency strata equally. Means describe that sample, not population prevalence.', '',
             '## Main method comparisons', '',
             'Family: adaptive minus each of 12 baselines, five conditions, three primary metrics: 180 contrasts.', '']
    for condition in CONDITIONS:
        lines += ['', f'### {condition}', '',
                  '| Comparator | Δ Y PSNR, nominal 95% interval | Δ Y SSIM, nominal 95% interval | '
                  'Δ LPIPS, nominal 95% interval | Family-wise supported direction |',
                  '|---|---:|---:|---:|---|']
        for method in LABELS:
            if method == 'aif_mixed':
                continue
            rows = [r for r in result['main']['rows'] if r['id'] == f'{condition}:aif_mixed-minus-{method}']
            supported = []
            for row in rows:
                if row['simultaneous_excludes_zero']:
                    direction = 'adaptive' if (row['point'] > 0) == (row['metric'] != METRICS[2]) else 'baseline'
                    supported.append(f"{('PSNR', 'SSIM', 'LPIPS')[METRICS.index(row['metric'])]}: {direction}")
            lines.append(f"| {LABELS[method]} | {' | '.join(format_effect(r) for r in rows)} | {'; '.join(supported) or 'unresolved'} |")
    for family, title in [('input', 'Within-adaptive input-condition contrasts'), ('blur', 'Fixed output-blur effects'),
                          ('relationships', 'Cross-model relationships and degradation interactions')]:
        lines += ['', f'## {title}', '',
                  f"Family contains {result[family]['contrasts']} contrasts. All specified contrasts for this family are retained.", '',
                  '| Contrast | Metric | Effect, nominal 95% interval | Simultaneous 95% band | Leave-one-environment-out range |',
                  '|---|---|---:|---:|---:|']
        for row in result[family]['rows']:
            lo, hi = row['leave_one_scene_out_range']
            lines.append(f"| {row['id']} | {row['metric']} | {format_effect(row)} | "
                         f"{format_effect(row, 'simultaneous_95_percent_band')} | [{lo:+.5f}, {hi:+.5f}] |")
    lines += ['', '## Descriptive consistency-stratum contrasts', '',
              'No bootstrap inference is assigned to these contrasts. Extreme strata have sparse sequence coverage in some environments;',
              'a joint sequence resample can omit a stratum entirely. The table retains observed effects and leave-one-environment-out sensitivity.', '',
              '| Contrast | Metric | Observed effect | Leave-one-environment-out range |', '|---|---|---:|---:|']
    for row in result['consistency']['rows']:
        lo, hi = row['leave_one_scene_out_range']
        lines.append(f"| {row['id']} | {row['metric']} | {row['point']:+.5f} | [{lo:+.5f}, {hi:+.5f}] |")
    lines += ['', '## Main comparison robustness by environment', '',
              '| Contrast | Metric | Favorable environments / 5 | Leave-one-environment-out effect range | Paired frame win fraction |',
              '|---|---|---:|---:|---:|']
    for row in result['main']['rows']:
        lo, hi = row['leave_one_scene_out_range']
        wins = sum((v > 0) if row['metric'] != METRICS[2] else (v < 0) for v in row['scene_effects'].values())
        lines.append(f"| {row['id']} | {row['metric']} | {wins} | [{lo:+.5f}, {hi:+.5f}] | {row['paired_frame_win_fraction']:.3f} |")
    lines += ['', 'Frame win fractions describe the selected sample; 600 frames are not treated as 600 independent scenes.', '',
              '## Texture and source-consistency context', '',
              '| Stratum | Mean LR luma variance | Mean LR gradient energy | Mean source-pair Y MSE | Mean lowpass source-pair Y MSE |',
              '|---|---:|---:|---:|---:|']
    for group, values in result['source_context'].items():
        lines.append(f"| {group} | {values['lr_y_variance']:.6f} | {values['lr_gradient_energy']:.6f} | "
                     f"{values['pair_y_mse']:.6f} | {values['lowpass_pair_y_mse']:.6f} |")
    lines += ['', '### Source-sequence coverage within strata', '',
              '| Environment | Most-similar sequences | Middle sequences | Least-similar sequences |', '|---|---:|---:|---:|']
    for scene, counts in result['stratum_sequence_support'].items():
        lines.append(f"| {scene} | {counts['most_similar']} | {counts['middle']} | {counts['least_similar']} |")
    lines += ['', f"Y-versus-RGB aggregate direction disagreements: {len(result['rgb_y_direction_disagreements'])}/120 comparisons."]
    lines += ['', 'No nuisance-adjusted causal effect is estimated. These diagnostics expose confounding rather than removing it.', '',
              'Full scene effects, MSE ratios, and analysis provenance are in analysis_evidence.json.']
    (output / 'analysis_evidence.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('G:/Output Images/QRISP Final Benchmark 2026-10-04'))
    parser.add_argument('--draws', type=int, default=50000)
    args = parser.parse_args()
    output = args.output
    pairs = read_json(output / 'selection.json')['pairs']
    summary = read_json(output / 'summary.json')
    if summary['status'] != 'complete':
        raise ValueError('Main quality results are not complete.')
    index = {p['image']: i for i, p in enumerate(pairs)}
    data, mse = {}, {}
    source_hashes = {}
    for method in LABELS:
        path = output / 'results' / method / 'per_image.jsonl'
        rows = records(path)
        source_hashes[str(path.relative_to(output))] = sha256(path)
        identities = {(r['condition'], r['image']) for r in rows}
        if len(rows) != 3000 or len(identities) != 3000:
            raise ValueError(f'Incomplete or duplicate scores for {method}.')
        for condition in CONDITIONS:
            chosen = sorted((r for r in rows if r['condition'] == condition), key=lambda r: index[r['image']])
            if len(chosen) != len(pairs):
                raise ValueError('Missing condition scores.')
            data[method, condition] = np.array([[r['metrics'][k] for k in METRICS] for r in chosen])
            mse[method, condition] = np.array([r['metrics']['y_mse_shave2'] for r in chosen])
            np.testing.assert_allclose(point_weights(pairs) @ data[method, condition],
                                       [summary['by_condition'][condition]['all'][method]['scene_balanced'][k] for k in METRICS], atol=1e-12)
    scenes, weights = sample_weights(pairs, args.draws, 2026100402)
    families = {name: [] for name in ('main', 'input', 'blur', 'consistency', 'relationships')}
    for condition in CONDITIONS:
        for method in LABELS:
            if method == 'aif_mixed':
                continue
            difference = data['aif_mixed', condition] - data[method, condition]
            add_contrast(families['main'], pairs, f'{condition}:aif_mixed-minus-{method}', difference)
            for k, row in enumerate(families['main'][-3:]):
                wins = (difference[:, k] > 0) if k < 2 else (difference[:, k] < 0)
                row['paired_frame_win_fraction'] = float(np.clip(point_weights(pairs) @ wins, 0, 1))
                row['median_paired_frame_difference'] = float(np.median(difference[:, k]))
    for left, right in itertools.combinations(CONDITIONS, 2):
        add_contrast(families['input'], pairs, f'{left}-minus-{right}', data['aif_mixed', left] - data['aif_mixed', right])
    for condition in CONDITIONS:
        for left, right in RELATIONSHIPS:
            add_contrast(families['relationships'], pairs, f'{condition}:{left}-minus-{right}',
                         data[left, condition] - data[right, condition])
    for method in ('repnet_x2', 'rt4ksr_x2', 'quicksrnet_small_x2', 'quicksrnet_medium_x2', 'span_x2', 'swinir_light_x2'):
        area_gap = data['aif_mixed', 'ssaa_area'] - data[method, 'ssaa_area']
        bicubic_gap = data['aif_mixed', 'bicubic_aa'] - data[method, 'bicubic_aa']
        add_contrast(families['relationships'], pairs, f'gap-interaction:bicubic-minus-area:aif_mixed-minus-{method}',
                     bicubic_gap - area_gap)
    for condition in CONDITIONS:
        add_contrast(families['consistency'], pairs, f'{condition}:adaptive-most-minus-least', data['aif_mixed', condition],
                     'most_similar', data['aif_mixed', condition], 'least_similar')
        delta = data['aif_mixed', condition] - data['fsr1_easu', condition]
        add_contrast(families['consistency'], pairs, f'{condition}:adaptive-EASU-gap-most-minus-least', delta,
                     'most_similar', delta, 'least_similar')
    path = output / 'blur_per_image.jsonl'
    blur = records(path)
    if len(blur) != 10800 or len({(r['method'], r['condition'], r['image'], r['sigma']) for r in blur}) != 10800:
        raise ValueError('Blur scores incomplete or duplicated.')
    source_hashes[str(path.relative_to(output))] = sha256(path)
    for condition in ('native', 'fxaa', 'smaa'):
        for method in ('aif_mixed', 'fsr1_easu'):
            for sigma in (0.5, 1.0, 2.0):
                chosen = sorted((r for r in blur if (r['method'], r['condition'], r['sigma']) == (method, condition, sigma)),
                                key=lambda r: index[r['image']])
                if len(chosen) != len(pairs):
                    raise ValueError('Missing blur condition.')
                delta = np.array([[r['metrics'][k] - r['baseline_metrics'][k] for k in METRICS] for r in chosen])
                add_contrast(families['blur'], pairs, f'{condition}:{method}:sigma-{sigma}-minus-unblurred', delta)
    result = {'analysis': 'Post-hoc paired quality-score analysis', 'draws': args.draws, 'seed': 2026100402,
              'resampling': 'Scenes, then source-sequence clusters; joint pairing across all methods and input conditions',
              'intervals': 'Nominal percentile 95%; separate-family 95% max-standardized-deviation simultaneous bands',
              'environment_count': len(scenes), 'sequence_cluster_count': len({(p['scene'], p['segment']) for p in pairs}),
              'metrics': list(METRICS), 'analysis_source_sha256': sha256(Path(__file__)), 'score_hashes': source_hashes,
              'numpy_version': np.__version__, 'source_context': {}, 'mse_ratios': {}}
    for group in GROUPS:
        result['source_context'][group] = {key: float(point_weights(pairs, group) @ np.array([p[key] for p in pairs]))
                                          for key in ('lr_y_variance', 'lr_gradient_energy', 'pair_y_mse', 'lowpass_pair_y_mse')}
    result['stratum_sequence_support'] = {
        scene: {group: len({p['segment'] for p in pairs if p['scene'] == scene and p['group'] == group})
                for group in GROUPS} for scene in scenes}
    result['rgb_y_direction_disagreements'] = []
    for condition in CONDITIONS:
        adaptive = summary['by_condition'][condition]['all']['aif_mixed']['scene_balanced']
        for method in LABELS:
            if method == 'aif_mixed':
                continue
            other = summary['by_condition'][condition]['all'][method]['scene_balanced']
            for y_key, rgb_key in ((METRICS[0], 'rgb_psnr_shave2_db'), (METRICS[1], 'rgb_ssim_shave2')):
                if np.sign(adaptive[y_key] - other[y_key]) != np.sign(adaptive[rgb_key] - other[rgb_key]):
                    result['rgb_y_direction_disagreements'].append({'condition': condition, 'method': method, 'metric': y_key})
    for condition in CONDITIONS:
        result['mse_ratios'][condition] = {method: float((point_weights(pairs) @ mse['aif_mixed', condition]) /
                                                       (point_weights(pairs) @ mse[method, condition])) for method in LABELS}
    for name, contrasts in families.items():
        if name == 'consistency':
            for contrast in contrasts:
                for key in [key for key in contrast if key.startswith('_')]:
                    del contrast[key]
            result[name] = {'contrasts': len(contrasts), 'inference': 'Descriptive only; sparse extreme-stratum sequence support', 'rows': contrasts}
            continue
        result[name] = summarize_family(contrasts, weights)
        print(f'{name}: {len(contrasts)} paired metric contrasts analyzed', flush=True)
    write_json(output / 'analysis_evidence.json', result)
    render_evidence(output, result)
    print('Saved analysis_evidence.json and analysis_evidence.md; original scores and manuscript unchanged.', flush=True)


if __name__ == '__main__':
    main()
