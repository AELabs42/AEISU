"""Compact scientific diagnostics from frozen QRISP quality scores."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path('G:/Output Images/QRISP Final Benchmark 2026-10-04')
DESTINATION = ROOT / 'docs/manuscript/figures'
PREVIEWS = ROOT / 'output/figures/final_qrisp'


def main():
    result = json.loads((SOURCE / 'analysis_evidence.json').read_text())
    plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.labelsize': 8,
                         'axes.titlesize': 9, 'legend.fontsize': 7, 'pdf.fonttype': 42})
    figure, axes = plt.subplots(1, 2, figsize=(7, 2.55), gridspec_kw={'width_ratios': [1, 1.1]})
    for method, label, marker, color in [('aif_mixed', 'Adaptive', 'o', 'black'),
                                        ('fsr1_easu', 'EASU', 's', '0.5')]:
        x, y = [0.0], [0.0]
        for sigma in (0.5, 1.0, 2.0):
            rows = {r['metric']: r for r in result['blur']['rows']
                    if r['id'] == f'native:{method}:sigma-{sigma}-minus-unblurred'}
            x.append(rows['y_psnr_shave2_db']['point'])
            y.append(rows['lpips_alex_v01']['point'])
        axes[0].plot(x, y, marker=marker, color=color, label=label, linewidth=1, markersize=3)
        for i, sigma in enumerate((0.5, 1.0, 2.0), 1):
            axes[0].annotate(str(sigma), (x[i], y[i]), xytext=(4, 3), textcoords='offset points', fontsize=7)
    axes[0].axhline(0, color='0.75', linewidth=0.6)
    axes[0].axvline(0, color='0.75', linewidth=0.6)
    axes[0].set(xlabel='Change in Y PSNR (dB; right is better)',
                ylabel='Change in LPIPS (up is worse)', title='(a) Fixed output blur: Native')
    axes[0].set_xlim(-1.48, 0.92)
    axes[0].set_ylim(-0.02, 0.34)
    axes[0].legend(loc='lower left', frameon=False)
    methods = [('repnet_x2', 'RepNet'), ('rt4ksr_x2', 'RT4KSR'), ('quicksrnet_small_x2', 'QuickSRNet S'),
               ('quicksrnet_medium_x2', 'QuickSRNet M'), ('span_x2', 'SPAN'), ('swinir_light_x2', 'SwinIR-Light')]
    effects, errors = [], []
    for method, label in methods:
        row = next(r for r in result['relationships']['rows'] if r['metric'] == 'y_psnr_shave2_db' and
                   r['id'] == f'gap-interaction:bicubic-minus-area:aif_mixed-minus-{method}')
        effects.append(row['point'])
        lo, hi = row['simultaneous_95_percent_band']
        errors.append([row['point'] - lo, hi - row['point']])
    positions = np.arange(len(methods))
    axes[1].barh(positions, effects, color='0.65', edgecolor='black', linewidth=0.6,
                 xerr=np.array(errors).T, error_kw={'linewidth': 0.7, 'capsize': 2})
    axes[1].set_yticks(positions, [label for _, label in methods])
    axes[1].invert_yaxis()
    axes[1].set(xlabel='Change in adaptive-minus-comparator gap (dB)',
                title='(b) Bicubic minus area: PSNR interaction')
    axes[1].axvline(0, color='black', linewidth=0.7)
    for axis in axes:
        axis.spines[['top', 'right']].set_visible(False)
    figure.tight_layout(w_pad=2)
    DESTINATION.mkdir(parents=True, exist_ok=True)
    PREVIEWS.mkdir(parents=True, exist_ok=True)
    figure.savefig(DESTINATION / 'final_qrisp_diagnostics.pdf', bbox_inches='tight', pad_inches=0.03)
    figure.savefig(PREVIEWS / 'final_qrisp_diagnostics.png', dpi=220, bbox_inches='tight', pad_inches=0.03)
    provenance = {'source': str(SOURCE),
                  'analysis_sha256': hashlib.sha256((SOURCE / 'analysis_evidence.json').read_bytes()).hexdigest(),
                  'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'blur': 'All fixed sigma settings; no selected optimum',
                  'interaction': 'Paired change in relative PSNR; separate-family simultaneous bootstrap bands'}
    (PREVIEWS / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print('Saved compact final QRISP diagnostic figure.')


if __name__ == '__main__':
    main()
