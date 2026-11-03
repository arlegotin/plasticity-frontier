"""Render the README comparisons from the committed, verified CSV exports."""
import argparse
import os
from pathlib import Path
import tempfile

from verify_public_results import rows, verify

INK, BLUE, ORANGE, GRID = '#142c3e', '#376799', '#bf5f28', '#dfe5e9'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = verify()
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='plasticity-mpl-') as config:
        os.environ['MPLCONFIGDIR'] = config
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plt.rcParams.update({
            'font.family': 'DejaVu Sans', 'font.size': 11, 'text.color': INK,
            'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
            'axes.edgecolor': GRID, 'axes.spines.top': False,
            'axes.spines.right': False, 'axes.titleweight': 'bold',
            'figure.facecolor': 'white', 'axes.facecolor': 'white',
            'svg.hashsalt': 'plasticity-frontier-v0.1.0', 'svg.fonttype': 'none',
        })

        def save(fig, name):
            for extension in ('svg', 'png'):
                metadata = {'Date': None} if extension == 'svg' else {'Software': 'Plasticity Frontier v0.1.0'}
                fig.savefig(args.output / f'{name}.{extension}', dpi=180, metadata=metadata)
            plt.close(fig)

        def heading(fig, title, subtitle):
            fig.text(.045, .96, title, fontsize=18, weight='bold', va='top')
            fig.text(.045, .885, subtitle, fontsize=10.5, va='top', color='#526473')

        primary = rows('primary_31_pairs.csv')
        summary = manifest['primary_summary']
        fig, (paired, delta) = plt.subplots(1, 2, figsize=(11.5, 4.9), gridspec_kw={'width_ratios': [1, 1.65]})
        fig.subplots_adjust(left=.08, right=.97, bottom=.17, top=.76, wspace=.42)
        heading(fig, 'A better first update: 6.45% lower expected error',
                '31 independent source preparations · same source data and nominal budgets for both selectors')
        for index, row in enumerate(primary):
            offset = (index-15)*.0035
            a, b = float(row['baseline_loss']), float(row['joint_loss'])
            paired.plot([offset, 1+offset], [a, b], color=GRID, lw=1, zorder=1)
            paired.scatter(offset, a, color=BLUE, marker='o', s=18, alpha=.8, zorder=2,
                           label='Baseline' if index == 0 else None)
            paired.scatter(1+offset, b, color=ORANGE, marker='^', s=27, alpha=.9, zorder=3,
                           label='Joint selector (ours)' if index == 0 else None)
        paired.set(xlim=(-.3, 1.3), ylim=(.63, .86), ylabel='Expected squared coefficient error ↓')
        paired.set_xticks([0, 1], ['Baseline selector\nmean 0.7548', 'Joint selector\nmean 0.7061'])
        paired.tick_params(axis='x', length=0)
        paired.grid(axis='y', color=GRID, alpha=.6)
        paired.legend(loc='upper left', frameon=False, fontsize=9)
        differences = [float(row['improvement_baseline_minus_joint']) for row in primary]
        low, high = summary['approximate_95_interval']
        delta.axhspan(low, high, color=ORANGE, alpha=.15, label='Approximate 95% CI for mean')
        delta.axhline(summary['paired_improvement'], color=ORANGE, lw=1.8, label='Mean paired improvement')
        delta.axhline(0, color=INK, lw=1)
        delta.scatter(range(1, 32), differences, color=ORANGE, marker='^', s=32, zorder=3)
        delta.set(xlim=(0, 32), ylim=(-.004, .108), xlabel='Independent preparation',
                  ylabel='Error reduction (baseline − joint) ↑')
        delta.set_xticks([1, 10, 20, 31])
        delta.legend(loc='upper right', frameon=False, fontsize=9)
        delta.grid(axis='y', color=GRID, alpha=.5)
        fig.text(.56, .018, 'Mean reduction 0.0487 · approximate 95% CI [0.0425, 0.0549]', ha='center', fontsize=10)
        save(fig, 'primary-improvement')

        methods = {row['method_id']: row for row in rows('benchmark_method_summary.csv')}
        fig, axis = plt.subplots(figsize=(11.5, 6.7))
        fig.subplots_adjust(left=.095, right=.97, top=.80, bottom=.20)
        heading(fig, 'Error and cost: lower-left is better',
                'One fixed prepared cohort · 2,048 shared tasks · pipeline means over 14 balanced timing blocks')
        for name, row in methods.items():
            cost = float(row['warm_pipeline_mean_ms']) / 1000
            error = float(row['empirical_mean_of_2048_losses'])
            if name == 'B04':
                axis.scatter(cost, error, s=125, color=ORANGE, marker='^',
                             edgecolors='white', linewidth=.7, zorder=8)
            else:
                axis.scatter(cost, error, s=65, color=BLUE, marker='o', zorder=5)

        annotations = {
            'B01': ('Isotropic (B01)', (18, -3), 'left'),
            'B02': ('Zero-head (B02)', (-5, 22), 'left'),
            'B03': ('Floor branch (B03)', (160, -11), 'left'),
            'B04': ('Joint frontier (ours, B04)', (34, -40), 'left'),
            'B11:F': ('Fitted floor (B11:F)', (95, 25), 'left'),
            'B11:Z': ('Fitted zero-head (B11:Z)', (40, -16), 'left'),
            'B08:ridge': ('Ridge regression', (-12, 18), 'right'),
            'B10:empirical_bayes': ('Empirical Bayes', (145, 8), 'left'),
        }
        for name, (label, offset, align) in annotations.items():
            row = methods[name]
            cost = float(row['warm_pipeline_mean_ms']) / 1000
            error = float(row['empirical_mean_of_2048_losses'])
            axis.annotate(f'{label}\n{error:.3f} error · {cost:.2f} s',
                          xy=(cost, error), xytext=offset, textcoords='offset points',
                          ha=align, va='center', fontsize=10,
                          color=ORANGE if name == 'B04' else INK,
                          arrowprops={'arrowstyle': '-', 'color': '#8497a5', 'lw': .9},
                          bbox={'boxstyle': 'round,pad=.25', 'facecolor': 'white',
                                'edgecolor': 'none', 'alpha': .94})
        axis.set(xlim=(0, 7.1), ylim=(.66, .81),
                 xlabel='Warm full-pipeline time (seconds per repetition; lower is better)',
                 ylabel='Mean squared coefficient error (lower is better)')
        axis.set_xticks(range(8))
        axis.set_yticks([.675, .700, .725, .750, .775, .800])
        axis.grid(color=GRID, alpha=.65)
        axis.set_axisbelow(True)
        axis.scatter([], [], s=125, color=ORANGE, marker='^', label='Our method: joint frontier (B04)')
        axis.scatter([], [], s=65, color=BLUE, marker='o', label='All other methods')
        axis.legend(loc='lower right', frameon=False, fontsize=10, labelspacing=.8)
        fig.text(.095, .065,
                 'All 14 methods are plotted; some points overlap. Times include fitting, adaptation and evaluation.\n'
                 'Warm B11:Z uses 2 repetitions; others use 1. This is full-pipeline time, not prediction latency.',
                 fontsize=10)
        save(fig, 'error-cost-tradeoff')
    print(f'Wrote two SVG/PNG figure pairs to {args.output}')


if __name__ == '__main__':
    main()
