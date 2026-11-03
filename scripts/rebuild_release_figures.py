"""Render the README comparisons from the committed, verified CSV exports."""
import argparse
import os
from pathlib import Path
import tempfile

from verify_public_results import rows, verify

LABELS = {
    'B01': 'Isotropic (B01)',
    'B02': 'Zero-head branch (B02)',
    'B03': 'Floor branch (B03)',
    'B04': 'Joint F/Z (B04)',
    'B11:F': 'Fitted floor branch (B11:F)',
    'B11:Z': 'Fitted zero-head branch (B11:Z)',
    'B08:ridge': 'Ridge regression',
    'B10:empirical_bayes': 'Empirical Bayes',
}
INK, BLUE, TEAL, ORANGE, GRID = '#142c3e', '#376799', '#087f70', '#bf5f28', '#dfe5e9'


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
        from matplotlib.ticker import MultipleLocator

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
            paired.scatter([offset, 1+offset], [a, b], c=[BLUE, TEAL], s=18, alpha=.8, zorder=2)
        paired.set(xlim=(-.3, 1.3), ylim=(.63, .86), ylabel='Expected squared coefficient error ↓')
        paired.set_xticks([0, 1], ['Baseline selector\nmean 0.7548', 'Joint selector\nmean 0.7061'])
        paired.tick_params(axis='x', length=0)
        paired.grid(axis='y', color=GRID, alpha=.6)
        differences = [float(row['improvement_baseline_minus_joint']) for row in primary]
        low, high = summary['approximate_95_interval']
        delta.axhspan(low, high, color=TEAL, alpha=.15, label='Approximate 95% CI for mean')
        delta.axhline(summary['paired_improvement'], color=TEAL, lw=1.8, label='Mean paired improvement')
        delta.axhline(0, color=INK, lw=1)
        delta.scatter(range(1, 32), differences, color=TEAL, s=24, zorder=3)
        delta.set(xlim=(0, 32), ylim=(-.004, .108), xlabel='Independent preparation',
                  ylabel='Error reduction (baseline − joint) ↑')
        delta.set_xticks([1, 10, 20, 31])
        delta.legend(loc='upper right', frameon=False, fontsize=9)
        delta.grid(axis='y', color=GRID, alpha=.5)
        fig.text(.56, .018, 'Mean reduction 0.0487 · approximate 95% CI [0.0425, 0.0549]', ha='center', fontsize=10)
        save(fig, 'primary-improvement')

        methods = {row['method_id']: row for row in rows('benchmark_method_summary.csv')}
        ordered = sorted(LABELS, key=lambda name: float(methods[name]['empirical_mean_of_2048_losses']))
        fig, axis = plt.subplots(figsize=(11.5, 5.6))
        fig.subplots_adjust(left=.31, right=.94, top=.76, bottom=.25)
        heading(fig, 'Empirical Bayes has the lowest error in this cohort',
                'One fixed prepared cohort · 2,048 shared tasks · observed mean squared coefficient error')
        for y, name in enumerate(ordered):
            value = float(methods[name]['empirical_mean_of_2048_losses'])
            has_direct = 'B05:'+name in methods
            axis.hlines(y, 0, value, color=GRID, lw=1)
            axis.scatter(value, y-.08 if has_direct else y, s=48,
                         color=TEAL if has_direct else ORANGE,
                         marker='o' if has_direct else 's', zorder=3)
            if has_direct:
                direct = float(methods['B05:'+name]['empirical_mean_of_2048_losses'])
                axis.scatter(direct, y+.10, s=48, facecolors='white', edgecolors=BLUE, linewidth=1.5, zorder=4)
            axis.text(value+.018, y, f'{value:.3f}', va='center', fontsize=10)
        axis.set_yticks(range(len(ordered)), [LABELS[name] for name in ordered])
        axis.set(xlim=(0, .90), ylim=(7.6, -.6), xlabel='Mean squared coefficient error (lower is better)')
        axis.tick_params(axis='y', length=0, pad=10)
        axis.xaxis.set_major_locator(MultipleLocator(.2))
        axis.grid(axis='x', color=GRID, alpha=.6)
        fig.text(.31, .045, 'Filled circles: networks · open circles: matching direct operators (B05)\nSquares: ridge / empirical Bayes. Direct controls have the same horizontal position.', fontsize=10)
        save(fig, 'method-comparison')

        order = []
        for name in ('B01', 'B02', 'B03', 'B04', 'B11:F', 'B11:Z'):
            order.extend([name, 'B05:'+name])
        order.extend(['B10:empirical_bayes', 'B08:ridge'])
        fig, axis = plt.subplots(figsize=(11.5, 7.3))
        fig.subplots_adjust(left=.31, right=.95, top=.81, bottom=.19)
        heading(fig, 'Pipeline cost depends on the method',
                '14 balanced timing blocks · seconds per repetition · includes fitting and the declared evaluation workload')
        labels = []
        for y, name in enumerate(order):
            cold = float(methods[name]['cold_pipeline_mean_ms'])/1000
            warm = float(methods[name]['warm_pipeline_mean_ms'])/1000
            axis.barh(y-.14, cold, height=.25, color=BLUE, label='Cold' if y == 0 else None)
            axis.barh(y+.14, warm, height=.25, color=TEAL, label='Warm' if y == 0 else None)
            axis.text(max(cold, warm)+.07, y, f'{warm:.3f}', va='center', fontsize=9)
            labels.append('Direct '+name[4:] if name.startswith('B05:') else LABELS[name])
        axis.set_yticks(range(len(order)), labels, fontsize=10)
        axis.set(xlim=(0, 7), ylim=(13.7, -.7), xlabel='Full-pipeline seconds per repetition (lower is better)')
        axis.tick_params(axis='y', length=0, pad=10)
        axis.set_axisbelow(True)
        axis.grid(axis='x', color=GRID, alpha=.6)
        axis.legend(loc='upper right', frameon=False, ncols=2)
        fig.text(.31, .035, 'Labels show warm means. These are full-pipeline measurements, not prediction latency.\nWarm B11:Z uses 2 repetitions; all other methods use 1. No energy was measured.', fontsize=10)
        save(fig, 'pipeline-cost')
    print(f'Wrote three SVG/PNG figure pairs to {args.output}')


if __name__ == '__main__':
    main()
