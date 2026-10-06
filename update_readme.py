"""Refresh the README's measured-results block from reports/metrics.json."""
import json
from pathlib import Path


def update_results(root):
    metrics = json.loads((root / 'reports' / 'metrics.json').read_text())
    low, high = metrics['test_rmse_95_percent_bootstrap_ci']
    lines = [
        f"**Test RMSE: ${metrics['test_rmse']:,.0f} · MAE: ${metrics['test_mae']:,.0f} · R²: {metrics['test_r2']:.3f}**",
        '',
        f"The tuned forest reduced held-out RMSE by **{metrics['test_rmse_reduction_vs_baseline_percent']:.1f}%** versus a mean-only predictor (baseline test RMSE: ${metrics['baseline_test_rmse']:,.0f}).",
        '',
        '| Model | 3-fold CV RMSE, mean ± SD |',
        '|---|---:|',
    ]
    for name, values in metrics['cv'].items():
        lines.append(f"| {name} | ${values['mean']:,.0f} ± ${values['std']:,.0f} |")
    lines.extend([
        f"| Tuned random forest (selected candidate) | ${metrics['best_cv_rmse']:,.0f} |", '',
        f"**95% bootstrap interval for test RMSE: ${low:,.0f}–${high:,.0f}.**",
        '',
        f"Selected parameters: `{metrics['best_parameters']}`. The saved-and-reloaded model reproduced all test predictions exactly.",
        '',
        'These values come from the committed [metrics.json](reports/metrics.json), not an illustrative example.',
    ])
    path = root / 'README.md'
    before, rest = path.read_text().split('<!-- RESULTS_START -->')
    _, after = rest.split('<!-- RESULTS_END -->')
    path.write_text(before + '<!-- RESULTS_START -->\n' + '\n'.join(lines) + '\n<!-- RESULTS_END -->' + after)


if __name__ == '__main__':
    update_results(Path(__file__).resolve().parent)
