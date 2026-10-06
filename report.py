"""Reproduce the portfolio figures and measured results: python report.py."""
import os
import tempfile
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'housing-matplotlib'))
import hashlib
import json
import platform
import tarfile
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import joblib
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import model as m

ROOT = Path(__file__).resolve().parent
FIG = ROOT / 'reports' / 'figures'


def save(name):
    plt.savefig(FIG / name, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    started = time.time()
    FIG.mkdir(parents=True, exist_ok=True)
    (ROOT / 'data').mkdir(exist_ok=True)
    (ROOT / 'artifacts').mkdir(exist_ok=True)
    csv = ROOT / 'data' / 'housing.csv'
    if not csv.exists():
        m.load_housing()  # Populate the original temp-directory cache.
        with tarfile.open(Path(tempfile.gettempdir()) / 'housing.tgz') as archive:
            csv.write_bytes(archive.extractfile('housing/housing.csv').read())
    housing = pd.read_csv(csv)
    assert housing.shape == (20640, 10)
    train, test = m.stratified_split(housing)
    X, y = m.split_features_labels(m.add_ratio_features(train))
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.titleweight': 'bold'})

    fig, ax = plt.subplots(figsize=(9, 7))
    points = ax.scatter(train.longitude, train.latitude, c=train.median_house_value / 1000,
                        s=8, alpha=.45, cmap='viridis', rasterized=True)
    fig.colorbar(points, ax=ax, label='Median house value ($ thousands)')
    ax.set(xlabel='Longitude', ylabel='Latitude', title='California housing: location and value\nTraining districts only')
    save('geography.png')

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col, label in zip(axes, ['median_house_value', 'median_income', 'total_rooms'],
                             ['Median house value ($)', 'Median income (scaled units)', 'Total rooms']):
        ax.hist(train[col], bins=45, color='#257d98', edgecolor='white', linewidth=.3)
        ax.set(xlabel=label, ylabel='Training districts')
        ax.ticklabel_format(axis='x', style='plain')
    fig.suptitle('Training-data distributions: capped values and skewed counts', fontweight='bold')
    fig.tight_layout()
    save('distributions.png')

    corr = m.explore_correlations(m.add_ratio_features(train)).sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(corr.index, corr.values, color=['#d47a50' if v < 0 else '#257d98' for v in corr])
    ax.axvline(0, color='#667788', linewidth=.8)
    ax.set(xlabel='Pearson correlation with median house value', title='Which features move with house values?\nTraining data; association does not establish causation')
    save('correlations.png')

    print('Cross-validating mean baseline, linear regression and random forest...', flush=True)
    cv = {}
    for name, estimator in [('Mean baseline', DummyRegressor()),
                            ('Linear regression', m.linear_model(m.build_preprocessing())),
                            ('Random forest', m.forest_model(m.build_preprocessing()))]:
        cv[name] = m.cross_val_rmse(estimator, X, y)
        print(name, cv[name], flush=True)
    print('Randomized search: 5 candidates x 3 folds...', flush=True)
    search = m.random_search(m.forest_model(m.build_preprocessing()), X, y)
    best = search.best_estimator_
    pd.DataFrame(search.cv_results_).to_csv(ROOT / 'reports' / 'search_results.csv', index=False)
    X_test, y_test = m.split_features_labels(m.add_ratio_features(test))
    pred = best.predict(X_test)
    ci = m.bootstrap_rmse_ci(y_test, pred)
    test_score = m.rmse(y_test, pred)
    baseline = DummyRegressor().fit(X, y)
    baseline_test = m.rmse(y_test, baseline.predict(X_test))
    print('Final test RMSE:', test_score, 'CI:', ci, flush=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    labels = list(cv) + ['Tuned forest*']
    values = [v['mean'] for v in cv.values()] + [-search.best_score_]
    errors = [v['std'] for v in cv.values()] + [search.cv_results_['std_test_score'][search.best_index_]]
    ax.bar(labels, np.array(values) / 1000, yerr=np.array(errors) / 1000, capsize=5,
           color=['#9aa9b5', '#458bb5', '#257d98', '#204b64'])
    ax.set(ylabel='Cross-validation RMSE ($ thousands)', title='Model comparison: lower error is better\nBars show 3-fold mean; error bars show fold standard deviation')
    fig.text(.13, .01, '*Best of 5 searched candidates; selection can make this CV score optimistic.', fontsize=9)
    save('model_comparison.png')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(y_test / 1000, pred / 1000, s=8, alpha=.3, color='#257d98')
    axes[0].plot([0, 520], [0, 520], '--', color='#d47a50')
    axes[0].set(xlabel='Actual ($ thousands)', ylabel='Predicted ($ thousands)', title='Held-out predictions')
    axes[1].scatter(pred / 1000, (pred - y_test) / 1000, s=8, alpha=.3, color='#257d98')
    axes[1].axhline(0, linestyle='--', color='#d47a50')
    axes[1].set(xlabel='Predicted ($ thousands)', ylabel='Predicted − actual ($ thousands)', title='Held-out residuals')
    fig.suptitle(f'Tuned forest | Test RMSE ${test_score:,.0f}', fontweight='bold')
    fig.tight_layout()
    save('test_diagnostics.png')

    names = best.steps[0][1].get_feature_names_out()
    importance = pd.Series(best.steps[-1][1].feature_importances_, index=names).sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    importance.tail(12).plot.barh(ax=ax, color='#257d98')
    ax.set(xlabel='Impurity-based importance', title='Top 12 transformed features\nModel-specific importance, not a causal effect')
    save('feature_importance.png')
    importance.sort_values(ascending=False).to_csv(ROOT / 'reports' / 'feature_importances.csv', header=['importance'])
    m.worst_errors(best, test, 10).to_csv(ROOT / 'reports' / 'worst_errors.csv', index_label='district_index')
    pd.DataFrame({'actual': y_test, 'predicted': pred, 'abs_error': np.abs(y_test - pred)}).to_csv(ROOT / 'reports' / 'test_predictions.csv', index_label='district_index')
    reloaded = m.save_and_reload(best, ROOT / 'artifacts' / 'housing_pipeline.pkl')
    np.testing.assert_allclose(pred, reloaded.predict(X_test), rtol=0, atol=0)
    raw_examples = test.drop(columns='median_house_value').head(3).to_dict('records')
    example_predictions = m.predict_new(reloaded, raw_examples)
    assert all(isinstance(value, float) for value in example_predictions)
    metrics = {'rows': len(housing), 'columns': len(housing.columns), 'train_rows': len(train), 'test_rows': len(test),
               'missing_values': housing.isna().sum().to_dict(), 'target_max': float(housing.median_house_value.max()),
               'target_at_max': int((housing.median_house_value == housing.median_house_value.max()).sum()),
               'cv': cv, 'best_cv_rmse': -search.best_score_, 'best_parameters': search.best_params_,
               'test_rmse': test_score, 'test_mae': mean_absolute_error(y_test, pred), 'test_r2': r2_score(y_test, pred),
               'test_rmse_95_percent_bootstrap_ci': ci, 'baseline_test_rmse': baseline_test,
               'test_rmse_reduction_vs_baseline_percent': 100 * (1 - test_score / baseline_test),
               'bootstrap_resamples': 200, 'search_candidates': 5, 'cv_folds': 3, 'random_state': 42,
               'n_estimators': 50, 'reloaded_predictions_identical': True,
               'dataset_sha256': hashlib.sha256(csv.read_bytes()).hexdigest(),
               'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
                            'scikit-learn': sklearn.__version__, 'matplotlib': matplotlib.__version__, 'joblib': joblib.__version__},
               'elapsed_seconds': round(time.time() - started, 2)}
    (ROOT / 'reports' / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    from update_readme import update_results
    update_results(ROOT)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    main()
