"""Small regression checks for preprocessing and inference contracts."""
import unittest
import numpy as np
import pandas as pd
import model as m


class WorkflowChecks(unittest.TestCase):
    def test_income_boundaries_and_index(self):
        df = pd.DataFrame({'median_income': [.5, 1.5, 2.9, 4.5, 5.99, 6, 15]}, index=range(10, 17))
        result = m.income_categories(df)
        self.assertEqual(result.tolist(), [1, 1, 2, 3, 4, 4, 5])
        self.assertTrue(result.index.equals(df.index))

    def test_categorical_missing_and_unknown(self):
        X = pd.DataFrame({'ocean_proximity': ['INLAND', 'NEAR BAY', 'INLAND', None, np.nan]})
        p = m.categorical_pipeline()
        out = p.fit_transform(X).toarray()
        np.testing.assert_array_equal(out, [[1, 0], [0, 1], [1, 0], [1, 0], [1, 0]])
        np.testing.assert_array_equal(p.transform(pd.DataFrame({'ocean_proximity': ['NEW']})).toarray(), [[0, 0]])

    def test_ratio_features_preserve_input(self):
        df = pd.DataFrame({'total_rooms': [10.], 'total_bedrooms': [2.], 'population': [6.], 'households': [2.]})
        original = df.copy(deep=True)
        result = m.add_ratio_features(df)
        pd.testing.assert_frame_equal(df, original)
        self.assertEqual(result[['rooms_per_house', 'bedrooms_ratio', 'people_per_house']].iloc[0].tolist(), [5., .2, 3.])

    def test_rmse_and_bootstrap(self):
        self.assertEqual(m.rmse([1, 2], [1, 2]), 0.)
        self.assertEqual(m.bootstrap_rmse_ci([1, 2], [2, 3]), (1., 1.))


if __name__ == '__main__':
    unittest.main()
