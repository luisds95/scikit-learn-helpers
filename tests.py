import unittest
import numpy as np
import pandas as pd
import datetime as dt
import feature_selection
import panel_data


class testHelpers(unittest.TestCase):

    def setUp(self) -> None:
        n_years = 6
        i_year = 2000
        f_year = i_year + n_years
        np.random.seed(123)
        v1 = np.random.normal(5, 10, n_years)
        v2 = np.random.normal(-5, 10, n_years)
        self.df1 = pd.DataFrame({'date': [i for i in range(i_year, f_year)], 'v1': v1})
        self.df2 = pd.DataFrame({'date': [i for i in range(i_year, f_year)], 'v1': v1, 'v2': v2})
        self.df3 = pd.DataFrame({'date': [dt.datetime(i, 5, 20) for i in range(i_year, f_year)], 'v1': v1, 'v2': v2})
        self.dfs = [self.df1, self.df2, self.df3]

    def test_standardize_input_format(self):
        for df in self.dfs:
            ndf, X, y, X_test, y_test = panel_data.standardize_input_format(df=df, X='v1', y='date')
            self.assertTrue(X.equals(X_test))
            self.assertTrue(y.equals(y_test))
            for i in [ndf, X, y, X_test, y_test]:
                self.assertTrue(isinstance(i, pd.DataFrame) or isinstance(i, pd.Series))

        for df in self.dfs:
            ndf, X, y, X_test, y_test = panel_data.standardize_input_format(X=df['v1'], y=df['date'])
            self.assertTrue(X.equals(X_test))
            self.assertTrue(y.equals(y_test))
            for i in [ndf, X, y, X_test, y_test]:
                self.assertTrue(isinstance(i, pd.DataFrame) or isinstance(i, pd.Series))

        ndf, X, y, X_test, y_test = panel_data.standardize_input_format(df=self.df1, X='v1', y='date', df_test=self.df2)
        for i in [ndf, X, y, X_test, y_test]:
            self.assertTrue(isinstance(i, pd.DataFrame) or isinstance(i, pd.Series))

        ndf, X, y, X_test, y_test = panel_data.standardize_input_format(df=self.df1, X='v1', y='date',
                                                                        X_test=self.df2['v1'], y_test=self.df2['date'])
        for i in [ndf, X, y, X_test, y_test]:
            self.assertTrue(isinstance(i, pd.DataFrame) or isinstance(i, pd.Series))

    def test_time_splitter(self):
        split = list(panel_data.time_splitter(self.df1, 'date', n=2, df_test=None))
        self.assertTrue(len(split) == 2)
        self.assertEqual(len(split[0][0]), self.df1.shape[0])
        self.assertTrue(split[0][0].sum() == 2)
        self.assertTrue(split[0][1].sum() == 2)
        self.assertLess(self.df1[split[0][0]]['date'].max(), self.df1[split[0][1]]['date'].min(),
                        'Train test overlaps with test set')
        self.assertLess(self.df1[split[1][0]]['date'].max(), self.df1[split[1][1]]['date'].min(),
                        'Train test overlaps with test set')


if __name__ == '__main__':
    unittest.main()
