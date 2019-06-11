"""
Time series or Panel Data Cross Validation helper functions.

Creator: Luis Da Silva.
luisds95.github.io
Last update: 11/06/2019.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def standardize_input_format(df=None, X=None, y=None, df_test=None, X_test=None, y_test=None):
    # Helper function to standardize inputs
    if df is not None:
        # X and y as list/string must be provided
        Xdf = df.loc[:, X]
        ydf = df.loc[:, y]
    else:
        # X and y as dataframe must be provided
        Xdf = X
        ydf = y
        df = pd.merge(X, y, right_index=True, left_index=True)

    if df_test is None:
        if X_test is None:
            X_test = Xdf
            y_test = ydf
        else:
            # X_test and y_test as dataframes must be provided
            pass
    else:
        if X_test is None:
            X_test = df_test.loc[:, X]
            y_test = df_test.loc[:, y]
        else:
            X_test = df_test.loc[:, X_test]
            y_test = df_test.loc[:, y_test]

    return df, Xdf, ydf, X_test, y_test


def time_splitter(df, time_var, n=5, df_test=None):
    """
    Yields masks for train and test data.
    :param df: Pandas DataFrame
    :param time_var: String
    :param n: Number of splits to test on. n train and n test datas are going to be yielded
    :param df_test: Optional. Use if the test set comes from a different dataset.
    :return: train and test mask.
    """
    if df_test is None:
        df_test = df

    time_min = df[time_var].min()
    interval = (df[time_var].max() - time_min) / (n + 1)

    for i in range(1, n + 1):
        train_offset = time_min + interval * i
        train_mask = df[time_var] <= train_offset
        test_offset = time_min + interval * (i + 1)
        test_mask = np.logical_and(df_test[time_var] >= train_offset, df_test[time_var] <= test_offset)
        yield train_mask, test_mask


def time_cv(model, time, df=None, X=None, y=None, n=5, df_test=None, X_test=None, y_test=None,
            scorers=[accuracy_score, f1_score, confusion_matrix], **kwargs):
    """
    Performs time series cross validation with times series split. A time variable is required.
    :param model: Model class with fit method.
    :param time: Time variable, string
    :param df: Principal pandas dataframe.
    :param X: Either a list of strings (if df is provided) or a dataframe.
    :param y: Either a strings (if df is provided) or a dataframe.
    :param n: int. Number of splits.
    :param df_test: Optional. Use if test comes from a different dataframe.
    :param X_test: Optional. Use if test comes from a different dataframe. Either a list of strings (if df is provided) or a dataframe.
    :param y_test: Optional. Use if test comes from a different dataframe. Either a strings (if df is provided) or a dataframe.
    :param scorers: list of scorers.
    :return: list of scores.
    """
    score = [[] for _ in scorers]
    df, X, y, X_test, y_test = standardize_input_format(df, X, y, df_test, X_test, y_test)

    for train, test in time_splitter(df, time, n, df_test):
        model.fit(X.loc[train], y.loc[train], **kwargs)
        preds = model.predict(X_test.loc[test]).round(0)

        for i in range(len(scorers)):
            score[i].append(scorers[i](y_test.loc[test], preds))

    return score


def time_search_cv(model, param_grid, time, df=None, X=None, y=None, n=5, df_test=None, X_test=None, y_test=None,
                   scorer=f1_score, verbose=True):
    """
    Performs hyperparameter gridsearch with time data
    :param model: Model class with fit method.
    :param param_grid: Dictionary of hyperparameters
    :param time: Time variable, string
    :param df: Principal pandas dataframe.
    :param X: Either a list of strings (if df is provided) or a dataframe.
    :param y: Either a strings (if df is provided) or a dataframe.
    :param n: int. Number of splits.
    :param df_test: Optional. Use if test comes from a different dataframe.
    :param X_test: Optional. Use if test comes from a different dataframe. Either a list of strings (if df is provided) or a dataframe.
    :param y_test: Optional. Use if test comes from a different dataframe. Either a strings (if df is provided) or a dataframe.
    :param scorer: scorer class.
    :param verbose: Wheter or not to print result.
    :return: Dictionary of best parameters.
    """
    param_grid = ParameterGrid(param_grid)
    best_score = -np.inf
    best_params = {}
    for params in param_grid:
        model.set_params(**params)
        score = np.mean(time_cv(model=model, time=time, df=df, X=X, y=y, n=n, df_test=df_test, X_test=X_test,
                                y_test=y_test, scorers=[scorer]))
        if score > best_score:
            best_score = score
            best_params = params
    if verbose:
        print('Best score:', best_score, '\nBest params:', best_params)
    return best_params