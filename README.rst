scikit-learn-helpers
====================

Helper functions for `scikit-learn <https://github.com/scikit-learn/scikit-learn>`_ which I found myself needing on a frequent basis while working on data science projects.

There are currently 5 main functions:

- ``tune_fit_model``: Fit a model, find best subset or forward selection subset and then tune hyperparameters. It also works if you don't need to find a subset or tune hyperparameters. Returns a dictionary with results.
- ``best_subset_selection``: Performs best subset selection (tests every possible model) and scores with cross validation. Returns both the best score and subset.
- ``fw_cv_selection``: Performs forwards subset selection with cross validation. Returns both the best score and subset.
- ``fill_na_array``: Fill missing values on an array with another array. Returns pandas dataframe.
- ``select_alpha``: Basic function intended to iteratively select the best value for alpha in those algorithms that use it as a parameter, such as LASSO and RIDGE regression.