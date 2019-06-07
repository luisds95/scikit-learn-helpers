Scikit-learn Helper Functions
=============================

This repository includes a set of procedures that are not directly encoded into [scikit-learn](https://github.com/scikit-learn/scikit-learn) and that I have found myself needing on a frequent basis while working on data science projects.

Model optimization with feature selection
------------------------------------------
- ``feature_selection.tune_fit_model``: An all in one suite. Fits a model, finds the best subset or forward selection subset and then tunes its hyperparameters. It also works if you don't need to find a subset or tune hyperparameters. Returns a dictionary with results.
- ``feature_selection.best_subset_selection``: Performs **best subset selection** (tests every possible model) and scores with cross validation. Returns both the best score and subset.
- ``feature_selection.fw_cv_selection``: Performs **forward subset selection** with cross validation. Returns both the best score and subset.

Time series and Panel Data
--------------------------
These functions are different from the ones provided by sklearn in the sense that they use a time variable to split the data.
- ``panel_data.time_cv``: Performs **time series cross validation**.
- ``panel_data.time_search_cv``: Performs **hyperparameter grid search** on time series or panel data.
- ``panel_data.time_splitter``: Yields a train and test mask.



More stuff
----------
- ``feature_selection.fill_na_array``: Fills missing values on an array with another array. Returns pandas dataframe.
- ``feature_selection.n_combinations``: Returns the number of possible combinations from taking ``r`` elements from an ``n`` sized array.