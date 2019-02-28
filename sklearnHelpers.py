"""
Set of functions and classes which are seek to perform some routines that
aren't built into scikit-learn.

Creator: Luis Da Silva.
Last update: 16/01/2019.
"""

# coding: utf-8
import numpy as np
import pandas as pd
import warnings, math
from datetime import datetime as dt
from functools import reduce
from itertools import combinations
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score


def n_combinations(n, r):
    return math.factorial(n)/(math.factorial(r)*math.factorial(n-r))


def best_subset_selection(X, y, model, cv=5, scoring=None, verbose=1):
    """
    Helper function to perform best subset selection (tests every possible model)
    """
    # Initializing
    all_features = list(X.columns)
    n_features = len(all_features)
    if verbose > 0:
        n_comb = reduce(lambda a,b : a+b, [n_combinations(n_features, r)
                                           for r in range(1, n_features+1)])
        init_time = dt.now()
        print("Initializing best subset selection process...")

    all_first = True

    for i in range(1, n_features + 1):
        comb_iterator = combinations(all_features, i)
        for comb in comb_iterator:
            subset = list(comb)
            score = np.mean(cross_val_score(model, X[subset], y, cv=cv, scoring=scoring))
            if all_first:
                all_first = False
                best_score = score
                best_subset = subset
            elif score > best_score:
                best_score = score
                best_subset = subset

        if verbose > 0 and (i-1) % verbose == 0:
            n_comb_so_far = reduce(lambda a,b : a+b, [n_combinations(n_features, r)
                                               for r in range(1,i+1)])
            perc = n_comb_so_far/n_comb
            time_diff = dt.now() - init_time
            time_remaining = time_diff * (1/perc - 1)
            print("{:.2f}% completed so far. Best score: {:.4f}.".format(perc*100, best_score))
            print("Time spent: {}. Time remaining: {}.".format(time_diff, time_remaining))

    return best_score, best_subset


def fw_cv_selection(X, y, model, cv=5, verbose=True, stopping=None, scoring=None):
    """
    Helper function to perform forward selection
    """
    if verbose:
        print("Initializing forward selection process...")
        itime = dt.now()
    
    all_features = list(X.columns)
    selected_features = []
    all_first = True
    no_update = 0
    
    for r in range(len(all_features)):
        rtime = dt.now()
        untested_features = [f for f in all_features if f not in selected_features]
        first = True

        for f1 in untested_features:
            features = selected_features + [f1]
            score = np.mean(cross_val_score(model, X[features], y, cv=cv, scoring=scoring))
            if first:
                first = False
                best_score_this_round = score
                best_feature_this_round = f1
                if all_first:
                    all_first = False
                    best_score = score
                    best_features = features
            elif score > best_score_this_round:
                best_score_this_round = score
                best_feature_this_round = f1
                if score > best_score:
                    best_features = features
                    best_score = score
                    no_update = -1
        no_update += 1
        
        # Stop searching if has been A rounds without updating
        if stopping is not None and no_update >= stopping:
            if verbose:
                print('Searching stopped on round {} after {} rounds without updating best subset.'.format(r+1,stopping))
            break
        
        if verbose:
            time_dif = dt.now() - itime
            r_dif = dt.now() - rtime
            t_pred = time_dif / (r+1) * len(all_features)
            print("Round {} completed in {} ({} total, {} predicted).\n{} features selected so far with score {}.".\
                  format(r+1, r_dif, time_dif, t_pred, len(best_features),best_score))

        selected_features.append(best_feature_this_round)

    return best_score, best_features


def fill_na_array(base, fill):
    """
    Fills NaN values with predictions
    """
    return pd.DataFrame(base).fillna(pd.DataFrame(fill))


def tune_fit_model(X, y, Model, param_grid=None, best_subset = False, forward_selection=False,
             predictDB = None, self_predict = False, scaling=None,
             scoring=None, cv=5, verbose=True, stopping=None):
    """
    Function to tune and fit a model at once by using cross validation.

    :param X: Explanatory features.
    :param y: Target variable.
    :param Model: Algorithm.
    :param param_grid: Dictionary with hyper parameters to test.
    :param best_subset: Whether to perform best subset selection or not.
    :param forward_selection: Whether to perform forward_selection or not.
    :param predictDB: Database on which make predictions.
    :param self_predict: Predict on X.
    :param scaling: List of variables to scale.
    :param scoring: Scoring method.
    :param cv: Number of folds to be used in cross-validation.
    :param verbose: if verbosity or not.
    :param stopping: number of rounds to keep going without score improving.
    :return: results dictionary.
    """
    
    # Scaling vars
    if scaling is not None:
        for col in scaling:
            X[col] = (X[col]-X[col].mean())/X[col].std()
        if predictDB is not None:
            for col in scaling:
                predictDB[col] = (predictDB[col]-predictDB[col].mean())/predictDB[col].std()
    
    if callable(Model):
        model = Model()
        params = model.get_params()
    else:
        model = Model
        params = model.get_params()

    # Finding best subset
    
    if best_subset:
        best_score, best_subset = best_subset_selection(X, y, model, scoring=scoring)
    elif forward_selection:
        best_score, best_subset = fw_cv_selection(X, y, model, verbose=verbose,
                                                  stopping=stopping, scoring=scoring)
    else:
        best_score = np.nan
        best_subset = list(X.columns)

    # Tuning parameters
    if param_grid is not None:
        tune_model = GridSearchCV(model, param_grid, scoring=scoring, cv=cv)
        tune_model.fit(X[best_subset], y)
        params = tune_model.best_params_
        model.set_params(**params)

    name = type(model).__name__
    results = {'name': name,
               'parameters': params,
               'score': best_score,
               'subset': best_subset}
    
    # Printing results
    if verbose:
        print("Model " + name + ":")
        if best_subset:
            print("Score: {}".format(best_score))
            print("Subset: {}".format(best_subset))
        if params:
            print("Parameters: {}".format(params))
        print("-"*20)

    model.fit(X[best_subset], y)
    results['model'] = model
    
    if self_predict:
        results['self_pred'] = model.predict(X[best_subset])
    
    # Make predictions on Held-out Dataset
    if predictDB is not None:
        results['predictions'] = model.predict(predictDB[best_subset])
    
    return results


def check_alpha_selection(algorithm, X, y, param, params, best_score,
                          best_param, precision=0.0001, multiplier=1000000,
                          cv=5, direction=1, change=10):
    """
    Test alpha value and look for a better one.

    :param algorithm: Un-called scikit-learn model that allows alpha
    :param X: X data frame
    :param y: target variable
    :param param: String name of the param to fit.
    :param params: Dictionary of params for model.
    :param cv: int. Number of folds for cross-validation.
    :param best_score: current best score
    :param best_param: current best value for parameter.
    :param multiplier: searching strength
    :param precision: decimal point precision
    :param direction: 1 or -1
    :param change: factor to calculate new multiplier if no better alpha is found
    :return: best_score, best_alpha, direction

    Note: Needs to be generalized and tested.
    """
    if multiplier < 1:
        # Stop condition
        return best_score, best_param, direction
    
    # Get new score (+)
    new_param = best_param + precision * multiplier * direction
    params[param] = new_param
    model = algorithm(**params)
    new_score = np.mean(cross_val_score(model, X, y, cv=cv))
    
    if new_score > best_score:
        # Better alpha found
        best_score = new_score
        best_param = new_param
        return best_score, best_param, direction
        
    else:
        # Try going the other way (-)
        new_param = best_param - precision * multiplier * direction
        params[param] = new_param
        model = algorithm(**params)
        new_score = np.mean(cross_val_score(model, X, y, cv=cv))
        if new_score > best_score:
            best_score = new_score
            best_param = new_param
            return best_score, best_param, -direction
        
        else:
            # No progress made, so change multiplier
            return check_alpha_selection(algorithm, X, y, cv, best_score,
                          best_param, multiplier/change, precision, 1)


def select_param(algorithm, X, y, param, params, precision=0.0001, multiplier=1000000,
                 cv=5, verbose=True, max_iters=1000):
    """
    Basic function intended to iteratively select the best value for alpha
    in those algorithms that use it as a parameter, such as LASSO and
    RIDGE regression.

    :param algorithm: Un-called scikit-learn model that allows alpha
    :param X: X Data Frame
    :param y: target variable
    :param precision: float decimal indicating how much to optimize
    :param multiplier: adjusting strength
    :param alpha: initial alpha level, int.
    :param cv: number of folds to be use in cross-validation.
    :param verbose: if verbosity or not.
    :return: new_alpha, new_score.

    Note: Needs generalization.
    """
    # Model
    model = algorithm(**params)
    
    # Initial score
    score = np.mean(cross_val_score(model, X, y, cv=cv))
    
    new_param = params[param]
    new_score = 0
    direction = 1
    improvement = precision + 1
    iters = 0
    
    while improvement > precision:
        if iters >= max_iters:
            print("Max iterations reached. Result may be unstable.")
            break

        old_score = new_score
        new_score, new_param, direction = check_alpha_selection(algorithm,
                          X, y, cv, score, new_param, multiplier, direction)
        improvement = new_score - old_score
        if verbose:
            print("Selected {}={} which improved performance in: {}".format(param, new_param,
                                                                            improvement))
    return new_param, new_score
