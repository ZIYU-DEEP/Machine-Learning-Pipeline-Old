"""
Title: model.py
Description: Conduct model training, testing and evaluation.
Author: Yeol Ye, University of Chicago
"""


import graphviz
import pandas as pd
import numpy as np

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, \
    BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import export_graphviz
from sklearn.metrics import *


# *******************************************************************
# Split the Data
# *******************************************************************
def split(features, target, random_state=0):
    """
    Split the data into training and testing set.

    Inputs:
        features: (DataFrame) the feature columns of the data
        target: (DataFrame) the target column of the data
        random_state: (int) the random state to be chosen

    Returns:
        Four DataFrame objects.
    """
    return train_test_split(features, target, random_state=random_state)


def time_train_test_split(data, colname, freq=None):
    """
    Creates temporal train/test splits. This function is adapted from
    https://github.com/rayidghani/magicloops.

    Inputs:
        data: (DataFrame) the data used for split
        time_colname: (string) the name of the column indicating time
        freq (string): the time gap for the testing, and it should be
            specified as '1M', '3M', etc.
    Returns:
        lst_train: (list) list of DataFrames used for training
        lst_test: (list) list of DataFrames used for testing
        lst_gap: (list )list of startpoints used for split
    """
    lst_gap = []
    lst_train = []
    lst_test = []
    lst_starts = pd.date_range(start=data[colname].min(),
                               end=data[colname].max(),
                               freq=freq)

    for i, start in enumerate(lst_starts[:-1]):
        cut = start + pd.DateOffset(1)
        train = data.loc[data[colname] <= start].drop(colname, axis=1)
        test = data.loc[(data[colname] > cut) &
                        (data[colname] < lst_starts[i + 1])]\
                   .drop(colname, axis=1)

        lst_train.append(train)
        lst_test.append(test)
        lst_gap.append(start)

    return lst_train, lst_test, lst_gap


# *******************************************************************
# Scale the Data
# *******************************************************************
def scale(X_train, X_test):
    """
    Split the training and testing data.

    Inputs:
        X_train: (DataFrame) the features used for training
        X_test: (DataFrame) the features used for testing

    Returns:
        DataFrame objects.
    """
    scaler = MinMaxScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test)


# *******************************************************************
# Build the Model
# *******************************************************************
def tree_classifier(X_train, y_train, max_depth=3):
    """
    Build the decisoin tree model.

    Inputs:
        X_train: (DataFrame) the features used for training
        X_test: (DataFrame) the features used for testing
        max_depth: (int) the max depth of the decision tree

    Returns:
        The DecisionTreeClassifier object
    """
    return DecisionTreeClassifier(max_depth=max_depth).fit(X_train, y_train)


# *******************************************************************
# Plot the Models
# *******************************************************************
def plot_decision_tree(clf, feature_names, target_name):
    """
    Plot the decision tree. This credits to the University of Michigan.
    This function requires the pydotplus module and assumes it's been
    installed.

    Inputs:
        clf: the model
        feature_names: (list) the list of strings to store feature names
        target_name: (string) the string of the target name

    Returns:
        None
    """
    export_graphviz(clf, out_file="adspy_temp.dot",
                    feature_names=feature_names,
                    class_names=target_name, filled=True, impurity=False)
    with open("adspy_temp.dot") as f:
        dot_graph = f.read()
    return graphviz.Source(dot_graph)


def plot_precision_recall(y_true, y_score, model_name):
    """
    Generates plots for precision and recall curve. This function is
    adapted from https://github.com/rayidghani/magicloops.

    Inputs:
        y_true: (Series) the Series of true target values
        y_score: (Series) the Series of scores for the model
        model_name: (string) the name of the model

    Returns:
        None
    """
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(
        y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)

    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score >= value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)

    pct_above_per_thresh = np.array(pct_above_per_thresh)

    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    ax1.set_ylim([0, 1])
    ax1.set_ylim([0, 1])
    ax2.set_xlim([0, 1])

    plt.title(model_name)
    plt.show()


# *******************************************************************
# Plot the Feature Importances
# *******************************************************************
def plot_feature_importances(clf, feature_names):
    """
    Plot the feature importances of the decision tree. This credit to the
    University of Michigan.

    Inputs:
        clf: the model
        feature_names: (list) the list of strings to store feature names

    Returns:
        None
    """
    c_features = len(feature_names)
    plt.barh(range(c_features), sorted(clf.feature_importances_), color='g',
             alpha=0.3)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature name")
    plt.yticks(np.arange(c_features), feature_names)


# *******************************************************************
# Generate the Precisions and Tables for Different Times
# *******************************************************************
def precision_at_k(y_true, y_scores, k):
    """
    Generates the precision for a model. This function is
    adapted from https://github.com/rayidghani/magicloops.

    Inputs:
        y_true: (Series) the true target value
        y_scores: (Series) the scores for the model
        k: (float) given threshold

    Returns:
        (Numpy Array) the calculated precisions
    """
    idx = np.argsort(np.array(y_scores))[::-1]
    y_scores, y_true = np.array(y_scores)[idx], np.array(y_true)[idx]
    cutoff_index = int(len(y_scores) * (k / 100.0))
    preds_at_k = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]

    return precision_score(y_true, preds_at_k)


def clfs_loop(models, clfs, grid, X_train, X_test, y_train, y_test,
              plot=False):
    """
    This function is adapted from https://github.com/rayidghani/magicloops.
    It will run each model and calculates corresponding AUC-ROC scores, then
    calculates precision at 1%, 5%, 10%, 20%, 30%, and 50%, and plots the
    precision-recall curve (optional).

    Inputs:
        models: (list) the list of names of models to be used
        clfs: (list) list of sklearn models
        grid: (dictionary) the grid of parameters
        X_train: (Series) the Series of features to train
        X_test: (Series) the Series of features to test
        y_train: (Series) the Series of targets to train
        y_test: (Series) the Series of targets to test
        plot: (bool) True if show plots, False otherwise

    Returns:
        results_df: (DataFrame) a table with results including performance metrics
    """
    results_df = pd.DataFrame(
        columns=('model_type', 'clf', 'parameters', 'auc-roc',
                 'p_at_1', 'p_at_2', 'p_at_5',
                 'p_at_10', 'p_at_20', 'p_at_30', 'p_at_50'))

    for i, clf in enumerate([clfs[m] for m in models]):
        print(models[i])
        params = grid[models[i]]
        for p in ParameterGrid(params):
            try:
                clf.set_params(**p)
                y_pred_probs = clf.fit(X_train, y_train).predict_proba(X_test)[
                               :, 1]
                y_pred_probs_sorted, y_test_sorted = \
                    zip(*sorted(zip(y_pred_probs, y_test), reverse=True))

                results_df.loc[len(results_df)] = \
                    [models[i], clf, p, roc_auc_score(y_test, y_pred_probs),
                     precision_at_k(y_test_sorted, y_pred_probs_sorted, 1.0),
                     precision_at_k(y_test_sorted, y_pred_probs_sorted, 2.0),
                     precision_at_k(y_test_sorted, y_pred_probs_sorted, 5.0),
                     precision_at_k(y_test_sorted, y_pred_probs_sorted, 10.0),
                     precision_at_k(y_test_sorted, y_pred_probs_sorted, 20.0),
                     precision_at_k(y_test_sorted, y_pred_probs_sorted, 30.0),
                     precision_at_k(y_test_sorted, y_pred_probs_sorted, 50.0)]

                if plot:
                    plot_precision_recall(y_test, y_pred_probs, clf)

            except IndexError as e:
                print('Error:', e)
                continue

    return results_df


def clfs_loop_temporal(train_sets, test_sets, target, models, clfs, grid,
                       plot=False):
    """
    This function is adapted from https://github.com/rayidghani/magicloops.

    Inputs:
        train_sets: (list of Series) the list of data to train across different time
        test_sets: (list of Series) the list of data to test across different time
        target: (string) the column name of the target
        models: (list) the name of models
        clfs: (list) list of sklearn models
        plot: (bool) True if show plots, False otherwise

    Returns:
        all_results: (list of DataFrames)
    """
    all_results = []

    for i in range(len(train_sets)):
        train = train_sets[i]
        test = test_sets[i]

        y_train = train[target]
        X_train = train.drop(target, axis=1)
        y_test = test[target]
        X_test = test.drop(target, axis=1)

        results = clfs_loop(models, clfs, grid, X_train, X_test, y_train,
                            y_test, plot)
        all_results.append(results)

    return all_results


# *******************************************************************
# Drop Time Columns
# *******************************************************************
def drop_time_col(train_sets, test_sets, time_colname):
    """
    Drop the column with the data type for time to avoid error in
    generating precision curves.

    Inputs:
        train_sets: (list) list of data for training
        test_sets: (list) list of data for testing
        time_colname: (string) the name of the column for time

    Returns:
        train_sets: (list) list of data for training
        test_sets: (list) list of data for testing
    """
    for i in range(len(train_sets)):
        train_sets[i] = train_sets[i].drop(time_colname, axis=1)
        test_sets[i] = test_sets[i].drop(time_colname, axis=1)
    return train_sets, test_sets
