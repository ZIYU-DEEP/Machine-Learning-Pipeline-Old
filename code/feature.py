"""
Title: feature.py
Description: Conduct feature engineering.
Author: Yeol Ye, University of Chicago
"""

import re

import pandas as pd
import numpy as np

from sklearn.preprocessing import OrdinalEncoder


# *******************************************************************
# Cut the Outliers
# *******************************************************************
def cut_outliers(data, col_name):
    """
    Filter out the outliers in a column of the data.

    Inputs:
        data: (DataFrame) the full data
        col_name: (string) the name of the column in need of cut outliers

    Returns:
        data: (DataFrame) the updated full data
    """
    col = data[col_name]
    data = data[np.abs(col-col.mean()) <= (3 * col.std())]
    return data


# *******************************************************************
# Discretize Continuous Features
# *******************************************************************
def discretize(data, bins, list_to_discrete):
    """
    Discretize the value of certain columns (specified in list_to_discrete)
    in the full data. Note that this function transforms data in place.

    Inputs:
        data: (DataFrame) the full data
        bins: (int) the bins to discretize the column. For example, if bins
            is 3, then the values of the data are cut into 3 ranges in order,
            and are transformed to 0, 1 and 2 according to its original value.
        list_to_discrete: (list) the list storing the name of columns to be
            discretized.

    Returns:
        None
    """
    for col_name in list_to_discrete:
        data[col_name] = pd.cut(data[col_name], bins, right=True,
                                precision=4).cat.codes


# *******************************************************************
# Convert the Categorical to Dummies
# *******************************************************************
def one_hot_encoding(data, list_to_dummy):
    """
    Convert categorical columns into dummy/indicator columns and drop the
    original categorical columns.

    Inputs:
        data: (DataFrame) the full data
        list_to_discrete: (list) the list storing the name of columns to be
            one-hot encoded.

    Returns:
        data: (DataFrame) the transformed full data
    """
    for col_name in list_to_dummy:
        df_dummy = pd.get_dummies(data[col_name], prefix=col_name)
        data = pd.concat([data, df_dummy], axis=1).drop(col_name, axis=1)
    return data


def one_hot_encoding_all(data):
    """
    Convert categorical columns into dummy/indicator columns and drop the
    original categorical columns.

    Inputs:
        data: (DataFrame) the full data

    Returns:
        data: (DataFrame) the transformed full data
    """
    list_to_dummy = [i for i in data.select_dtypes(include=['object']).columns]

    for col_name in list_to_dummy:
        df_dummy = pd.get_dummies(data[col_name], prefix=col_name)
        data = pd.concat([data, df_dummy], axis=1).drop(col_name, axis=1)
    return data


# *******************************************************************
# Convert N/A Values to Zero
# *******************************************************************
def fill_null(data):
    """
    Convert the N/A values in columns into zeros.

    Inputs:
        data: (DataFrame) the full data

    Returns:
        data: (DataFrame) the transformed full data
    """
    col_with_null = list(data.columns[data.isnull().any()])
    for i in col_with_null:
        data[i] = data[i].fillna(0)
    return data

def temporal_split(data, var, test_length, gap):
    """
    Split the data into several train and test pairs with fixed-length test sets
    and variant-length training set. Training sets are all the observations up
    till the test sets plus gap (to give the results enough time to show up).
    The first training set is of the same length as the test sets.

    Inputs:
        - data (DataFrame) data matrix to be split it to training and test sets
        - var (string): split in terms of the column
        - test_length (string): length of the test set (e.x. 6M, 10D)
        - gap (string): length of the gap between training and test set, and
            after test set until the end

    Returns:
        (list of tuples of DataFrames) each tuple is a training and test pair

    """
    gap_delta = pd.Timedelta(int(re.findall(r'[0-9]+', gap)[0]),
                             re.findall(r'[a-zA-Z]+', gap)[0])
    test_delta = pd.Timedelta(int(re.findall(r'[0-9]+', test_length)[0]),
                              re.findall(r'[a-zA-Z]+', test_length)[0])
    pairs = []

    training_end = (data[var].min() + test_delta).date()
    test_start = training_end + gap_delta
    test_end = test_start + test_delta
    max_date = data[var].max().date() - gap_delta

    while test_end <= max_date:
        pairs.append((data[data[var] < training_end],
                      data[(data[var] >= test_start) & (
                                  data[var] < test_end)]))

        test_start = test_end
        test_end += test_delta
        training_end = test_start - gap_delta

    print(("Using a test length of '%s', with a '%s' gap between "
           "training and test sets, and at the end of the data set "
           "observations of the last '%s' cannot be used for either "
           "training or testing.") % (test_length, gap, gap))
    print(("Print the start and end date for each of the %s pairs of "
           "training and test sets to verify the split works.\n") %
           len(pairs))
    for i, (train, test) in enumerate(pairs):
        messages = ["<TRAINING-TEST PAIR %s>" % i,
                    "<TRAINING SET TIME RANGE> %s - %s" %
                    (train[var].min().date(), train[var].max().date()),
                    "<TEST SET TIME RANGE> %s - %s" %
                    (test[var].min().date(), test[var].max().date())]

        for message in messages:
            print(message)
        print("\n\n")

    return pairs


def con_fill_na(data, fill_con):
    """
    Take the continuous variables, impute the missing features with column
    medians.

    Returns:
        (self) pipeline with missing values in the numerical columns imputed

    """
    print("\n\nStart to impute missing values continuous variables:")

    for var in fill_con:
        imputed = data[var].median()
        data[var] = data[var].fillna(imputed)

        print(("\tMissing values in '%s' imputed with column median %4.3f.") %
              (var, imputed))

    return data

def str_fill_na(data, to_fill_obj):
    """
    Fill in missing data with desired string entry.

    Returns:
        (self) pipeline with missing values in the object columns filled.

    """
    print("\n\nStart to fill in missing values:")

    for var, fill in to_fill_obj.items():
        # if no value is provided to fill in, use the most frequent one
        if fill is None:
            fill = data[var].mode()[0]
        data[var].fillna(value=fill, inplace=True)

        print("\tFilled missing values in '%s' with '%s'." % (var, fill))

    return data


def to_combine(data, to_combine):
    """
    Combine some unnecessary levels of multinomials.

    Returns:
        (self) pipeline with less frequent levels in the multinomial columns
            combined.

    """
    print("\n\nStart to combine unnecessary levels of multinomials.")

    for var, dict_combine in to_combine.items():
        if not dict_combine:
            dict_combine = {"YES": [val for val in data[var].unique()
                                    if val != "NO"]}

        for combined, lst_combine in dict_combine.items():
            if not lst_combine:
                freqs = data[var].value_counts(normalize=True)
                lst_combine = freqs[freqs < 0.05].index
            data.loc[data[var].isin(lst_combine), var] = combined

        print("\tCombinations of levels on '%s'." % var)

    return data


def to_binary(data, to_binary):
    """
    Transform variables to binaries.

    Returns:
        (self) pipeline with chosen columns transformed to binaries.

    """
    print(("\n\nFinished transforming the following variables: %s to "
           "binaries.") % (list(to_binary.keys())))

    for var, cats in to_binary.items():
        enc = OrdinalEncoder(categories=cats)
        data[var] = enc.fit_transform(np.array(data[var]).reshape(-1, 1))

    return data


def one_hot(data, to_one_hot):
    """
    Creates binary/dummy variables from multinomials, drops the original
    and inserts the dummies back.

    Returns:
        (self) pipeline with one-hot-encoding applied on categorical vars.

    """
    print(("\n\nFinished applying one-hot-encoding to the following "
           "categorical variables: %s\n\n") % to_one_hot)

    for var in to_one_hot:
        dummies = pd.get_dummies(data[var], prefix=var)
        data.drop(var, axis=1, inplace=True)
        data = pd.concat([data, dummies], axis=1)

    return data


def compare_train_test(data, train_features):
    """
    Compare the features in the training and test set after preprocessing.
    For those in the training set but not the test set, insert a column with
    all zeros at the same column index in the test set. For those in the
    test set but are not in the training set, drop them from the test set.

    """
    test_features = data.columns

    to_drop = [var for var in test_features if var not in train_features]
    data.drop(to_drop, axis=1, inplace=True)
    print(("\n\n%s are in the test set but are not in the training "
           "set, dropped from the test set.") % to_drop)

    to_add = [(i, var) for (i, var) in enumerate(train_features)
              if var not in test_features]
    print(("Start to add those are in the training set but not the "
           "test set to the test set:"))
    for i, var in to_add:
        data.insert(loc=i, column=var, value=0)
        print("\t'%s' added to the %sth column of the test set with all zeros."
              % (var, i))

    return data
