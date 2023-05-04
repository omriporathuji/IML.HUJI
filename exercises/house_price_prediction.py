from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn, Optional
from datetime import datetime
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"

PRICE_COL = 'price'
COLS_TO_DROP = ['id', 'lat', 'long', 'date']
NON_NEG_COLS = ['price', 'floors', 'sqft_basement', 'yr_built', 'yr_renovated']
GT_0_COLS = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'sqft_living15', 'sqft_lot15']
RANGES = {'waterfront': range(2), 'condition': range(1, 6), 'grade': range(1, 14), 'view': range(5)}
CATEGORICAL_COLS = ['zipcode']
MIN_YEAR = 1900
MAX_BATH = 20
MAX_BED = 20
MAX_LOT = 1 * 10 ** 6
TEST_REPEAT = 10

train_columns = None
train_means = None


def division_(a, b):
    """
    Division function that handles zero division by returning 0
    """
    if b == 0.0:
        return 0.0
    return a / b


def train_preprocess(X: pd.DataFrame):
    """
    preprocess train data
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem
    Returns
    -------
    Post-processed train design matrix
    """
    global train_columns  # needed for the train and test columns to be the same after get-dummies
    global train_means  # needed for missing or outlier test values

    X.dropna(inplace=True)  # remove rows with NaNs
    X.drop_duplicates(inplace=True)
    X.drop(COLS_TO_DROP, inplace=True, axis=1)  # Remove the columns decided to have little impact or no relevance

    for col in NON_NEG_COLS:  # remove rows in which a column expected to non-negative has a negative value
        X = X[X[col] >= 0]
    for col in GT_0_COLS:  # remove rows in which a column expected to greater than 0 has a zero or negative value
        X = X[X[col] > 0]
    # Some columns should contain values in specific ranges, e.g. 1-5 (integers) - so we filter those that do not
    # conform to the expected range
    for col, r in RANGES.items():
        X = X[X[col].isin(r)]

    # Filter by min/max values of certain columns
    X = X[X['yr_built'] >= MIN_YEAR]
    X = X[X['sqft_lot'] <= MAX_LOT]
    X = X[X['bathrooms'] <= MAX_BATH]
    X = X[X['bedrooms'] <= MAX_BED]

    X = X.astype({'zipcode': int})  # Encode zip codes
    for col in CATEGORICAL_COLS:
        X = pd.get_dummies(X, prefix=col, columns=[col])

    train_columns = X.columns  # Store columns for later use in test
    train_means = X.mean(axis=0)  # Store mean values for later use in test
    return X


def test_preprocess(X: pd.DataFrame):
    """
    preprocess test data
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem
    Returns
    -------
    Post-processed test design matrix
    """
    X.drop(COLS_TO_DROP, inplace=True, axis=1)  # Remove the columns decided to have little impact or no relevance
    X.fillna(0)  # Replace NaN values with 0

    X = X.astype({'zipcode': int})
    for col in CATEGORICAL_COLS:  # Encode zip codes
        X = pd.get_dummies(X, prefix=col, columns=[col])

    X = X.reindex(columns=train_columns, fill_value=0)  # force test table to have same columns as train

    # Fill mean train values in certain columns with outlier values
    for col in NON_NEG_COLS:
        X.loc[df[col] < 0, col] = train_means[col]
    for col in GT_0_COLS:
        X.loc[df[col] <= 0, col] = train_means[col]
    for col, r in RANGES.items():
        X.loc[~X[col].isin(r), col] = train_means[col]

    # Fill mean train values in certain columns with outlier values
    X.loc[X['yr_built'] < MIN_YEAR, 'yr_built'] = train_means['yr_built']
    X.loc[X['sqft_lot'] > MAX_LOT, 'sqft_lot'] = train_means['sqft_lot']
    X.loc[X['bathrooms'] > MAX_BATH, 'bathrooms'] = train_means['bathrooms']
    X.loc[X['bedrooms'] > MAX_BED, 'bedrooms'] = train_means['bedrooms']

    return X


def preprocess_data(X: pd.DataFrame, y: Optional[pd.Series] = None):
    """
    preprocess data
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector corresponding given samples

    Returns
    -------
    Post-processed design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    # If we got a label vector and there is no price col in X, set y to be the price col
    if y is not None and PRICE_COL not in X.columns:
        X = X.merge(y.rename(PRICE_COL), left_index=True, right_index=True)

    if y is not None:
        X = train_preprocess(X)
    else:
        X = test_preprocess(X)

    # Add some ratio columns
    X['bathrooms_to_bedrooms'] = X.apply(lambda row: division_(row['bathrooms'], row['bedrooms']), axis=1)
    X['bedrooms_to_area'] = X.apply(lambda row: division_(row['bedrooms'], row['sqft_living']), axis=1)
    X['living_to_lot'] = X.apply(lambda row: division_(row['sqft_living'], row['sqft_lot']), axis=1)
    X['living_to_nearby'] = X.apply(lambda row: division_(row['sqft_living'], row['sqft_living15']), axis=1)
    X['lot_to_nearby'] = X.apply(lambda row: division_(row['sqft_lot'], row['sqft_lot15']), axis=1)
    X['new'] = np.where((X['yr_built'] >= np.percentile(X['yr_built'].unique(), 70)) |
                        (X['yr_renovated'] >= np.percentile(X['yr_renovated'].unique(), 70)), 1, 0)

    # Instead of using years, try to determine whether or not a house is new/very new
    X['very_new'] = np.where((X['yr_built'] >= np.percentile(X['yr_built'].unique(), 90)) |
                             (X['yr_renovated'] >= np.percentile(X['yr_renovated'].unique(), 90)), 1, 0)
    X = X.drop('yr_built', axis=1).drop('yr_renovated', axis=1)

    if y is not None:
        return X.drop(PRICE_COL, axis=1), X[PRICE_COL]
    if PRICE_COL not in X:
        return X
    return X.drop(PRICE_COL, axis=1)


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    std_y = np.std(y)

    for col in X.loc[:, ~X.columns.str.startswith('zipcode')]:  # iter over columns except for zipcode cols
        cov = np.cov(X[col], y)[0, 1]
        std_X = np.std(X[col])
        pc = cov / (std_X * std_y)
        fig = px.scatter(x=X[col], y=y, labels={'x': col, 'y': 'price'}, trendline='ols',
                         title="Correlation Between {} and House Price<br />Pearson Correlation: {}".format(col, pc))
        fig.write_html('{}/{}.html'.format(output_path, col))


def fit_gradually(train_X: pd.DataFrame, test_X: pd.DataFrame, train_y: pd.Series, test_y: pd.Series,
                  output_path: str = ".") -> NoReturn:
    """
    Gradually fit a linear regression over increasing percentages of the train dataset
    Parameters
    ----------
    train_X: DataFrame of shape (n_samples, n_features). Design matrix of regression problem on which to fit the model

    train_y: array-like of shape (n_samples, ), response vector to evaluate against

    test_X: DataFrame of shape (n_samples, n_features). Design matrix of regression problem on which to test loss

    test_y: array-like of shape (n_samples, ), response vector to evaluate loss against

    output_path: str (default ".")
        Path to folder in which plots are s
    """
    percentages = list(range(10, 101))
    mean, std = np.zeros(len(percentages)), np.zeros(len(percentages))
    for i, p in enumerate(percentages):
        batch_results = np.zeros(TEST_REPEAT)
        for j in range(TEST_REPEAT):
            sample_X = train_X.sample(frac=(p / 100.0))
            sample_y = train_y.loc[sample_X.index]
            model = LinearRegression()
            batch_results[j] = model.fit(sample_X.to_numpy(), sample_y).loss(test_X.to_numpy(), test_y.to_numpy())
        mean[i] = np.mean(batch_results)
        std[i] = np.std(batch_results)

    fig = px.line(x=percentages, y=mean, labels={'x': 'Percentage of Sample', 'y': 'MSE of Price'}, markers=True,
                  title="MSE of House Price by Percent of Train Samples")
    lower = mean - 2 * std
    upper = mean + 2 * std
    fig.add_trace(
        go.Scatter(x=percentages, y=lower, fill=None, mode="lines", line=dict(color="lightgrey"), showlegend=False))
    fig.add_trace(go.Scatter(x=percentages, y=upper, fill='tonexty', mode="lines", line=dict(color="lightgrey"),
                             showlegend=False))
    fig.write_html('{}/MSE.html'.format(output_path))


if __name__ == '__main__':
    np.random.seed(0)
    df = pd.read_csv("../datasets/house_prices.csv")

    # Question 1 - split data into train and test sets
    train_X, train_y, test_X_orig, test_y = split_train_test(df, df['price'])

    # Question 2 - Preprocessing of housing prices dataset
    train_X, train_y = preprocess_data(train_X, train_y)

    test_X = test_X_orig[test_X_orig['price'].notna()]  # remove rows without price label
    diff = test_X_orig.index.difference(test_X.index)
    test_y.drop(diff, inplace=True)
    test_X = preprocess_data(test_X)

    cur_datetime = f"{datetime.now():%Y_%m_%d_%H_%M_%S}"
    path = os.path.join('./figs', cur_datetime)
    if not os.path.exists(path):
        os.makedirs(path)

    # Question 3 - Feature evaluation with respect to response
    #feature_evaluation(train_X, train_y, output_path=path)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    fit_gradually(train_X, test_X, train_y, test_y, output_path=path)
