import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"

TEMP_THRESHOLD = -20  # Derived from lowest temperature recorded in any of the countries in the data
K = 5


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=['Date']).dropna().drop_duplicates()
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df = df[df['Temp'] > TEMP_THRESHOLD]
    df = df.astype({'Year': str})
    return df


def plot_by_country(df: pd.DataFrame, country: str):
    """
    Plot a temperature by day-of-year graph for the given country
    """
    df = df[df['Country'] == country]
    fig = px.scatter(df, x='DayOfYear', y='Temp', color='Year', symbol='Year',
                     title='Temperature by Day of Year')
    fig.show()
    agg_df = df.groupby('Month', as_index=False).agg(std=('Temp', 'std'))
    fig2 = px.bar(agg_df, x='Month', y='std', title='Standard Deviation of Temperature by Month')
    fig2.show()


def plot_all_counties(df: pd.DataFrame):
    """
    Plot the mean temperature by month for all available countries
    """
    agg_df = df.groupby(['Country', 'Month'], as_index=False).agg(std=('Temp', 'std'), mean=('Temp', 'mean'))
    fig = px.line(agg_df, x='Month', y='mean', error_y='std', color='Country', symbol='Country',
                  title='Mean Temperature by Month').update_layout(yaxis_title='Avg Temp')
    fig.show()


def fit_for_country(df: pd.DataFrame, country: str):
    """
    Fit a polynomial model (predicting the temperature) for a specific country, using polynomial degrees between 1 and
    10. Show a bar plot of the loss by polynomial degree.
    """
    df = df[df['Country'] == country]
    train_x, train_y, test_x, test_y = split_train_test(df['DayOfYear'], df['Temp'])
    loss = []
    for i, k in enumerate(range(1, 11)):
        model = PolynomialFitting(k)
        model.fit(train_x.to_numpy(), train_y.to_numpy())
        loss.append(np.round(model.loss(test_x.to_numpy(), test_y.to_numpy()), 2))

    loss_df = pd.DataFrame({'k': list(range(1, 11)), 'loss': loss})
    fig = px.bar(loss_df, x='k', y='loss', text='loss', title='Test Error for k Values')
    fig.show()


def plot_all_by_country(df: pd.DataFrame, country: str, k: int):
    """
    Given a polynomial degree k, fit a model over the given country, and evaluate the model over different countries.
    Show a figure of the loss by country.
    """
    model_df = df[df['Country'] == country]
    model = PolynomialFitting(k)
    model.fit(model_df['DayOfYear'].to_numpy(), model_df['Temp'].to_numpy())
    countries = df['Country'].unique()
    countries = np.delete(countries, np.argwhere(countries == country))

    loss = []
    for c in countries:
        filtered_df = df[df['Country'] == c]
        loss.append(np.round(model.loss(filtered_df['DayOfYear'], filtered_df['Temp']), 2))

    loss_df = pd.DataFrame({'Country': countries, 'Loss': loss})
    fig = px.bar(loss_df, x='Country', y='Loss', color='Country', text='Loss',
                 title='Loss by Countries for Model fitted Over Israel')
    fig.show()


if __name__ == '__main__':
    np.random.seed(5)
    # Question 1 - Load and preprocessing of city temperature dataset
    X = load_data('../datasets/City_Temperature.csv')

    # Question 2 - Exploring data for specific country
    plot_by_country(X, 'Israel')

    # Question 3 - Exploring differences between countries
    plot_all_counties(X)

    # Question 4 - Fitting model for different values of `k`
    fit_for_country(X, 'Israel')

    # Question 5 - Evaluating fitted model on different countries
    plot_all_by_country(X, 'Israel', K)
