import pandas as pd
from preprocess import DataPreprocessor
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnsCorrector(BaseEstimator, TransformerMixin):
    """ Выброс фичей, добавление отсутствующих, исправление порядка """
    def __init__(self, cols, mode):
        """
        :param cols: columns to process
        :param mode: drop or keep
        """
        self.cols = cols
        self.mode = mode
        self.required_order = None

    def fit(self, X, y=None):
        self.cols = [col for col in self.cols if col in X.columns]
        if self.mode == 'keep':
            self.required_order = [col for col in X.columns if col in self.cols]
        else:
            self.required_order = [col for col in X.columns if col not in self.cols]
        return self

    def transform(self, X):
        df = X.copy()
        absence = list(set(X.columns) ^ set(self.required_order))
        df[absence] = 0
        return df[self.required_order]


class LastDaysRate(BaseEstimator, TransformerMixin):
    """ Доля дней из последних N, в которые товар был куплен. Адаптируется к данным.
    """
    def __init__(self, *, n_days):
        self.n_days = n_days

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        last_days = X['day'] > X['day'].max() - self.n_days
        last_days_rate = (X[last_days].groupby('item_id')['day'].nunique() / self.n_days).rename(f'last_{self.n_days}_days_rate')
        return X.merge(last_days_rate, on='item_id', how='left').fillna(0)


class BasketRate(BaseEstimator, TransformerMixin):
    """ Доля уникальных чеков за последние N дней, в которых присутствовал товар. Адаптируется к данным. """
    def __init__(self, *, n_days):
        self.n_days = n_days

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        last_days = X['day'] > X['day'].max() - self.n_days
        basket_rate = (X[last_days].groupby('item_id')['basket_id'].nunique() / X['basket_id'].nunique()).rename(f'basket_rate_for_{self.n_days}_days')
        return X.merge(basket_rate, on='item_id', how='left').fillna(0)
