from functools import wraps
from sklearn.base import BaseEstimator, TransformerMixin


class Merger(BaseEstimator, TransformerMixin):
    def __init__(self, source, *, on, cols=None):
        on_cols = on if isinstance(on, (list, tuple)) else [on]
        self.source = source if cols is None else source[on_cols + cols]
        self.on = on

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.merge(self.source, on=self.on, how='left')


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


def fit_last_days(func):
    def wrapper(self, X, **kwargs):
        self.last_days = X['day'] > X['day'].max() - self.n_days
        return func(self, X, **kwargs)
    return wrapper


class LastDaysBase(BaseEstimator, TransformerMixin):
    """ Базовый класс фичей с отбором последних N дней """
    def __init__(self, *, n_days):
        self.n_days = n_days
        self.last_days = None

    def fit(self, X, y=None):
        return self


class LastDaysRate(LastDaysBase):
    """ Доля дней из последних N, в которые товар был куплен """
    @fit_last_days
    def transform(self, X):
        last_days_rate = (X[self.last_days].groupby('item_id')['day'].nunique() / self.n_days).rename(f'last_{self.n_days}_days_rate')
        return X.merge(last_days_rate, on='item_id', how='left').fillna(0)


class BasketRate(LastDaysBase):
    """ Доля уникальных чеков за последние N дней, в которых присутствовал товар. Адаптируется к данным. """
    @fit_last_days
    def transform(self, X):
        basket_rate = (X[self.last_days].groupby('item_id')['basket_id'].nunique() / X['basket_id'].nunique()).rename(f'basket_rate_for_{self.n_days}_days')
        return X.merge(basket_rate, on='item_id', how='left').fillna(0)


class DepartmentSellRate(LastDaysBase):
    """ Доля продаж категории в последние N дней """
    def __init__(self, item_data, **kwargs):
        super().__init__(**kwargs)
        self.item_data = item_data

    @fit_last_days
    def transform(self, X):
        merged = X[self.last_days].merge(self.item_data, on='item_id', how='left')
        cat_rate = (merged.groupby('department')['item_id'].count() / merged['item_id'].count()).rename(f'department_sell_rate_for_{self.n_days}_days')
        cat_rate_merged = self.item_data[['item_id', 'department']].merge(cat_rate, on='department', how='left')
        return X.merge(cat_rate_merged, on='item_id', how='left').drop(columns='department')


class SameDepartmentPurchases(LastDaysBase):
    """ Кол-во приобретенных пользователем товаров из той же категории за последние N дней """
    def __init__(self, item_data, **kwargs):
        super().__init__(**kwargs)
        self.item_data = item_data

    @fit_last_days
    def transform(self, X):
        merged = X[self.last_days].merge(self.item_data[['item_id', 'department']], on='item_id', how='left')
        same_department_purchases = merged.groupby(['user_id', 'department'])['item_id'].count().\
            rename(f'same_department_purchases_for_{self.n_days}_days')
        return X.merge(self.item_data[['item_id', 'department']], on='item_id', how='left').\
            merge(same_department_purchases, on=['user_id', 'department'], how='left').\
            fillna(0).drop(columns='department')


class MeanDepartmentExpenses(LastDaysBase):
    """ Средняя сумма покупок пользователя в категории за N последних дней """
    def __init__(self, item_data, **kwargs):
        super().__init__(**kwargs)
        self.item_data = item_data

    @fit_last_days
    def transform(self, X):
        feature_title = f'mean_department_expenses_for_{self.n_days}_days'
        mean_department_expenses = X[self.last_days].groupby(['user_id', 'item_id'])[['quantity', 'sales_value']].\
            agg({'quantity': 'sum', 'sales_value': 'mean'}).prod(axis=1).rename(feature_title).reset_index()
        # это на случай, если вдруг sales_value это уже конечная цена за указанное число единиц
        # mean_department_expenses = X[self.last_days].groupby(['user_id', 'item_id'])['sales_value'].mean().\
        #     rename(feature_title).reset_index()

        mean_department_expenses = mean_department_expenses.\
            merge(self.item_data[['item_id', 'department']], on='item_id', how='left').\
            groupby(['user_id', 'department'])[feature_title].mean()

        return X.merge(self.item_data[['item_id', 'department']], on='item_id', how='left').\
            merge(mean_department_expenses, on=['user_id', 'department'], how='left').\
            fillna(0).drop(columns='department')
