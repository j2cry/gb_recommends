""" Модуль с написанными ранее функциями """


import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from scipy.sparse import csr_matrix
from typing import List


class DataProcessor:
    def __init__(self, train, test=None, top_config=None, uim_config=None):
        """ Initialize data preparer """
        if uim_config is None:
            uim_config = {}
        if top_config is None:
            top_config = {}
        self.placeholder_id = 0
        self.top_config = top_config
        self.uim_config = uim_config
        # init required variables
        self.__measure_title = 'DataPreparer@popularity'
        self.top_k_items = None
        # user-item matrices
        self.train = train
        self.train_uim = None
        self.train_uim_sparse = None
        self.train_uim_weighted = None
        self.test = None
        if test is not None:
            self.test = test
            self.test_uim = None
            self.test_uim_sparse = None
            self.test_uim_weighted = None
        # remap dictionaries
        self.id_to_itemid = None
        self.id_to_userid = None
        self.itemid_to_id = None
        self.userid_to_id = None
        # debug variables
        self.top_train = None

    def fit(self):
        """ Prepare top K items, user-item matrix, id-remap dictionaries
        :param top_config: top K items select configuration
            :keyword fields: признаки, по которым измеряется мера важности товара
            :keyword k: количество товаров, отбираемых в топ
            :keyword beta: множители значимости для каждого признака в оценке
            :keyword scaler: класс масштабирования данных
        :param uim_config: user-item matrix preparer configuration
            :keyword aggfunc: pivot table aggregation function
            :keyword weights: функция взвешивания.
                              На входе и на выходе item-user матрица (т.е. транспонированная user-item)
        """
        self.top_k_items = self.k_popular_items(**self.top_config)
        self.train_uim_sparse = self.prepare_uim(**self.uim_config)
        if self.test is not None:
            self.test_uim_sparse = self.prepare_test_uim(self.top_config, **self.uim_config)

    @staticmethod
    def popularity_measure(data, fields: List[str] = ['quantity'], beta: List[float] = None, scaler=None, **kwargs):
        """ Расчет оценки важности товара в покупке
        :param data: исходные данные
        :param fields: признаки, по которым измеряется мера важности товара
        :param beta: множители значимости для каждого признака в оценке
        :param scaler: класс масштабирования данных
        """
        # fields = list(fields)
        b = [1.] * len(fields) if beta is None else np.array(beta)
        assert len(fields) == len(b), '`fields` and `beta` dimensions must equal'
        assert (scaler is None) or issubclass(scaler, TransformerMixin), 'scaler must be a subclass of TransformerMixin'
        prepared = scaler().fit_transform(data[fields]) * b if scaler else data[fields] * b
        values = np.linalg.norm(prepared, ord=2, axis=1)
        return values

    def k_popular_items(self, **top_config):
        """ Расчет оценки важности товара в покупке и отбор топ K наиболее популярных товаров """
        k = top_config.pop('k', 10)
        self.train.loc[:, self.__measure_title] = self.popularity_measure(self.train, **top_config)
        popularity = self.train.groupby('item_id')[self.__measure_title].sum()
        return popularity.sort_values(ascending=False).head(k).index.tolist()

    def prepare_uim(self, aggfunc='count', weights=None):
        """ Подготовка user-item матрицы
        :param aggfunc: pivot table aggregation function
        :param weights: функция взвешивания. На входе и на выходе item-user матрица (т.е. транспонированная user-item)
        """
        self.top_train = self.train.copy()
        # товары не из топ5000 превращаем в один товар
        self.top_train.loc[~self.top_train['item_id'].isin(self.top_k_items), 'item_id'] = self.placeholder_id
        # подготовка обучающих данных: составление таблицы user-item на основе популярности товара для пользователя
        uim = pd.pivot_table(self.top_train,
                             index='user_id', columns='item_id', values=self.__measure_title,
                             aggfunc=aggfunc, fill_value=0)
        # обнуляем значимость товаров, не входящих в топ5к
        uim[self.placeholder_id] = 0
        # переведем в нужный формат
        self.train_uim = uim.astype(float)
        # remap dictionaries
        self.id_to_itemid = dict(enumerate(self.train_uim.columns.values))
        self.id_to_userid = dict(enumerate(self.train_uim.index.values))
        self.itemid_to_id = {v: k for k, v in self.id_to_itemid.items()}
        self.userid_to_id = {v: k for k, v in self.id_to_userid.items()}
        # применим веса        
        self.train_uim_weighted = csr_matrix(weights(self.train_uim.T).T).tocsr() if weights else csr_matrix(self.train_uim).tocsr()
        return csr_matrix(self.train_uim).tocsr()

    def prepare_test_uim(self, top_config, aggfunc='count', weights=None):
        # отсеиваем из test товары, не попавшие в train
        id_in_train = self.test['item_id'].isin(self.top_k_items)
        data_test = self.test[id_in_train].copy()
        # измеряем меру популярности товаров для создания pivot table
        data_test[self.__measure_title] = self.popularity_measure(data_test, **top_config)
        self.test_uim = pd.pivot_table(data_test,
                                       index='user_id', columns='item_id', values=self.__measure_title,
                                       aggfunc=aggfunc, fill_value=0)
        # нужны ли тут remap-словари?
        # применим веса
        self.test_uim_weighted = csr_matrix(weights(self.test_uim.T).T).tocsr() if weights else csr_matrix(self.test_uim).tocsr()
        return csr_matrix(self.test_uim).tocsr()

