""" модель, выдающая N кандидат-предиктов каждому юзеру (ALS / Cosine / ItemItem / TfIdf / Bm25 / etc) """
import pandas as pd
from functools import partial
from scipy.sparse import csr_matrix
from implicit.nearest_neighbours import ItemItemRecommender, CosineRecommender, TFIDFRecommender, BM25Recommender
from implicit.als import AlternatingLeastSquares

DEFAULT_I2I_FIT = {
    'K': 2      # K nearest users recommender
}

DEFAULT_ALS_FIT = {
    'factors': 44,
    'regularization': 0.001,
    'iterations': 15,
    'calculate_training_loss': True,
    'use_gpu': False,
    'random_state': 23
}

SUPPORTED_MODELS = {
    'ItemItem': ItemItemRecommender,
    'Cosine': CosineRecommender,
    'TFIDF': TFIDFRecommender,
    'BM25': BM25Recommender,
    'ALS': AlternatingLeastSquares
}


class CandidateModel:
    def __init__(self, model_name, *, train, weighted=None, top_items=None,
                 placeholder_id=0, user_to_idx=None, idx_to_item=None, item_to_idx=None):
        """ Initialize candidate model class
        :param model_name: model identifier
        :param train: train dataset
        :param weighted: weighted train dataset
        :param top_items: list of top items. Specify this for missed predictions filling
        :param placeholder_id: item ID placeholder for items out of top-K
        :param user_to_idx: user ID to index remap dictionary
        :param idx_to_item: item index to ID remap dictionary
        :param item_to_idx: item ID to index remap dictionary
        """
        assert model_name in SUPPORTED_MODELS.keys(), f'Specified model `{model_name}` is not supported'
        self.model_name = model_name
        self.train = csr_matrix(train)
        self.weighted = csr_matrix(weighted) if weighted is not None else self.train
        self.top_items = top_items
        self.model = None
        # remap dictionaries
        self.placeholder_id = placeholder_id
        assert (not user_to_idx or isinstance(user_to_idx, dict)), '`user_to_idx` must be dict'
        self.user_to_idx = user_to_idx
        assert (not idx_to_item or isinstance(idx_to_item, dict)), '`idx_to_item` must be dict'
        self.idx_to_item = idx_to_item
        assert (not item_to_idx or isinstance(item_to_idx, dict)), '`item_to_idx` must be dict'
        self.item_to_idx = item_to_idx
        # recommender parameters
        self.current_rec_params = {
            'user_items': self.train,
            'N': 50,
            'filter_already_liked_items': False,
            'filter_items': [self.item_to_idx[self.placeholder_id]] if self.placeholder_id and self.item_to_idx else None,
            'recalculate_user': True        # has no effect for ItemItemRecommender subclasses
        }

    def fit(self, **fit_params):
        """ Fit candidate model """
        # switcher for ALS
        params = DEFAULT_ALS_FIT.copy() if self.model_name == 'ALS' else DEFAULT_I2I_FIT.copy()
        params.update(fit_params)
        self.model = SUPPORTED_MODELS.get(self.model_name)(**params)
        self.model.fit(self.weighted.T, show_progress=False)

    def predict(self, user_array, **rec_params):
        """ Get predicts for given users """
        self.current_rec_params.update(rec_params)
        # prepare predictor & predict
        predictor = partial(self.__recommend, rec_params=self.current_rec_params)
        predicts = (user_array if hasattr(user_array, 'apply') else pd.Series(user_array)).apply(predictor)
        # check predicts amount and fill missing
        if self.top_items:
            k = self.current_rec_params['N']
            predicts = self.fill_from_top(predicts, k)
        return predicts

    def __recommend(self, user_id, rec_params):
        """ Apply to remap dictionaries if they are given, otherwise return predicted indices """
        uid = self.user_to_idx.get(user_id, None) if self.user_to_idx else user_id
        if uid is None:
            return list()
        rec_score = self.model.recommend(uid, **rec_params)
        return [self.idx_to_item[rec[0]] if self.idx_to_item else rec[0] for rec in rec_score]

    def fill_from_top(self, predicts, k):
        """ Fill missing predicts from top K items """
        predicts = predicts.copy()
        predicts_amount = predicts.apply(len)
        if (low_pred := predicts.index[predicts_amount < k]).any():
            predicts[low_pred] = predicts[low_pred].apply(lambda pred: list(pred) + [item for item in self.top_items if item not in pred][:k - len(pred)])
        return predicts
