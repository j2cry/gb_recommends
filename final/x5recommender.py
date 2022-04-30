import pathlib
import pickle
import configparser
import pandas as pd
from datasplit import DataSplit, prepare_true_values
from sklearn.preprocessing import StandardScaler
from metrics import calc_mean_metric, precision_at_k

DEFAULT_CONF_PATH = pathlib.Path('default.cnf').as_posix()
DEFAULT_CANDIDATES = 100

# preprocessor fitting parameters
PRE_FIT = {
    'top_config': {'fields': ['quantity', 'sales_value'],
                   'beta': [1., 1.],
                   'k': 5000,
                   'scaler': StandardScaler
                   },
    'uim_config': {'aggfunc': 'sum'},
}


class X5Recommender:
    """ Recommender production class """
    def __init__(self, config_path=DEFAULT_CONF_PATH):
        self.cm = None
        self.featuring = None
        self.rank_model = None
        self.reload(config_path)

    def reload(self, config_path=DEFAULT_CONF_PATH):
        """ Reload models according to the configuration """
        config = configparser.ConfigParser()
        config.read(config_path)
        self.cm = pickle.load(open(config['PATHS']['candidate_model'], 'rb'))
        self.featuring = pickle.load(open(config['PATHS']['featuring_pipeline'], 'rb'))
        self.rank_model = pickle.load(open(config['PATHS']['rank_model'], 'rb'))

    def predict(self, data, *, k):
        """ Get predictions and ranking metric on given data
        :param data: purchases data like `retail_train.csv`
        :param k: number of required recommended items
        """
        candidates = self.get_candidates(data['user_id'].unique())
        # both warm & cold: candidates for cold users are predicted from top 5k items
        merged_data, _ = self.merge_candidates(data, candidates)
        featured_data = self.featuring.transform(merged_data)
        true_ranked_values = prepare_true_values(merged_data)

        proba = pd.Series(self.rank_model.predict_proba(featured_data).T[1], name='proba')
        ranked_pred = pd.concat([featured_data[['user_id', 'item_id']], proba], axis=1)
        # collect recommends
        sorted_candidates = candidates.merge(ranked_pred, on=['user_id', 'item_id'], how='left')\
            .sort_values(by=['user_id', 'proba'], ascending=[True, False])\
            .groupby('user_id').head(k)
        predict_candidates = sorted_candidates.groupby('user_id')['item_id'].unique()
        # calc rank metric
        rank_metric = calc_mean_metric(precision_at_k,
                                       true_ranked_values['actual'], predict_candidates.reset_index(drop=True), k=k)
        # fill missing predictions from top K items
        predicts = self.cm.fill_from_top(predict_candidates, k)
        return predicts, rank_metric

    def get_candidates(self, users, *, save_to=None) -> pd.DataFrame:
        """ Get candidates for specified users
        :param users: array-like users collection to whom candidates are predicted
        :param save_to: filepath to save candidates to
        :return: DataFrame with predicted candidates
        """
        if not isinstance(users, pd.Series):
            users = pd.Series(users, name='user_id')

        target_candidates = self.cm.predict(users, N=DEFAULT_CANDIDATES)
        candidates = pd.DataFrame.from_dict(target_candidates.to_dict(), orient='index').set_index(users)
        candidates = candidates.stack().reset_index(level=1, drop=True).rename('item_id').reset_index()
        if save_to is not None:
            candidates.to_csv(save_to, index=False)
        return candidates

    @staticmethod
    def merge_candidates(data, candidates, users=None):
        """ Prepare dataset lv2 for featuring
        :param data: required data to be prepared
        :param candidates: dataset with stacked candidates
        :param users: array-like collection of users to process, others will be dropped
            Keep `None` to process them all.
        """
        if users is not None:
            keep_users = data['user_id'].isin(users)
            target = data[keep_users].copy()
        else:
            target = data.copy()
        # keep candidates for only required users
        required_users = candidates['user_id'].isin(target['user_id'].unique())
        target['target'] = 1  # flag means this item was really bought
        target = candidates[required_users].merge(target, on=['user_id', 'item_id'], how='left').fillna(0)
        return target.drop(columns='target'), target['target']

    def demo(self, k=5):
        purchases = pd.read_csv('retail_train.csv')
        splitter = DataSplit(purchases, 'week_no', [6, 4])

        train = purchases[splitter.part1].copy()
        valid = purchases[splitter.part2].copy()
        true_train = prepare_true_values(train)
        true_valid = prepare_true_values(valid)

        print('predicting train...')
        train_predicts, train_rank_pr = self.predict(train, k=k)
        print('predicting valid...')
        valid_predicts, valid_rank_pr = self.predict(valid, k=k)

        print('calculating metrics...')
        train_pr = calc_mean_metric(precision_at_k, true_train['actual'], train_predicts.reset_index(drop=True), k=k)
        valid_pr = calc_mean_metric(precision_at_k, true_valid['actual'], valid_predicts.reset_index(drop=True), k=k)

        print(train_rank_pr, valid_rank_pr)
        print(train_pr, valid_pr)


if __name__ == '__main__':
    x5rec = X5Recommender()
    x5rec.demo()
