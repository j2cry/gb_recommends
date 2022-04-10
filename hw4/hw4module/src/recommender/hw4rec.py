import enum
import numpy as np
import pandas as pd

from implicit.als import AlternatingLeastSquares
from functools import partial
from itertools import zip_longest
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin
from typing import List


# these must be initialized
userid_to_id = {}
itemid_to_id = {}
id_to_userid = {}
id_to_itemid = {}


class BColor(enum.Enum):
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def cprint(msg, color: BColor):
    print(f'{color}{msg}{BColor.ENDC}')


def popularity_measure(source, fields: List[str], k=5000, beta: List[float] = None, add_target=None, scaler=None):
    """ Расчет оценки важности товара в покупке и отбор топ K наиболее популярных товаров
    :param source - исходные данные
    :param fields - признаки, по которым измеряется мера важности товара
    :param k - количество товаров, отбираемых в топ
    :param beta - множители значимости для каждого признака в оценке
    :param add_target - название финального признака. Признак не добавляется, если target = None
    :param scaler - класс масштабирования данных
    """
    b = [1.] * len(fields) if beta is None else np.array(beta)
    assert len(fields) == len(b), '`fields` and `beta` dimensions must equal'
    assert issubclass(StandardScaler, TransformerMixin) or scaler is None, 'scaler must be a subclass of TransformerMixin'
    _df = source[['item_id']].copy()
    prepared = scaler().fit_transform(source[fields]) * b if scaler else source[fields] * b
    values = np.linalg.norm(prepared, ord=2, axis=1)
    _df['popularity'] = values
    if add_target:
        source.loc[:, add_target] = values
    popularity = _df.groupby('item_id')['popularity'].sum()
    return popularity.sort_values(ascending=False).head(k).index.tolist()


def check_model(uim, mdl_params, rec_params, res, ttl='als'):
    """
    :param uim: user-item matrix
    :param mdl_params: model init parameters
    :param rec_params: recommendation parameters
    :param res: true values, including user_id
    :param ttl: model title
    :return: predicted values (DataFrame)
    """
    mdl = AlternatingLeastSquares(**mdl_params)
    mdl.fit(uim.T, show_progress=False)
    # rec_params['user_items'] = uim
    res[ttl] = res['user_id'].apply(partial(recommender, mdl=mdl, params=rec_params))
    return mdl


def recommender(user_id, mdl, params):
    """ Предсказатель-интерпретатор """
    uid = userid_to_id.get(user_id, None)
    if uid is None:
        return list()
    rec_score = mdl.recommend(userid_to_id[user_id], **params)
    return [id_to_itemid[rec[0]] for rec in rec_score]


def precision_at_k(recommended_list, bought_list, k=5):
    """"""
    flags = np.sum(np.isin(bought_list, recommended_list[:k]))
    return flags / k


def ap_k(recommended_list, bought_list, k=5):
    """"""
    flags = np.isin(recommended_list, bought_list)
    if np.sum(flags) == 0:
        return 0

    func = partial(precision_at_k, recommended_list, bought_list)
    rel_items = np.arange(1, k + 1)[flags[:k]]
    return np.sum(list(map(func, rel_items))) / np.sum(flags)


def calc_metric(metric_func, source: pd.DataFrame):
    """ Подсчет метрики
    :param metric_func - функция измерения метрики. Первый аргумент - рекомендации, второй - актуальные значения
    :param source - данные для подсчета метрики
    """
    def metric_wrapper(pred, act):
        return metric_func(pred, act) if len(pred) != 0 else 0

    metric = pd.DataFrame()
    for col in source.columns:
        if col == 'user_id':
            metric[col] = source[col]
        elif col == 'actual':
            continue
        else:
            metric[col] = source[[col, 'actual']].apply(lambda row: metric_wrapper(*row.values), axis=1)
    return metric


def compare_metrics(res, saveto=None):
    """ Build dataframe with metrics comparison """
    pr_at_k = calc_metric(partial(precision_at_k, k=5), res)
    ap_at_k = calc_metric(lambda pred, act: ap_k(pred, act, k=min(5, len(pred))), res)
    smr = pd.DataFrame([pr_at_k.mean(), ap_at_k.mean()], index=['precision@k', 'map@k']).drop(columns='user_id')
    if saveto:
        smr.T.to_csv(saveto)
    return smr


def get_nearest(mdl, elem_id, k, mode):
    """ Get top K the nearest users/items to the given
    :param mdl: ALS fitted model
    :param elem_id: real user/item id
    :param k: number of items to find
    :param mode: users/items return switcher
    :return: list of similar users/items depend on mode
    """
    if (mode == 'user') or (mode == 0):
        return [id_to_userid[idx] for idx, _ in mdl.similar_users(userid=userid_to_id[elem_id], N=k + 1)[1:]]
    if (mode == 'item') or (mode == 1):
        return [id_to_itemid[idx] for idx, _ in mdl.similar_items(itemid=itemid_to_id[elem_id], N=k + 1)[1:]]
    return []


def filter_top_for_users(items, users, measure='popularity', k=5):
    """ Get users top purchases
    :param items: data grouped by users and items
    :param users: user ids array
    :param measure: ranging measure
    :param k: number of items to find
    :return ungrouped dataframe
    """
    filter_mask = (items['user_id'].isin(users)) & (items['item_id'] != -1)
    return items[filter_mask].sort_values(by=['user_id', measure], ascending=[True, False]).groupby('user_id').head(k)


def basic_filter(items, k, placeholder=()):
    """ Из списка товаров берем K первых, отличный от товара-заглушки, а если таких нет, то возвращаем заглушку """
    return result[:k] if (result := [item for item in items if item != -1]) else placeholder


def check_items_count(items, k):
    """ Check number of predictions for each user
    :param items: Series with users predictions. User ids must be in index
    :param k: number of required predictions
    :return: corrected predictions
    """
    # если похожие пользователи мало покупали, то рекомендаций может не хватить
    sizes = items.apply(len)
    if (low_pred := items.index[sizes < k]).any():
        cprint(f"Some users have less than {k} predictions!", BColor.WARNING)
        print(low_pred.tolist())
        # какая-то обработка подобных ситуаций
    if (nan_pred := items.index[sizes == 0]).any():
        cprint(f"Some users have no predictions at all!", BColor.FAIL)
        print(nan_pred.tolist())
        # какая-то обработка подобных ситуаций
    return items


def agg_func(src):
    """ Аггрегатор похожих товаров: для каждого товара берем верхние в очереди если они еще не в подборке """
    arr = np.array(list(zip_longest(*src)), dtype='float')
    res = []
    for row in range(arr.shape[0]):
        for col in range(arr.shape[1]):
            if np.isnan(item := arr[row, col]):
                continue
            if item not in res:
                res.append(item)
            else:
                for col_item in arr[row + 1:, col]:
                    if not np.isnan(col_item) and col_item not in res:
                        res.append(col_item)
                        break
    return np.array(res, dtype='int')


def similar_item_recommend(mdl, users, data, measure='popularity', k=5,
                           filter_func=basic_filter, placeholder=(), title='similar_items'):
    """ Recommend similar items based on top K purchases
    :param mdl: ALS fitted model
    :param users: user ids to recommend for
    :param data: source dataset
    :param measure: target field in the dataset
    :param k: number of items to recommend
    :param filter_func: additional filters like func(items: list) -> list
    :param placeholder: value to use if no predictions available
    :param title: name of target column
    :return: list of predictions for given user
    """
    # по userid получаем топ покупок пользователей
    group_items = data.groupby(['user_id', 'item_id'])[measure].sum().reset_index()
    user_item_top = filter_top_for_users(group_items, users, measure, k)

    # для каждого товара из топа пользователя находим ближайшие K товаров из топ5к
    user_item_top[title] = user_item_top['item_id'].apply(lambda x: get_nearest(mdl, x, k, 'item'))
    # для каждого товара итеративно берем его ближайший, если его еще нет в предложке,
    preds = user_item_top.groupby('user_id')[title].agg(agg_func)

    # теперь можем дополнительно отфильтровать полученные списки
    #   если фильтр не указан - берем первые К товаров
    preds = preds.apply(lambda val: filter_func(val, k, placeholder) if filter_func and callable(filter_func) else lambda x: x[:k])

    # добавляем тех, для кого предсказания отсутствуют
    items = pd.Series([np.array(placeholder)] * len(users), index=users, name=title)
    items.update(preds)
    # проверяем количество предсказаний
    items = check_items_count(items, k)
    return items


def similar_user_recommend(mdl, users, data, measure='popularity', k=5,
                           filter_func=basic_filter, placeholder=(), title='similar_users'):
    """ Recommend items based on similar user purchases
    :param mdl: ALS fitted model
    :param users: user ids to recommend for
    :param data: source dataset
    :param measure: target field in the dataset
    :param k: number of items to recommend
    :param filter_func: additional filters like func(items: list) -> list
    :param placeholder: value to use if no predictions available
    :param title: name of target column
    :return: list of predictions for given user
    """
    # для каждого юзера из запроса находим K ближайших
    sim = pd.Series(users).apply(lambda uid: get_nearest(mdl, uid, k, 'user'))
    # для каждого пользователя в запросе составляем общий список товаров из топ К покупок каждого ближайшего пользователя
    # полученные списки содержат наиболее релевантные товары ближайшего(-их) пользователя(-ей)
    all_items = data.groupby(['user_id', 'item_id'])[measure].sum().reset_index()
    items = sim.apply(lambda x: filter_top_for_users(all_items, x, measure, k)['item_id'].drop_duplicates().values)
    # теперь можем дополнительно отфильтровать полученные списки
    #   если фильтр не указан - берем первые К товаров
    items = items.apply(lambda val: filter_func(val, k, placeholder) if filter_func and callable(filter_func) else lambda x: x[:k])
    # индексируем номерами пользователей
    items.name = title
    items.index = users
    # проверяем количество предсказаний
    items = check_items_count(items, k)
    return items
