# downloaded as .py from jupyter

import torch
import numpy as np
from functools import partial


def hit_rate(recommended_list, bought_list):
    flags = np.isin(bought_list, recommended_list)
    hit_rate = (flags.sum() > 0) * 1
    return hit_rate


def hit_rate_at_k(recommended_list, bought_list, k=5):
    """ Hit rate@k = (был ли хотя бы 1 релевантный товар среди топ-k рекомендованных) """
    # с использованием numpy
    flags = np.isin(recommended_list[:k], bought_list)
    return (flags.sum() > 0) * 1

    # без использования numpy
    # return (len(set(bought_list) & set(recommended_list[:k])) > 0) * 1


def money_precision_at_k(recommended_list, bought_list, prices_recommended, k=5):
    """ Доля дохода по рекомендованным объектам
    :param recommended_list - список id рекомендаций
    :param bought_list - список id покупок
    :param prices_recommended - список цен для рекомендаций
    """
    flags = np.isin(recommended_list[:k], bought_list)
    prices = np.array(prices_recommended[:k])
    return flags @ prices / prices.sum()


def recall_at_k(recommended_list, bought_list, k=5):
    """ Recall on top k items """
    flags = np.isin(bought_list, recommended_list[:k])
    return flags.sum() / len(bought_list)


def money_recall_at_k(recommended_list, bought_list, prices_recommended, prices_bought, k=5):
    """ Доля дохода по релевантным рекомендованным объектам
    :param recommended_list - список id рекомендаций
    :param bought_list - список id покупок
    :param prices_recommended - список цен для рекомендаций
    :param prices_bought - список цен покупок
    """
    flags = np.isin(recommended_list[:k], bought_list)      # get recommend to bought matches
    prices = np.array(prices_recommended[:k])               # get prices of recommended items
    return flags @ prices / np.sum(prices_bought)


def precision_at_k(recommended_list, bought_list, k=5):
    flags = np.isin(bought_list, recommended_list[:k])
    return flags.sum() / k


def ap_k(recommended_list, bought_list, k=5):
    # переработано
    flags = np.isin(recommended_list, bought_list)
    if sum(flags) == 0:
        return 0

    # sum_ = 0
    # for i in range(0, k-1):
    #     if flags[i]:
    #         sum_ += precision_at_k(recommended_list, bought_list, k=i+1)
    # result = sum_ / flags.sum()
    # return result

    func = partial(precision_at_k, recommended_list, bought_list)
    rel_items = np.arange(1, k + 1)[flags[:k]]                  # получаем номера релевантных объектов
    return np.sum(list(map(func, rel_items))) / flags.sum()     # считаем avg precision@k для этих объектов


# v1
def map_k_v1(recommended_list, bought_list, k=5, u=1, w=None):
    """ Среднее AP@k по u пользователям """
    apk = []
    w = w if w is not None else np.ones(shape=(u, ))
    for user, user_weight in zip(range(u), w):
        apk.append(ap_k(recommended_list[user], bought_list[user], k) * user_weight)
    
    return np.mean(apk)


# v2
def map_k_v2(recommended_list, bought_list, k=5, u=1, w=None):
    """ Среднее AP@k по u пользователям """
    func = partial(ap_k, k=k)
    weights = w if w is not None else np.ones(shape=(u, ))
    apk = np.array(list(map(func, recommended_list[:u], bought_list[:u]))) * weights
    return apk.mean()


def reciprocal_rank(recommended_list, bought_list, n=1, k=5):    
    """ обратный ранг n релевантных рекомендаций среди первых k рекомендаций
    Это не совсем канонический reciprocal rank, но при n=1 должен работать как принято
    
    :param recommended_list - список рекомендаций
    :param bought_list - список покупок
    :param n - учитывать первые n релевантных рекомендаций
    :param k - искать релевантные среди первых k рекомендаций
    """
    flags = np.isin(recommended_list[:k], bought_list)
    ranks = np.arange(1, k + 1)[flags][:n]      # ранги первых n рекомендаций из первых k. равен 0 если рекомендация нерелевантна
    ideal_ranks = np.arange(1, n + 1)
    return (1 / ranks).sum() / (1 / ideal_ranks).sum() if flags.any() else 0


def mean_reciprocal_rank_v1(recommended_list, bought_list, n=1, k=5):
    """ Среднеобратный ранг """
    ranks = []
    for data in zip(recommended_list, bought_list):
        ranks.append(reciprocal_rank(*data, k=k, n=n))
    return np.mean(ranks)


def mean_reciprocal_rank_v2(recommended_list, bought_list, n=1, k=5):
    """ Среднеобратный ранг - без for-loop """
    func = partial(reciprocal_rank, n=n, k=k)
    ranks = list(map(func, recommended_list, bought_list))
    return np.mean(ranks)


# def cumulative_gain(y_true, y_pred, k=3) -> float:
#     """ Cumulative gain at k """
#     _, argsort = torch.sort(y_pred, descending=True, dim=0)
#     return float(y_true[argsort[:k]].sum())


def compute_gain(y_value, gain_scheme: str):
    # vectorized
    if gain_scheme == "exp2":
        gain = 2 ** y_value - 1
    elif gain_scheme == "const":
        gain = y_value
    else:
        raise ValueError(f"{gain_scheme} method not supported, only exp2 and const.")
    return gain


def dcg(ys_pred, ys_true, gain_scheme: str = 'const', k=3) -> float:
    """ Discounted Cumulative Gain at K """
    argsort = np.argsort(np.array(ys_pred))[::-1]   # sort @k with numpy
    # _, argsort = torch.sort(torch.Tensor(ys_pred), descending=True, dim=0)      # the same with torch
    arg_mask = argsort < ys_true.size       # отсеиваем индексы, если они out of bounds
    ys_true_sorted = ys_true[argsort[arg_mask][:k]]

    gains = compute_gain(ys_true_sorted, gain_scheme)
    log_weights = np.log2(np.arange(k) + 2)
    return float((gains / log_weights).sum())

    # ret = 0
    # for idx, cur_y in enumerate(ys_true_sorted, 1):
    #     gain = compute_gain(cur_y, gain_scheme)
    #     ret += gain / (np.log2(idx + 1))
    # return float(ret)


def idcg(ys_true, gain_scheme: str = 'const', k=3) -> float:
    """ Ideal DCG at K """
    y_true_sorted, _ = torch.sort(ys_true, descending=True, dim=0)
    gains = compute_gain(y_true_sorted[:k], gain_scheme)        
    log_weights = np.log2(np.arange(k) + 2)
    return float((gains / log_weights).sum())


def ndcg(ys_pred, ys_true, gain_scheme: str = 'const', k=3) -> float:
    """ Normalized Discounted Cumulative Gain at K """
    pred_dcg = dcg(ys_pred, ys_true, gain_scheme, k)
    ideal_dcg = dcg(ys_true, ys_true, gain_scheme, k)
    return pred_dcg / ideal_dcg


# useful links
# 1. https://habr.com/ru/company/econtenta/blog/303458/
