import numpy as np
import pandas as pd
from functools import partial

def precision(true_values, pred_values, **kwargs):
    flags = np.isin(pred_values, true_values)
    return np.sum(flags) / len(pred_values)

def recall(true_values, pred_values, **kwargs):
    flags = np.isin(pred_values, true_values)
    return np.sum(flags) / len(true_values)

def precision_at_k(true_values, pred_values, k=5):
    flags = np.isin(pred_values[:k], true_values)
    return np.sum(flags) / k

def recall_at_k(true_values, pred_values, k=5):
    flags = np.isin(pred_values[:k], true_values)
    return np.sum(flags) / len(true_values)

def ap_k(true_values, pred_values, k=5):
    """ Average precision at k items """
    flags = np.isin(true_values, pred_values)
    if np.sum(flags) == 0:
        return 0

    func = partial(precision_at_k, true_values, pred_values)
    rel_items = np.arange(1, k + 1)[flags[:k]]
    return np.sum(list(map(func, rel_items))) / np.sum(flags)

def calc_mean_metric(metric_func, true_values, pred_values, k=5):
    """ Serialize metric calculations
    :param metric_func: metric function like metric_func(true_values, pred_values)
    :param true_values: actual true values
    :param pred_values: actual predictions
    """
    def metric_wrapper(pred, act):
        return metric_func(pred, act, k=k) if len(pred) != 0 else 0

    if isinstance(pred_values, pd.DataFrame):
        metric = pd.DataFrame(columns=pred_values.columns)
        for col in pred_values.columns:
            metric[col] = pd.concat([true_values, pred_values[col]], axis=1).apply(lambda row: metric_wrapper(*row.values), axis=1)
    else:
        metric = pd.concat([true_values, pred_values], axis=1).apply(lambda row: metric_wrapper(*row.values), axis=1)
    return metric.mean()
