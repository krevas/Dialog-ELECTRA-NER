import random

import torch
import numpy as np
from numpy.lib.function_base import average
from scipy.stats import pearsonr, spearmanr
from seqeval import metrics as seqeval_metrics
from sklearn import metrics as sklearn_metrics


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def simple_accuracy(labels, preds):
    return (labels == preds).mean()


def acc_score(labels, preds):
    return {
        "acc": simple_accuracy(labels, preds),
    }


def pearson_and_spearman(labels, preds):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def f1_pre_rec(labels, preds, is_ner=True):
    if is_ner:
        return {
            "precision": seqeval_metrics.precision_score(labels, preds, suffix=True),
            "recall": seqeval_metrics.recall_score(labels, preds, suffix=True),
            "f1": seqeval_metrics.f1_score(labels, preds, suffix=True),
        }
    else:
        return {
            "precision": sklearn_metrics.precision_score(labels, preds, average="macro"),
            "recall": sklearn_metrics.recall_score(labels, preds, average="macro"),
            "f1": sklearn_metrics.f1_score(labels, preds, average="macro"),
        }


def show_ner_report(labels, preds):
    return seqeval_metrics.classification_report(labels, preds, suffix=True)

def token_check(token,space,tag):
    check = False
    index = 0
    if len(token) == 4 and token[:2] == '##' and token[-1] in ['의','에','이','와',
                                                             '과','은','가','부','를']:
        if token[2::] not in ['본부','정부','평가']:
            check = True
        if token in ['##회의','##서부','##아이']and space == 0:
            check = False
    elif token in ['시에','##지역인','시의']:
        check = True
    if space == 1:
        index = 2
    else:
        index = 1
    return (check, index)

def compute_metrics(task_name, labels, preds):
    assert len(preds) == len(labels)
    if task_name == "ner":
        return f1_pre_rec(labels, preds, is_ner=True), "f1"
    else:
        raise KeyError(task_name)
