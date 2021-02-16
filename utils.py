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

def token_check(token,tag):
    check = False
    index = 0
    if token in ['##준이','##서가','##숙이',
                 '##정은','##홍의','##명이',
                 '##훈의','##수와','##상의'] and tag == 'PS':
        check = True
        index = 2
    elif token in ['##대가','##간의','##시에',
                   '##단의','##여의','##리의',
                   '프랑스와','##위의','##역의',
                   '##역과','##부와','##부를',
                   '##도와','##원이','##오의',
                   '##z의','##일의']:
        check = True
        index = 2
    elif token in ['##일까','시에']:
        check = True
        index = 1
    return (check, index)

def compute_metrics(task_name, labels, preds):
    assert len(preds) == len(labels)
    if task_name == "ner":
        return f1_pre_rec(labels, preds, is_ner=True), "f1"
    else:
        raise KeyError(task_name)
