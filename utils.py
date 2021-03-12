import random

import torch
import numpy as np
from numpy.lib.function_base import average
from seqeval import metrics as seqeval_metrics
from sklearn import metrics as sklearn_metrics
from sklearn.preprocessing import LabelBinarizer
from itertools import chain
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


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

def plot_confusion_matrix(cm, classes,
        normalize=False,
        title='Confusion matrix',
        cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Answer label')
    plt.xlabel('Predict label')

    np.set_printoptions(precision=2)  

def bio_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))

    tagset = set(lb.classes_)
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])

    def isO(y):
        if y == 'O':
            return False
        else:
            return True
    y_true = list(chain.from_iterable(y_true))
    y_pred = list(chain.from_iterable(y_pred))
    
    y_true_rev, y_pred_rev = [], []
    for true, pred in zip(y_true, y_pred):
        if true != 'O' and pred != 'O':
            y_true_rev.append(true)
            y_pred_rev.append(pred)
    tagset.remove('O')
    cnf_matrix = confusion_matrix(y_true_rev, y_pred_rev, labels=tagset)
    
    plt.rcParams["figure.figsize"] = (10,10)
    plt.figure()
    plot_confusion_matrix(cnf_matrix,classes=tagset,normalize=False,
                                   title=title)
    plt.savefig('fig_{}.png'.format(title), dpi=300)

def bio_classification_report(y_true, y_pred):
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

    tagset = set(lb.classes_) - {'O'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

    return sklearn_metrics.classification_report(
            y_true_combined,
            y_pred_combined,
            labels = [class_indices[cls] for cls in tagset],
            target_names = tagset,
            )

def token_check(token,space,tag):
    check = False
    index = 0
    if len(token) == 4 and token[:2] == '##' and token[-1] in ['의','에','이','와',
                                                             '과','은','가','부','를']:
        if token[2::] not in ['본부','정부','평가','베이','웨이']:
            check = True
        if token in ['##레이','##라이','##서부','##케이','##사이','##에이','##바이']:
            check = False
        if token in ['##회의','##아이','##하이',
                     '##베이','##파이','##북부']and space == 0:
            check = False
    elif token in ['시에','##지역인','시의',
                   '##인양','##주로','전인',
                   '##지로','경찰의','##씨도',
                   '만이','이모','달도',
                   '##부로','##일대','##비로',
                   '후인','이상의']:
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
