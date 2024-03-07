import numpy as np


def accuracy(y_true, y_pred):
    acc = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return acc
