import numpy as np
from sklearn import metrics


def accuracy_sample(y_pred, y_true):
    """Compute the accuracy for each sample

    Note that the input y_pred and y_true are both numpy array.
    Args:
        y_pred (numpy.array): shape [seq_len*2, batch_size, ntoken]
        y_true (numpy.array): shape [seq_len*2, batch_size]
    """
    y_pred = y_pred.argmax(axis=2)
    print('Shape of y_pred:', y_pred.shape)
    print('Shape of y_true:', y_true.shape)
    return metrics.accuracy_score(y_pred=y_pred, y_true=y_true)

def accuracy_amino_acid(y_pred, y_true):
    '''Compute teh accuracy for each amino acid.

    Args:
        y_pred (numpy.array): shape [batch_size, seq_len, ntoken]
        y_true (numpy.array): shape [batch_size, seq_len]
    '''
    y_pred = y_pred.argmax(axis=2)
    return metrics.accuracy_score(y_pred=y_pred.flatten(), y_true=y_true.flatten())

    
def correct_count(y_pred, y_true):
    '''Count the correct prediction for each amino acid.

    Args:
        y_pred (numpy.array): shape [batch_size, seq_len, ntoken]
        y_true (numpy.array): shape [batch_size, seq_len]
    '''
    y_pred = y_pred.argmax(axis=2)
    y_true_copy = y_true.copy()
    y_true_copy[y_true_copy==21] = -100
    return (y_pred == y_true_copy).sum(), np.count_nonzero(y_true_copy != -100)