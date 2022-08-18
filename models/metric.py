from colorsys import yiq_to_rgb
import numpy as np
from sklearn import metrics


def accuracy_sample(y_pred, y_true):
    """Compute the accuracy for each sample

    Note that the input y_pred and y_true are both numpy array.
    Args:
        y_pred (numpy.array): shape [seq_len*2, batch_size, ntoken]
        y_true (numpy.array): shape [seq_len*2, batch_size]
    """
    # y_pred = y_pred.argmax(axis=2)
    # print('y_pred:', y_pred)
    # print('y_true:', y_true)
    return metrics.accuracy_score(y_pred=y_pred, y_true=y_true, normalize=True, sample_weight=None)


def accuracy_amino_acid(y_pred, y_true):
    '''Compute teh accuracy for each amino acid.

    Args:
        y_pred (numpy.array): shape [batch_size, seq_len, ntoken]
        y_true (numpy.array): shape [batch_size, seq_len]
    '''
    y_pred = y_pred.argmax(axis=2)
    print('y_pred__',y_pred.shape)
    return metrics.accuracy_score(y_pred=y_pred.flatten(), y_true=y_true.flatten())


    
def correct_count(y_pred, y_true):
    '''Count the correct prediction for each amino acid.

    Args:
        y_pred (numpy.array): shape [batch_size, seq_len, ntoken]
        y_true (numpy.array): shape [batch_size, seq_len]
    '''
    return (y_pred == y_true).sum(), len(y_true)

def correct_predictions(output_probabilities, targets):

    out_classes = output_probabilities.ge(0.5).byte().float()
    correct = (out_classes == targets).sum()
    return correct.item()


def calculatePR(y_pred, y_true):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    TP = np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 1)))
    FP = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 1)))
    TN = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 0)))
    FN = np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 0)))
    # precision = TP/(TP+FP)
    # recall = TP/(TP+FN)
    return TP, FP, TN, FN   