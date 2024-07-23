from cProfile import label
from colorsys import yiq_to_rgb
import itertools
from tkinter.tix import Tree
import numpy as np
from sklearn import metrics

import json
import pandas as pd


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
    if type(y_pred) != np.ndarray:
        y_pred = np.array(y_pred)
    return metrics.accuracy_score(y_pred=y_pred.reshape(y_pred.size), y_true=y_true.reshape(y_true.size), normalize=True, sample_weight=None)


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
    # print(type(y_pred), type(y_true),y_true)
    if not isinstance(y_true, np.ndarray):
        y_pred = np.array(list(y_pred))
    try:
        y_true = np.asarray(y_true)
        len_of_y_true = len(np.asarray(y_true)) if y_true.ndim != 0 else 1
    except Exception as e:
        print(f"Error converting y_true to array: {e}")
        len_of_y_true = 1
    # len_of_y_true = len(y_true) if y_true.size != () else 1
    return (y_pred == y_true).sum(), len_of_y_true

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
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    return precision, recall

def calculate_AUC(probability, target):
    fpr, tpr, thresholds = metrics.roc_curve(y_score= np.array(probability), y_true= np.array(target), pos_label=1)
    AUC = metrics.auc(fpr, tpr)
    return AUC

def calculate_AUPRC(probability, target):
    precision, recall, thresholds = metrics.precision_recall_curve(probas_pred= np.array(probability), y_true= np.array(target), pos_label=1)
    # print('recall', recall)
    # print('precision',precision)
    AUPRC = metrics.auc(recall, precision)
    average_precision = metrics.average_precision_score(y_true=np.array(target), y_score=np.array(probability))
    return AUPRC

def calculateMCC(y_pred,y_true):
    return metrics.matthews_corrcoef(y_true, y_pred)


def roc_auc(y_pred, y_true):
    """
    The values of y_pred can be decimicals, within 0 and 1.
    Args:
        y_pred ([type]): [description]
        y_true ([type]): [description]
    Returns:
        [type]: [description]
    """
    return metrics.roc_auc_score(y_score=y_pred, y_true=y_true)


class MAA_metrics(object):

    def __init__(self, token_with_special_list, blosum_dir, blosum=False):
        self.token_with_special_list = token_with_special_list
        self.blosum_dir = blosum_dir
        self.blosum = blosum
        
        self.AMINO_ACIDS = "ACDEFGHIKLMNPQRSTUVWY"
        self.BLOSUM = self._load_blosum()

    def _load_blosum(self):

        with open(self.blosum_dir) as source:
            d = json.load(source)
            retval = pd.DataFrame(d)
        retval = pd.DataFrame(0, index=list(self.AMINO_ACIDS), columns=list(self.AMINO_ACIDS))

        for x,y in itertools.product(retval.index, retval.columns):
            if x == "U" or y == "U":
                continue
            retval.loc[x,y] = d[x][y]
        retval.drop(index='U', inplace=True)
        retval.drop(columns='U', inplace=True)
        return retval
    
    def compute_metrics(self, pred, top_n=3):
        # print('pred',pred)
        labels = pred.label_ids.squeeze() # shape (n,32)
        preds = pred.predictions # shape (n, 32, 26)

        n_mask_total = 0 
        top_one_correct, top_n_correct = 0, 0 
        blosum_values = []

        for i in range(labels.shape[0]):
            masked_idx = np.where(labels[i] != -100)[0]
            n_mask = len(masked_idx)
            n_mask_total += n_mask
            pre_arr = preds[i, masked_idx]
            truth = labels[i, masked_idx] # The masked token indices
            # argsort returns indices in ASCENDING order
            pred_sort_idx = np.argsort(pre_arr, axis=1)
            top_one_correct += np.sum(truth == pred_sort_idx[:, -1])
            top_n_preds = pred_sort_idx[:, -top_n:]

            for truth_idx, top_n_idx in zip(truth, top_n_preds):
                top_n_correct += truth_idx in top_n_idx
                # check blosum score
                if self.blosum:
                    truth_res = self.token_with_special_list[truth_idx]
                    pred_res = self.token_with_special_list[top_n_idx[-1]]
                    for aa_idx in range(min(len(truth_res), len(pred_res))):
                        if truth_res[aa_idx] in self.BLOSUM.index and pred_res[aa_idx] in self.BLOSUM.index:
                            blosum_values.append(self.BLOSUM.loc[truth_res[aa_idx], pred_res[aa_idx]])
      
        assert top_one_correct <= top_n_correct <= n_mask_total
        if self.blosum:
            retval = {
                f"top_{top_n}_acc": top_n_correct / n_mask_total,
                "acc": top_one_correct / n_mask_total,
                "average_blosum": np.mean(blosum_values)}
        else:
            retval = {
                f"top_{top_n}_acc": top_n_correct / n_mask_total,
                "acc": top_one_correct / n_mask_total} 
        return retval





