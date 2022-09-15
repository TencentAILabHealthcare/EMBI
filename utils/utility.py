# -*- coding: utf-8 -*-

import json
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import numpy as np
import time
from functools import wraps


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

def encode_seq(sequence):
    alphabet = ['A', 'C', 'D', 'E', 'F', 'G','H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    integer_encoded = [char_to_int[char] for char in sequence]
    onehot_encoded = list()

    for value in integer_encoded:
        letter = [0 for _ in range(len(alphabet))]
        letter[value] = 1
        onehot_encoded.append(letter)

    return np.array(onehot_encoded)  

def encode_24(seq):
    left = encode_seq(seq[:12])
    right = encode_seq(seq[-12:])
    if len(seq) < 12:
        middle = np.zeros((24-len(seq) * 2, 20), dtype='int32')
        merge = np.concatenate((left, middle, right), axis=0)
    else:
        merge = np.concatenate((left, right), axis=0)
    return merge   

def encode_24_blosum50(seq, blosum50_dict):
    left = encode_seq_blosum50(seq[:12], blosum50_dict)
    right = encode_seq_blosum50(seq[-12:],blosum50_dict)
    if len(seq) < 12:
        middle = np.zeros((24-len(seq) * 2, 20), dtype='int32')
        merge = np.concatenate((left, middle, right), axis=0)
    else:
        merge = np.concatenate((left, right), axis=0)
    return merge

def encode_seq_blosum50(seq, blosum50_dict):
    alphabet = ['A', 'C', 'D', 'E', 'F', 'G','H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    result = np.zeros((len(seq), len(alphabet)), dtype='int32')
    for i,s in enumerate(list(seq)):
        for j, a in enumerate(alphabet):
            blosum50 = blosum50_dict[s+a]
            result[i][j] = blosum50
    return result

def getMHCSeqDict():
    MHC_pesudo_seq = pd.read_csv('/aaa/louisyuzhao/project2/immuneDataSet/yixinguo/Data/NetMHCpan_train/MHC_pseudo.dat', 
                                header=None, sep = '\s+')
    MHC_pesudo_seq.columns = ['HLA','Seq']
    MHC_pesudo_seq = MHC_pesudo_seq.drop_duplicates()
    MHC_pesudo_seq_dict = dict(zip(MHC_pesudo_seq.HLA.to_list(), MHC_pesudo_seq.Seq.to_list()))
    return MHC_pesudo_seq_dict

def timefn(fn):
    '''compute time cost'''
    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        print(f"@timefn: {fn.__name__} took {t2 - t1: .5f} s")
        return result
    return measure_time

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
