
from os import posix_spawn
from random import shuffle
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
import numpy as np
import torch

class BaseDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, seed, validation_split,  
                 test_split, shuffle, sampler_type, num_workers, collate_fn=default_collate):
        self.validation_split = validation_split
        # print('validation split',validation_split)
        self.test_split = test_split
        self.seed = seed
        self.dataset = dataset
        # print('datset',dataset)
        self.batch_idx = 0
        self.n_samples = len(dataset)
        print('n_samples',self.n_samples)
        if sampler_type == 'balanced':
            self.sampler, self.valid_sampler, self.test_sampler = self._split_sampler_traing_balanced()
            self.init_kwargs = {
                'dataset': dataset,
                'batch_size': batch_size,
                # 'shuffle': self.shuffle,
                'collate_fn': collate_fn,
                'num_workers': num_workers
            }
        else:
            self.shuffle = shuffle
            self.sampler, self.valid_sampler, self.test_sampler = self._split_sampler()
            self.init_kwargs = {
                'dataset': dataset,
                'batch_size': batch_size,
                'shuffle': self.shuffle,
                'collate_fn': collate_fn,
                'num_workers': num_workers
            }

        super().__init__(sampler=self.sampler, **self.init_kwargs)
    
    def _split_sampler(self):
        idx_full = np.arange(self.n_samples)

        np.random.seed(self.seed)
        np.random.shuffle(idx_full)

        if isinstance(self.validation_split, int) or isinstance(self.test_split, int):
            assert self.validation_split > 0 or self.test_split > 0
            assert self.validation_split < self.n_samples or self.test_split < self.n_samples, \
                "validation set size or test set size is configured to be larger than entire dataset."
            len_valid = self.validation_split
            len_test = self.test_split
        else:
            len_valid = int(self.n_samples * self.validation_split)
            len_test = int(self.n_samples * self.test_split)
        valid_idx = idx_full[0:len_valid]
        test_idx = idx_full[len_valid:(len_valid+len_test)]
        train_idx = np.delete(idx_full, np.arange(0, len_valid+len_test))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        self.shuffle = False
        self.n_samples = len(train_idx)
        return train_sampler, valid_sampler, test_sampler
    
    def _split_sampler_traing_balanced(self):
        idx_full = np.arange(self.n_samples)

        np.random.seed(self.seed)
        np.random.shuffle(idx_full)

        if isinstance(self.validation_split, int) or isinstance(self.test_split, int):
            assert self.validation_split > 0 or self.test_split > 0
            assert self.validation_split < self.n_samples or self.test_split < self.n_samples, \
                "validation set size or test set size is configured to be larger than entire dataset."
            len_valid = self.validation_split
            len_test = self.test_split
        else:
            len_valid = int(self.n_samples * self.validation_split)
            len_test = int(self.n_samples * self.test_split)
        print('len_valid',len_valid)
        valid_idx = idx_full[0:len_valid]
        test_idx = idx_full[len_valid:(len_valid+len_test)]
        # train_idx = np.delete(idx_full, np.arange(0, len_valid+len_test))
        # print('dataset_tensor',self.dataset.source[])

        source = np.array(self.dataset.source)
        class_sample_count = np.array(
            [len(np.where(source == t)[0]) for t in np.unique(source)])

        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in source])
        samples_weight[valid_idx] = 0.0
        samples_weight[test_idx] = 0.0

        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        
        train_sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

        valid_sampler = SubsetRandomSampler(valid_idx)
        test_sampler = SubsetRandomSampler(test_idx)
        return train_sampler, valid_sampler, test_sampler

        
    def split_dataset(self, valid=False, test=False):
        if valid:
            assert len(self.valid_sampler) >= 0, "validation set size ratio is not positive"
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs) if len(self.valid_sampler) != 0 else None
        if test:
            assert len(self.test_sampler) >= 0, "test set size ratio is not positive"
            return DataLoader(sampler=self.test_sampler, **self.init_kwargs) if len(self.test_sampler) != 0 else None
    

    
 
        


