from cmath import log
from os.path import join
from pickletools import read_uint1
from matplotlib import dates
from matplotlib.pyplot import axis
from numpy import dtype
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import ESMTokenizer
from bert_data_prepare.tokenizer import get_tokenizer
from data.base_dataloader import BaseDataLoader
import pandas as pd
import numpy as np


class EpitopeMHCBertDataset(Dataset):
    def __init__(self, original_data, 
                        epitope_tokenizer, 
                        MHC_tokenizer, 
                        epitope_split_fun,
                        MHC_split_fun,
                        epitope_max_seq_length, 
                        MHC_max_seq_length,
                        logger):
        self.epitope = original_data[0]
        self.MHC = original_data[1]
        # self.chain1_cdr3 = original_data[2]
        # self.chain2_cdr3 = original_data[3]
        # self.immunogenic = original_data[1]
        self.binder = original_data[2]
        self.epitope_tokenizer = epitope_tokenizer
        self.MHC_tokenizer = MHC_tokenizer
        self.epitope_split_fun = epitope_split_fun
        self.MHC_split_fun = MHC_split_fun
        self.epitope_max_seq_length = epitope_max_seq_length
        self.MHC_max_seq_length = MHC_max_seq_length
        self.logger = logger
        self._has_logged_example = False

    
    def __len__(self):
        return len(self.epitope)

    def __getitem__(self, index):
        epitope = self.epitope[index]
        # print('epitope',epitope)
        MHC = self.MHC[index]
        # print('MHC',MHC)
        # chain1_cdr3 = self.chain1_cdr3[index]
        # chain2_cdr3 = self.chain2_cdr3[index]
        # print('chain1_cdr3',chain1_cdr3)
        # print('chain_cdr3',chain1_cdr3)
        # immunogenic = torch.tensor(self.immunogenic[index], dtype=torch.float32)
        binder = self.binder[index]
        binder_tensor = torch.tensor(self.binder[index], dtype=torch.float32)
        # print('immunogenic',immunogenic)
        epitope_tensor = self.epitope_tokenizer(
            self._insert_whitespace(self.epitope_split_fun(epitope)),
            truncation=True,
            max_length = self.epitope_max_seq_length,
            padding="max_length",
            return_tensors="pt"
        )
        # # print('encoded_epitope',encoded_epitope)

        MHC_tensor = self.MHC_tokenizer(
            self._insert_whitespace(self.MHC_split_fun(MHC)),
            truncation=True,
            max_length = self.MHC_max_seq_length,
            padding="max_length",
            return_tensors="pt"
        )
        epitope_tensor = {k:torch.squeeze(v) for k,v in epitope_tensor.items()}
        MHC_tensor = {k:torch.squeeze(v) for k,v in MHC_tensor.items()}

        if not self._has_logged_example:
            self.logger.info(f"Example of tokenized epitope: {epitope} -> {epitope_tensor}")
            self.logger.info(f"Example of tokenized MHC: {MHC} -> {MHC_tensor}")
            self.logger.info(f"Example of label: {binder} -> {binder_tensor}")
            self._has_logged_example = True

        return epitope_tensor, MHC_tensor, binder_tensor                              

    def _insert_whitespace(self, token_list):
        """
        Return the sequence of tokens with whitespace after each char
        """
        return " ".join(token_list)

# def loadData(data, batch=300, n_workers=12, shuffle=True):
#     return DataLoader(data, batch_size=batch, shuffle=shuffle, num_workers=n_workers)

class EpitopeMHCBertDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size,logger, 
                seed, validation_split, test_split, 
                epitope_vocab_dir, MHC_vocab_dir,  
                epitope_tokenizer_dir, MHC_tokenizer_dir, 
                epitope_seq_name='epitope',
                MHC_seq_name='MHC',
                epitope_max_seq_length=None,
                MHC_max_seq_length=None,
                epitope_tokenizer_name="common",
                MHC_tokenizer_name="common",
                shuffle=True,
                num_workers=1,
                ):
        self.data_dir = data_dir
        self.epitope_vocab_dir = epitope_vocab_dir
        self.MHC_vocab_dir = MHC_vocab_dir
        self.epitope_tokenizer_dir = epitope_tokenizer_dir
        self.MHC_tokenizer_dir = MHC_tokenizer_dir
        self.epitope_seq_name = epitope_seq_name
        self.MHC_seq_name = MHC_seq_name
        self.epitope_max_seq_length = epitope_max_seq_length
        self.MHC_max_seq_length = MHC_max_seq_length
        self.logger = logger
        self.shuffle = shuffle

        self.logger.info('Load pretrained tokenizer from EpitopeBert and MHCBert ...')

        self.logger.info(f'Creating {epitope_seq_name} tokenizer...')
        self.EpitopeTokenizer = get_tokenizer(
            tokenizer_name=epitope_tokenizer_name, 
            logger = self.logger
        )
        self.epitope_tokenizer = self.EpitopeTokenizer.get_bert_tokenizer(
            max_len=self.epitope_max_seq_length,
            tokenizer_dir=epitope_tokenizer_dir
        )
        self.logger.info(f'Creating {MHC_seq_name} tokenizer...')
        self.MHCTokenizer = get_tokenizer(
            tokenizer_name=MHC_tokenizer_name, 
            logger = self.logger
        )
        self.MHC_tokenizer = self.MHCTokenizer.get_bert_tokenizer(
            max_len=self.MHC_max_seq_length,
            tokenizer_dir=self.MHC_tokenizer_dir
        )
        # self.iedb_df, self.PRIME_df, self.Allele_data, self.VDJdb_df = self._load_data()
        self.epitope_BA_df, self.test_dataset = self._load_data()

        # self.logger.info('IEDB, PIRME data load successfully')
        # self.iedb_allele_hla_dict, self.PRIME_allele_hla_dict, self.VDJdb_allele_hla_dict = self._load_allele_hla_seq()
        # self.cdr3b_healthy = self._load_CDR3b_from_TCRdb(self.iedb_df)
        # self.logger.info('healthy CDR3b extracted completely')

        self.train_dataset = self._prepare_dataset(self.epitope_BA_df)
        # print('train_dataset', self.train_dataset.shape)
        self.logger.info('Load train dataset successfully')
        super().__init__(self.train_dataset, batch_size, seed, shuffle, validation_split, test_split, num_workers)
        
        ### test_dataset
        # self.test_dataset = self.test_df
        self.test_dataloader = DataLoader(dataset=self.test_dataset, batch_size=batch_size, shuffle=shuffle)

    def get_test_dataloader(self):
        self.logger.info('Number of test data {}'.format(len(self.test_dataset)))
        return self.test_dataloader

    def _load_data(self):
        Epitope_BA_df = pd.read_csv(join(self.data_dir, 'Epitope_BA_data.csv'),dtype=str)
        test_df = pd.read_csv(join(self.data_dir, '20220825benchmark_data_from_DeepNetBim.csv'),dtype=str)
        return Epitope_BA_df, test_df
    
    def _process_BA_df(self, Epitope_BA_df):

        Epitope_BA_df_random = Epitope_BA_df
        data_x_epitopes = Epitope_BA_df_random['Epitope'].to_list()
        data_MHC_pseduo_seq = Epitope_BA_df_random['HLA_pseudo_seq'].to_list()
        data_Binder = [int(i) for i in Epitope_BA_df_random['Binder'].to_list()]
        return data_x_epitopes, data_MHC_pseduo_seq, data_Binder
        
    def get_test_dataloader(self):
        self.logger.info('Number of test data {}'.format(len(self.test_dataset)))
        return self.test_dataloader
    # def _prepare_testdata(self, test_df):
    #     original_data = self._process_BA_df(test_df)
    #     dataset = EpitopeMHCBertDataset(
    #         original_data, 
    #         epitope_split_fun=self.EpitopeTokenizer.split,
    #         MHC_split_fun=self.MHCTokenizer.split,
    #         epitope_tokenizer=self.epitope_tokenizer, 
    #         MHC_tokenizer=self.MHC_tokenizer,
    #         epitope_max_seq_length=self.epitope_max_seq_length,
    #         MHC_max_seq_length=self.MHC_max_seq_length,
    #         logger=self.logger
    #     )
    #     return dataset

    def _prepare_dataset(self, epitope_BA_df):
             
        Epitope_BA_dataset = self._process_BA_df(epitope_BA_df)

        dataset = EpitopeMHCBertDataset(
            Epitope_BA_dataset, 
            epitope_split_fun=self.EpitopeTokenizer.split,
            MHC_split_fun=self.MHCTokenizer.split,
            epitope_tokenizer=self.epitope_tokenizer, 
            MHC_tokenizer=self.MHC_tokenizer,
            epitope_max_seq_length=self.epitope_max_seq_length,
            MHC_max_seq_length=self.MHC_max_seq_length,
            logger=self.logger
        )
        return dataset
    

    def _assert_aa(self, s='X123'):
        s = s.strip()
        if s >= 'A' and s <= 'Z':
            remainder = s[1:]
            return all([i <= '9' and i >= '0' for i in remainder])
        else:
            return False
