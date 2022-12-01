from os.path import join
from pickletools import read_uint1
# from tkinter import _Padding
from matplotlib import dates
from matplotlib.pyplot import axis
from numpy import dtype
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import ESMTokenizer
from data.base_dataloader import BaseDataLoader
import pandas as pd
import numpy as np
import json
import urllib
import ssl
import os
import random
# import blosum as bl


class EpitopeMHCTCRDataset(Dataset):
    def __init__(self, original_data, tokenizer, max_seq_length) -> None:
        self.epitope = original_data[0]
        self.MHC = original_data[1]
        self.binder = original_data[2]
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
    
    def __len__(self):
        return len(self.epitope)

    def __getitem__(self, index):
        epitope = self.epitope[index]
        # print('epitope',epitope)
        MHC = self.MHC[index]
        binder = torch.tensor(self.binder[index], dtype=torch.float32)
        # print('immunogenic',immunogenic)
        encoded_epitope = self.tokenizer(
            epitope,
            truncation="only_first",
            max_length = self.max_seq_length,
            padding="max_length",
            return_tensors="pt"
        )
        # # print('encoded_epitope',encoded_epitope)

        encoded_MHC = self.tokenizer(
            MHC,
            truncation="only_first",
            max_length = self.max_seq_length,
            padding="max_length",
            return_tensors="pt"
        )
        epitope_epitope_MHC_inputs = torch.concat((encoded_epitope['input_ids'],encoded_MHC['input_ids']),
                                        axis=1)
        
        epitope_epitope_MHC_attention_mask = torch.concat((encoded_epitope['attention_mask'],encoded_MHC['attention_mask']),
                                        axis=1)
        return epitope_epitope_MHC_inputs, epitope_epitope_MHC_attention_mask, binder                               

class EpitopeMHCTCRDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size,logger, 
                seed, sampler_type, shuffle, validation_split, test_split, 
                num_workers=1, max_seq_length=34):
        self.data_dir = data_dir
        self.max_seq_length = max_seq_length
        self.logger = logger

        self.logger.info('Load pretrained tokenizer from ESM-1b ...')
        self.tokenizer = ESMTokenizer.from_pretrained("facebook/esm-1b", do_lower_case=False, local_files_only=True)

        self.epitope_BA_df, self.BA_testdata = self._load_data()
        self.train_dataset = self._prepare_dataset(self.epitope_BA_df)
        self.logger.info('Load train dataset successfully')
        super().__init__(self.train_dataset, batch_size, seed, validation_split, test_split, shuffle=shuffle, sampler_type=sampler_type, num_workers=num_workers)
        
        ### test_dataset
        self.test_dataset = self._prepare_testdata(self.BA_testdata)
        self.test_dataloader = DataLoader(dataset=self.test_dataset, batch_size=batch_size, shuffle=shuffle)

    def get_test_dataloader(self):
        self.logger.info('Number of test data {}'.format(len(self.test_dataset)))
        return self.test_dataloader

    def _load_data(self):
        # ap training data 20220826_EL_Training_data_from_netMHCpan.csv
        # ap test data 20220914_IEDB_data_for_test.csv
        # immu training data 20220916_PRIME_data_for_training.csv
        # immu test data 20220916_PRIME_data_for_test.csv

        Epitope_BA_df = pd.read_csv(join(self.data_dir, '20220826_EL_Training_data_from_netMHCpan.csv'),dtype=str)
        Epitope_BA_df_test = pd.read_csv(join(self.data_dir, '20220914_IEDB_data_for_test.csv'),dtype=str)
        return Epitope_BA_df, Epitope_BA_df_test
    
    def _process_BA_df(self, Epitope_BA_df):

        Epitope_BA_df_random = Epitope_BA_df
        data_x_epitopes = Epitope_BA_df_random['Epitope'].to_list()
        data_MHC_pseduo_seq = Epitope_BA_df_random['HLA_pseudo_seq'].to_list()
        data_Binder = [int(i) for i in Epitope_BA_df_random['Binder'].to_list()]
        return data_x_epitopes, data_MHC_pseduo_seq, data_Binder
    
    def _prepare_testdata(self, test_df):
        original_data = self._process_BA_df(test_df)
        dataset = EpitopeMHCTCRDataset(original_data, tokenizer=self.tokenizer, max_seq_length=self.max_seq_length)
        return dataset

    def _prepare_dataset(self, epitope_BA_df):
        Epitope_BA_dataset = self._process_BA_df(epitope_BA_df)
        dataset = EpitopeMHCTCRDataset(Epitope_BA_dataset, tokenizer=self.tokenizer, max_seq_length=self.max_seq_length)
        return dataset


    def _allele2seq(self, allele_name):
        allele_seq = ''
        
        allele_tmp = []
        if isinstance(allele_name, str):
            tmp = allele_name.split(', ')
            if tmp[0].startswith('HLA-'):
                allele_tmp.append(tmp[0].split('-')[1])
        else:
            allele_seq = ''

        for i in allele_tmp:
            if isinstance(i,str):
                if '/' in i:
                    i = i.split('/')[0]+":01:01"
                elif i.endswith(':01:01:01'):
                    i = i[:-3] 
                elif ' ' in i:
                    i = i.split(' ')[0] + ":01:01"
                else:
                    i = i + ":01:01"
                tmp = self.Allele_data.loc[self.Allele_data['Allele'] == i,'AlleleID']
                if tmp.shape[0] == 0:
                    tmp = []
                else:
                    tmp = tmp.iloc[0]
            else:
                tmp =[]
            if tmp != []:
                allele_seq = self._query_seq(tmp)
            else:
                allele_seq = ''
        return allele_seq

    def _generate_allele_seq_PRIME(self, PRIME_df):
        epitope_from_PRIME_allele_HLA_dict = {}
        epitope_from_PRIME_allele = list(set(PRIME_df['Allele'].to_list()))
        Allele_data_allele = self.Allele_data['Allele'].to_list()
        for i in epitope_from_PRIME_allele:
            tmp = i[0] + '*' +i[1:3] + ':' + i[-2:]
            tmp_0101= tmp + ':01:01'
            tmp_01 = tmp + ':01'
            if tmp in Allele_data_allele:
                hla_accession = self.Allele_data.loc[self.Allele_data['Allele'] == tmp,'AlleleID'].iloc[0]
            elif tmp_0101 in Allele_data_allele:
                hla_accession = self.Allele_data.loc[self.Allele_data['Allele'] == tmp_0101,'AlleleID'].iloc[0]
            elif tmp_01 in Allele_data_allele:
                hla_accession = self.Allele_data.loc[self.Allele_data['Allele'] == tmp_01,'AlleleID'].iloc[0]
            else:
                hla_accession = ''
            
            if hla_accession != '':
                allele_seq = self._query_seq(hla_accession)
            else:
                allele_seq = ''
            if i not in epitope_from_PRIME_allele_HLA_dict:
                epitope_from_PRIME_allele_HLA_dict[i] = allele_seq
        return epitope_from_PRIME_allele_HLA_dict    

    def _generate_allele_seq(self, iedb_df):
        result_dict = {}
        for i in iedb_df['MHC Allele Names'].to_list():
            if i not in result_dict:
                result_dict[i] = self._allele2seq(i)
        return result_dict


    
    def _query_seq(self, hla):
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        imgt_hla_api = 'https://www.ebi.ac.uk/cgi-bin/ipd/api/allele/'
        response = urllib.request.urlopen(imgt_hla_api + hla)
        allele_seq = json.loads(response.read().decode('utf-8'))['sequence']['protein']
        return allele_seq


    def _assert_aa(self, s='X123'):
        s = s.strip()
        if s >= 'A' and s <= 'Z':
            remainder = s[1:]
            return all([i <= '9' and i >= '0' for i in remainder])
        else:
            return False

