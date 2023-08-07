from os.path import join
import torch
from torch.utils.data import Dataset, DataLoader
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
        self.binder = original_data[2]
        # self.source = original_data[3]
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
        MHC = self.MHC[index]
        binder = self.binder[index]
        # source = self.source[index]
        binder_tensor = torch.tensor(self.binder[index], dtype=torch.float32)
        # source_tensor = torch.tensor(source, dtype=torch.float32)
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


class EpitopeMHCBertDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size,logger, 
                seed, validation_split, test_split, 
                epitope_vocab_dir, MHC_vocab_dir,  
                epitope_tokenizer_dir, MHC_tokenizer_dir, 
                sampler_type,
                epitope_seq_name='epitope',
                MHC_seq_name='MHC',
                epitope_max_seq_length=None,
                MHC_max_seq_length=None,
                epitope_tokenizer_name="common",
                MHC_tokenizer_name="common",
                shuffle=True,
                num_workers=1,
                ):
        self.seed = seed
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
        self.sampler_type = sampler_type

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
        self.epitope_BA_df, self.test_df = self._load_data()

        self.train_dataset = self._prepare_dataset(self.epitope_BA_df)
        self.logger.info('Load train dataset successfully')
        super().__init__(self.train_dataset, batch_size, seed, validation_split, test_split, shuffle, sampler_type, num_workers)
        
        ### test_dataset
        print('test_df',self.test_df.shape)
        self.test_dataset = self._prepare_dataset(self.test_df)
        self.test_dataloader = DataLoader(dataset=self.test_dataset, batch_size=batch_size, shuffle=shuffle)

    def get_test_dataloader(self):
        self.logger.info('Number of test data {}'.format(len(self.test_dataset)))
        return self.test_dataloader

    def _load_data(self):
        train_df = pd.read_csv(join(self.data_dir, 'fine_tune_data_train.csv'),dtype=str)
        test_df = pd.read_csv(join(self.data_dir, 'fine_tune_data_test.csv'),dtype=str)
        return train_df, test_df
    
    def _process_BA_df(self, Epitope_BA_df):

        Epitope_BA_df_random = Epitope_BA_df
        data_x_epitopes = Epitope_BA_df_random['Epitope'].to_list()
        data_MHC_pseduo_seq = Epitope_BA_df_random['HLA_pseudo_seq'].to_list()
        data_Binder = [int(float(i)) for i in Epitope_BA_df_random['Binder'].to_list()]
        # data_source = [int(i) for i in Epitope_BA_df_random['source'].to_list()]
        return data_x_epitopes, data_MHC_pseduo_seq, data_Binder

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
    
    def get_epitope_tokenizer(self):
        return self.epitope_tokenizer
    
    def get_MHC_tokenizer(self):
        return self.MHC_tokenizer
