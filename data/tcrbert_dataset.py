
import torch.utils.data as Data
import torch
from data.base_dataloader import BaseDataLoader
from transformers import BertTokenizer
from torch.utils.data import DataLoader
import pandas as pd
from os.path import join
import random


class TCRBertDataset(Data.Dataset):
    def __init__(self, original_data, tokenizer, max_seq_length):
        self.epitope = original_data[0]
        self.MHC = original_data[1]
        # self.chain1_cdr3 = original_data[2]
        # self.chain2_cdr3 = original_data[3]
        # self.immunogenic = original_data[1]
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

        # epitope_epitope_MHC_inputs = torch.concat((encoded_epitope['input_ids'],encoded_MHC['input_ids']),
        #                                 axis=1)
        epitope_epitope_MHC_inputs = encoded_epitope['input_ids']                                
        # epitope_epitope_MHC_attention_mask = torch.concat((encoded_epitope['attention_mask'],encoded_MHC['attention_mask']),
        #                                 axis=1)
        epitope_epitope_MHC_attention_mask = encoded_epitope['attention_mask']                            
        return epitope_epitope_MHC_inputs, epitope_epitope_MHC_attention_mask, binder                               

class TCRBertDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size,logger, 
                seed=0, shuffle=True, validation_split=0.1, test_split=0.2, 
                num_workers=1, response_type = 'T cell', epitope_type='Peptide', cdr3_chain='both', max_seq_length=25):
        self.data_dir = data_dir
        self.response_type = response_type
        self.epitope_type = epitope_type
        self.cdr3_chain = cdr3_chain
        self.max_seq_length = max_seq_length
        self.logger = logger

        self.logger.info('Load pretrained tokenizer from TCRBert ...')
        self.tokenizer = BertTokenizer.from_pretrained("wukevin/tcr-bert", do_lower_case=False, local_files_only=True)

        self.epitope_BA_df = self._load_data()
        self.train_dataset = self._prepare_dataset(self.epitope_BA_df)
        self.logger.info('Load train dataset successfully')
        super().__init__(self.train_dataset, batch_size, seed, shuffle, validation_split, test_split, num_workers)
        
        ### test_dataset
        self.test_dataset = self._prepare_testdata(self.epitope_BA_df)
        self.test_dataloader = DataLoader(dataset=self.test_dataset, batch_size=batch_size, shuffle=shuffle)

    def get_test_dataloader(self):
        self.logger.info('Number of test data {}'.format(len(self.test_dataset)))
        return self.test_dataloader


    def _load_data(self):
        Epitope_BA_df = pd.read_csv(join(self.data_dir, 'Epitope_BA_data.csv'),dtype=str)
        self.logger.info('Binding affinity data ...')
        return Epitope_BA_df

    def _process_BA_df(self, Epitope_BA_df):
        idx = list(range(Epitope_BA_df.shape[0]))
        idx_random = random.sample(idx, 100000) 
        Epitope_BA_df_random = Epitope_BA_df.iloc[idx_random]
        data_x_epitopes = Epitope_BA_df_random['Epitope'].to_list()
        data_MHC_pseduo_seq = Epitope_BA_df_random['HLA_pseudo_seq'].to_list()
        data_Binder = [int(i) for i in Epitope_BA_df_random['Binder'].to_list()]
        return data_x_epitopes, data_MHC_pseduo_seq, data_Binder
    
    def _prepare_testdata(self, iedb_df):
        original_data = self._process_BA_df(iedb_df)
        dataset = TCRBertDataset(original_data, tokenizer=self.tokenizer, max_seq_length=self.max_seq_length)
        return dataset
    

    def _prepare_dataset(self, epitope_BA_df):
        Epitope_BA_dataset = self._process_BA_df(epitope_BA_df)
        dataset = TCRBertDataset(Epitope_BA_dataset, tokenizer=self.tokenizer, max_seq_length=self.max_seq_length)
        return dataset
    
