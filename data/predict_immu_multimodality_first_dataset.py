from cmath import log
from os.path import join
# from matplotlib.pyplot import axis
from torch.utils.data import Dataset, DataLoader
from bert_data_prepare.tokenizer import get_tokenizer
import pandas as pd
import torch


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
        # keep HLA name
        self.HLA_name = original_data[3]
        self.ID = original_data[4]
        # self.target = original_data[4]
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
        binder_tensor = torch.tensor(self.binder[index], dtype=torch.float32)
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
        HLA_name = self.HLA_name[index]
        # for covid_19
        # target = self.target[index]
        ID = self.ID[index]
        return epitope_tensor, MHC_tensor, binder_tensor, HLA_name, ID                    

    def _insert_whitespace(self, token_list):
        """
        Return the sequence of tokens with whitespace after each char
        """
        return " ".join(token_list)

class EpitopeMHCBertDataLoader(object):
    def __init__(self, data_dir, batch_size,logger, 
                seed, predict_file,  
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
        self.batch_size = batch_size
        self.seed = seed
        self.predict_file = predict_file
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
        self.predict_df = self._load_data()

        self.logger.info('Load train dataset successfully')
        # super().__init__(self.train_dataset, batch_size, seed, validation_split, test_split, shuffle, sampler_type, num_workers)
        
        ### test_dataset
    def get_epitope_tokenizer(self):
        return self.epitope_tokenizer
    
    def get_MHC_tokenizer(self):
        return self.MHC_tokenizer

    def get_test_dataloader(self):
        self.logger.info('Number of test data {}'.format(len(self.test_dataset)))
        return self.test_dataloader

    def _load_data(self):
        test_df = pd.read_csv(join(self.data_dir, self.predict_file),dtype=str)
        return test_df
    
    def _process_BA_df(self, Epitope_BA_df):

        Epitope_BA_df_random = Epitope_BA_df
        data_x_epitopes = Epitope_BA_df_random['Epitope'].to_list()
        data_MHC_pseduo_seq = Epitope_BA_df_random['HLA_pseudo_seq'].to_list()
        data_Binder = [0] * Epitope_BA_df.shape[0]
        # data_Binder = [int(float(i)) for i in Epitope_BA_df_random['Binder'].to_list()]
        data_HLA_name = Epitope_BA_df_random['Allele'].to_list()
        data_ID = Epitope_BA_df_random['ID'].to_list()
        return data_x_epitopes, data_MHC_pseduo_seq, data_Binder, data_HLA_name, data_ID
        
    def get_predict_dataset(self):
        df = self.predict_df    
        predict_dataset = self._prepare_dataset(df)
        predict_dataloader = DataLoader(dataset=predict_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        return predict_dataloader

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
    
