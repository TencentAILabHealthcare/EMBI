import torch.utils.data as Data
import torch
from data.base_dataloader import BaseDataLoader
from torch.utils.data import DataLoader
import pandas as pd
from os.path import join
import random
import numpy as np
# import blosum as bl


class MLPDataset(Data.Dataset):
    def __init__(self, epitope_onehot, epitope_BLOSUM50, MHC_onehot,MHC_BLOSUM50, binder):
        self.epitope_onehot = epitope_onehot
        self.epitope_BLOSUM50 = epitope_BLOSUM50
        # self.MHC_allele_onehot = MHC_allele_onehot
        self.MHC_BLOSUM50 = MHC_BLOSUM50
        self.MHC_onehot = MHC_onehot
        # self.max_seq_length = max_seq_length
        self.binder = binder
    
    def __len__(self):
        return len(self.epitope_onehot)

    def __getitem__(self, index):
        epitope_onehot = self.epitope_onehot[index]
        epitope_BLOSUM50 = self.epitope_BLOSUM50[index]
        MHC_onehot = self.MHC_onehot[index]
        MHC_BLOSUM50 = self.MHC_BLOSUM50[index]
        # MHC_allele_onehot = self.MHC_allele_onehot[index]
        # MHC_encoding = torch.tensor(MHC_allele_onehot, dtype=torch.float32)
        epitope_encoding = np.concatenate((epitope_onehot, epitope_BLOSUM50), axis=1).flatten()
        # epitope_encoding = torch.tensor(epitope_encoding, dtype=torch.float32)
        # print('epitope shape',epitope_encoding.shape)
        # epitope_encoding = torch.tensor(epitope_encoding,dtype=torch.float32)
        # print()
        MHC_encoding = np.concatenate((MHC_onehot, MHC_BLOSUM50), axis=1).flatten()
        # MHC_encoding = torch.tensor(MHC_encoding,dtype=torch.float32)
        # hla_encoding_reshape_r = np.repeat(hla_encoding,24,axis=0)
        # epitope_MHC_encoding = np.concatenate((epitope_encoding, hla_encoding_reshape_r),axis=1)
        
        epitope_MHC_encoding = np.concatenate((epitope_encoding, MHC_encoding))
        # print('shape of epitope_MHC_encoding', epitope_MHC_encoding.shape)
        binder = torch.tensor(self.binder[index], dtype=torch.float32)
        epitope_MHC = torch.tensor(epitope_MHC_encoding,dtype=torch.float32)

        return epitope_MHC, binder

class MLPDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size,logger, 
                seed=0, shuffle=True, validation_split=0.1, test_split=0.2, 
                num_workers=1):
        self.data_dir = data_dir
        self.logger = logger
        self.epitope_BA_df = self._load_data()
        self.blosum50_dict = self._load_blosum50()

        self.train_dataset = self._prepare_dataset(self.epitope_BA_df)
        self.logger.info('Load train dataset successfully')
        super().__init__(self.train_dataset, batch_size, seed, shuffle, validation_split, test_split, num_workers)

        self.test_dataset = self.train_dataset
        self.test_dataloader = DataLoader(dataset=self.train_dataset, batch_size=batch_size, shuffle=shuffle)

    def get_test_dataloader(self):
        self.logger.info('Number of test data {}'.format(len(self.test_dataset)))
        return self.test_dataloader

    def _load_data(self):
        Epitope_BA_df = pd.read_csv(join(self.data_dir, 'Epitope_BA_data.csv'),dtype=str)
        self.logger.info('Binding affinity data ...')
        return Epitope_BA_df

    def _load_blosum50(self):
        with open(join(self.data_dir, 'Blosum50Matrix.txt'), 'r') as mf:
            lines = mf.readlines()
            dictaa = {}
            aminoacidstring = lines[0]
            aminoacidstring = aminoacidstring.split()
            i=1
            while(i<=(len(lines)-1)):
                row = lines[i]
                row = row.split()
                j=1
                for c in row[1:25]:
                    dictaa[aminoacidstring[i-1]+aminoacidstring[j-1]] = c
                    j += 1
                i += 1
        # print(dictaa)
        return dictaa

    def _process_BA_df(self, Epitope_BA_df):

        # Epitope_BA_df_positive = Epitope_BA_df[Epitope_BA_df['Binder'] == '1']
        # # Epitope_BA_df_positive_5000 = self._random_choice(Epitope_BA_df_positive, 5000)
        # Epitope_BA_df_negative = Epitope_BA_df[Epitope_BA_df['Binder'] == '0']
        # # Epitope_BA_df_negative_5000 = self._random_choice(Epitope_BA_df_negative,5000)
        # Epitope_BA_df_random = pd.concat([Epitope_BA_df_positive, Epitope_BA_df_negative])
        # Epitope_BA_df_random = Epitope_BA_df[Epitope_BA_df.Allele.str.startswith('HL')]
        Epitope_BA_df_random = Epitope_BA_df

        data_x_epitopes_onehot = [self._encode_24(i) for i in Epitope_BA_df_random['Epitope'].to_list()]
        data_x_epitopes_blosum50 = [self._encode_24_blosum50(i) for i in Epitope_BA_df_random['Epitope'].to_list()]

        data_MHC_pseduo_seq_onehot = [self._encode_seq(i) for i in Epitope_BA_df_random['HLA_pseudo_seq'].to_list()]
        data_MHC_pseduo_seq_blosum50 = [self._encode_seq_blosum50(i) for i in Epitope_BA_df_random['HLA_pseudo_seq'].to_list()]

        # MHC allele name one_hot enconding
        # data_MHC_allele_onehot = pd.get_dummies(Epitope_BA_df_random.Allele, prefix='Allele').to_numpy().tolist()
        # print(len(data_MHC_allele_onehot[0]))


        data_Binder = [int(i) for i in Epitope_BA_df_random['Binder'].to_list()]

        return data_x_epitopes_onehot, data_x_epitopes_blosum50, data_MHC_pseduo_seq_onehot, data_MHC_pseduo_seq_blosum50, data_Binder
   
    def _random_choice(self, df, num):
        idx = list(range(df.shape[0]))
        idx_random = random.sample(idx, num)
        df_random = df.iloc[idx_random]
        return df_random

    def _encode_seq(self, sequence):
        alphabet = ['A', 'C', 'D', 'E', 'F', 'G','H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        char_to_int = dict((c, i) for i, c in enumerate(alphabet))
        integer_encoded = [char_to_int[char] for char in sequence]
        onehot_encoded = list()

        for value in integer_encoded:
            letter = [0 for _ in range(len(alphabet))]
            letter[value] = 1
            onehot_encoded.append(letter)

        return np.array(onehot_encoded)  

    def _encode_24(self, seq):
        left = self._encode_seq(seq[:12])
        right = self._encode_seq(seq[-12:])
        if len(seq) < 12:
            middle = np.zeros((24-len(seq) * 2, 20), dtype='int32')
            merge = np.concatenate((left, middle, right), axis=0)
        else:
            merge = np.concatenate((left, right), axis=0)
        return merge   

    def _encode_24_blosum50(self, seq):
        left = self._encode_seq_blosum50(seq[:12])
        right = self._encode_seq_blosum50(seq[-12:])
        if len(seq) < 12:
            middle = np.zeros((24-len(seq) * 2, 20), dtype='int32')
            merge = np.concatenate((left, middle, right), axis=0)
        else:
            merge = np.concatenate((left, right), axis=0)
        return merge
    
    def _encode_seq_blosum50(self, seq):
        alphabet = ['A', 'C', 'D', 'E', 'F', 'G','H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        result = np.zeros((len(seq), len(alphabet)), dtype='int32')
        for i,s in enumerate(list(seq)):
            for j, a in enumerate(alphabet):
                blosum50 = self.blosum50_dict[s+a]
                result[i][j] = blosum50
        return result

    def _prepare_dataset(self, epitope_BA_df):
        epitope_onehot, epitope_BLOSUM50, MHC_onehot, MHC_blosum50, binder  = self._process_BA_df(epitope_BA_df)
        dataset = MLPDataset(epitope_onehot, epitope_BLOSUM50, MHC_onehot, MHC_blosum50, binder)
        return dataset