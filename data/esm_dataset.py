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
        # print('MHC',MHC)
        # chain1_cdr3 = self.chain1_cdr3[index]
        # chain2_cdr3 = self.chain2_cdr3[index]
        # print('chain1_cdr3',chain1_cdr3)
        # print('chain_cdr3',chain1_cdr3)
        # immunogenic = torch.tensor(self.immunogenic[index], dtype=torch.float32)
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
        # encoded_epitope_MHC = self.tokenizer(
        #     text = epitope,
        #     text_pair = MHC,
        #     max_length = self.max_seq_length,
        #     padding="max_length",
        #     return_tensors="pt"
        # )
        """
        # print('encoded_MHC',encoded_MHC)
      
        encoded_chain1_cdr3 = self.tokenizer(
            chain1_cdr3,
            truncation="only_first",
            max_length = 20,
            padding="max_length",
            return_tensors="pt"
        ) 
        # print('encoded_chain1_cdr3', encoded_chain1_cdr3)
        encoded_chain2_cdr3 = self.tokenizer(
            chain2_cdr3,
            truncation="only_first",
            max_length = 20,
            padding="max_length",
            return_tensors="pt"
        )        
        # print('encoded_chain_cdr3',encoded_chain_cdr3)
        """
        # epitope_epitope_MHC_inputs = encoded_epitope_MHC['input_ids']
        # # don't use MHC to test model
        epitope_epitope_MHC_inputs = torch.concat((encoded_epitope['input_ids'],encoded_MHC['input_ids']),
                                        axis=1)
        # epitope_epitope_MHC_inputs = encoded_epitope['input_ids']                            
        

        epitope_epitope_MHC_attention_mask = torch.concat((encoded_epitope['attention_mask'],encoded_MHC['attention_mask']),
                                        axis=1)
        # epitope_epitope_MHC_attention_mask = encoded_epitope['attention_mask']
        # don't use MHC to test model
        # epitope_MHC_cdr3_attention_mask = torch.concat((
        #     encoded_epitope['attention_mask'], 
        #     encoded_chain1_cdr3['attention_mask'],encoded_chain2_cdr3['attention_mask']),
        #     axis=1) 
        # print('epitope_MHC_cdr3_inputs',epitope_MHC_cdr3_inputs.shape)
        # print('epitope_MHC_cdr3_attention_mask',epitope_MHC_cdr3_attention_mask.shape)
        # print('immunogenic',immunogenic.shape)

   
        return epitope_epitope_MHC_inputs, epitope_epitope_MHC_attention_mask, binder                               
        


# def loadData(data, batch=300, n_workers=12, shuffle=True):
#     return DataLoader(data, batch_size=batch, shuffle=shuffle, num_workers=n_workers)

class EpitopeMHCTCRDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size,logger, 
                seed=0, shuffle=True, validation_split=0.1, test_split=0.2, 
                num_workers=1, response_type = 'T cell', epitope_type='Peptide', cdr3_chain='both', max_seq_length=25):
        self.data_dir = data_dir
        self.response_type = response_type
        self.epitope_type = epitope_type
        self.cdr3_chain = cdr3_chain
        self.max_seq_length = max_seq_length
        self.logger = logger

        self.logger.info('Load pretrained tokenizer from ESM-1b ...')
        self.tokenizer = ESMTokenizer.from_pretrained("facebook/esm-1b", do_lower_case=False, local_files_only=True)

        # self.iedb_df, self.PRIME_df, self.Allele_data, self.VDJdb_df = self._load_data()
        self.epitope_BA_df, self.BA_testdata = self._load_data()

        # self.logger.info('IEDB, PIRME data load successfully')
        # self.iedb_allele_hla_dict, self.PRIME_allele_hla_dict, self.VDJdb_allele_hla_dict = self._load_allele_hla_seq()
        # self.cdr3b_healthy = self._load_CDR3b_from_TCRdb(self.iedb_df)
        # self.logger.info('healthy CDR3b extracted completely')

        self.train_dataset = self._prepare_dataset(self.epitope_BA_df)
        # print('train_dataset', self.train_dataset.shape)
        self.logger.info('Load train dataset successfully')
        super().__init__(self.train_dataset, batch_size, seed, shuffle, validation_split, test_split, num_workers)
        
        ### test_dataset
        self.test_dataset = self._prepare_testdata(self.BA_testdata)
        self.test_dataloader = DataLoader(dataset=self.test_dataset, batch_size=batch_size, shuffle=shuffle)

    def get_test_dataloader(self):
        self.logger.info('Number of test data {}'.format(len(self.test_dataset)))
        return self.test_dataloader

    def _load_data(self):
        # iedb_df = pd.read_csv(join(self.data_dir, 'iedb_receptor_full_v3.csv'), dtype=str)
        # PRIME_df = pd.read_csv(join(self.data_dir,'Epitope_info_from_PRIME.csv'), dtype=str)
        # Allele_data = pd.read_csv(join(self.data_dir, 'Allelelist.390.txt'),comment='#',dtype=str)
        # VDJdb_df = pd.read_csv(join(self.data_dir, 'VDJdb_paired.tsv'), dtype=str,sep='\t')
        # return iedb_df, PRIME_df, Allele_data, VDJdb_df
        Epitope_BA_df = pd.read_csv(join(self.data_dir, 'Epitope_BA_data.csv'),dtype=str)
        Epitope_BA_df_test = pd.read_csv(join(self.data_dir, '20220818_BA_testdata.csv'),dtype=str)
        return Epitope_BA_df, Epitope_BA_df_test
    
    def _process_BA_df(self, Epitope_BA_df):
        # random choice 100000
        # idx = list(range(Epitope_BA_df.shape[0]))
        # idx_random = random.sample(idx, 100000) 
        # Epitope_BA_df_random = Epitope_BA_df.iloc[idx_random]

        Epitope_BA_df_random = Epitope_BA_df
        data_x_epitopes = Epitope_BA_df_random['Epitope'].to_list()
        data_MHC_pseduo_seq = Epitope_BA_df_random['HLA_pseudo_seq'].to_list()
        data_Binder = [int(i) for i in Epitope_BA_df_random['Binder'].to_list()]
        return data_x_epitopes, data_MHC_pseduo_seq, data_Binder
    
    def _load_allele_hla_seq(self):      
        with open(join(self.data_dir, 'IEDB_allele_HLA.json'),'r') as f:
            iedb_allele_hla_dict = json.load(f)
        # print('load success')
        # iedb_allele_hla_dict = json.loads(iedb_data)
        with open(join(self.data_dir, 'PRIME_allele_HLA.json'),'r') as f:    
            PRIME_allele_hla_dict = json.load(f)
        # PRIME_allele_hla_dict = json.loads(PRIME_data)
        with open(join(self.data_dir, 'VDJdb_allele_HLA.json'),'r') as f:    
            VDJdb_allele_hla_dict = json.load(f)

        return iedb_allele_hla_dict, PRIME_allele_hla_dict, VDJdb_allele_hla_dict


    def _load_CDR3b_from_TCRdb(self, iedb_df):
        def extractAAseq(file):
            data = pd.read_csv(file, sep='\t')
            AAseq_list = list(set(data['AASeq'].to_list()))
            return AAseq_list
        PRJNA_file_dir =join(self.data_dir,'TCRbFromTCRdb')
        PRJNA_files = [os.path.join(PRJNA_file_dir, f) for f in os.listdir(PRJNA_file_dir)]
        PRJNA_CDR3b = list(map(extractAAseq, PRJNA_files))
        PRJNA_CDR3b_unique = [item for sublist in PRJNA_CDR3b for item in sublist]
        # print('1:',len(PRJNA_CDR3b_unique))

        """ not filter Epitope from iedb for test 
        ### filter CDR3b in iedb ###
        iedb_immunogencity = self._process(iedb_df)
        # print('immunogencity',iedb_immunogencity[0])
        PRJNA_CDR3b_unique_filter = []
        print(len(PRJNA_CDR3b_unique))
        for i in PRJNA_CDR3b_unique:
            # print('i',i)
            if i not in iedb_immunogencity[0]:
                PRJNA_CDR3b_unique_filter.append(i)
        print('2:',len(PRJNA_CDR3b_unique_filter))

        """

        return PRJNA_CDR3b_unique

        
    def _process(self, df):
        df = df.dropna(subset=['Description','Chain 1 CDR3 Calculated','Chain 2 CDR3 Calculated' ])
        # print('df before:',df.shape)
        # df.drop_duplicates(subset=['Description','MHC Allele Names'])
        # print('df after:',df.shape)
        epitope_types = df['Description'].apply(self._classify_epitopes)
        df_filter = df.loc[epitope_types==self.epitope_type]
        df_filter = df_filter.loc[df_filter['Response Type'].str.match(self.response_type)] 

        data_x_epitopes = df_filter['Description'].to_list()
        self.logger.info("Number of epitopes: {} ({} unique).".format(len(data_x_epitopes), len(np.unique(data_x_epitopes))))

        data_y1_CDR3 = self._merge_to_curation_if_both_exist(df_filter['Chain 1 CDR3 Calculated'], df_filter['Chain 1 CDR3 Curated'])
        data_y1_CDR3 = ['' if str(x) == 'nan' else x for x in data_y1_CDR3]
        data_y2_CDR3 = self._merge_to_curation_if_both_exist(df_filter['Chain 2 CDR3 Calculated'], df_filter['Chain 2 CDR3 Curated'])
        data_y2_CDR3 = ['' if str(x) == 'nan' else x for x in data_y2_CDR3]
        data_y_CDR3 = [[y1, y2] for y1, y2 in zip(data_y1_CDR3, data_y2_CDR3)]
        self.logger.info("Number of unique CDR3: Chain 1 = {} Chain 2 = {} Chain 1&2 = {}".format(
            np.unique(data_y1_CDR3).size, np.unique(data_y2_CDR3).size, np.unique(data_y_CDR3).size,)) 
        
        ### to do add data MHC ###
        data_MHC_seq = df_filter['MHC Allele Names'].apply(lambda x: self.iedb_allele_hla_dict[x] if (str(x) != 'nan') else '').to_list()
        self.logger.info('MHC seq success!')
        ## to do 
        data_immunogenic = [1] * df_filter.shape[0]
        print('data_immunogenic type',type(data_immunogenic[0]))
        """
        if self.cdr3_chain == "chain 1":
            data_y_CDR3 = data_y1_CDR3
        elif self.cdr3_chain == "chain 2":
            data_y_CDR3 = data_y2_CDR3
        elif not self.cdr3_chain == "both":
            raise ValueError
        print('data_y_CDR3',data_y_CDR3)
        """
        return [data_x_epitopes, data_immunogenic]
    
    ### select non-immunogenic epitope data from PRIME ### 
    def _process_RPIME_df(self, PRIME_df, cdr3b_healthy):
        """ only select non-immunogenic from PRIME
        df_filter = PRIME_df[PRIME_df['Immunogenicity'] == '0']
        data_x_epitopes = df_filter['Mutant'].to_list()
        self.logger.info("Number of epitopes from PRIME: {} ({} unique).".format(len(data_x_epitopes), len(np.unique(data_x_epitopes))))

        ### healthy data seleect from cdr3b_TCRdb randomly ### 

        data_y1_CDR3 = random.sample(cdr3b_healthy, df_filter.shape[0])
        data_y2_CDR3 = random.sample(cdr3b_healthy, df_filter.shape[0])

        # data_y_CDR3 = [[y1, y2] for y1, y2 in zip(data_y1_CDR3, data_y2_CDR3)]
        # allele_seq_dict = self._generate_allele_seq_PRIME(df_filter)
        data_MHC_seq = df_filter['Allele'].apply(lambda x: self.PRIME_allele_hla_dict[x] if (str(x) != 'nan') else '').to_list()
        data_immunogenic = [int(i) for i in df_filter['Immunogenicity'].to_list()]
        """

        # select all data
        data_x_epitopes = PRIME_df['Mutant'].to_list()
        data_immunogenic = [int(i) for i in PRIME_df['Immunogenicity'].to_list()]
        self.logger.info("Number of epitopes from PRIME: {} ({} unique).".format(len(data_x_epitopes), len(np.unique(data_x_epitopes))))
        return [data_x_epitopes, data_immunogenic]

    def _process_VDJdb_df(self, VDJdb_df):
        VDJdb_paired_ = VDJdb_df.iloc[:26804]
        VDJdb_paired_['CDR3_chain1'] = VDJdb_df.iloc[26804:]['CDR3'].to_list()
        VDJdb_paired_ = VDJdb_paired_.drop_duplicates(subset=['CDR3','CDR3_chain1','MHC A','Epitope'])
        data_x_epitopes = VDJdb_paired_['Epitope'].to_list()
        
        data_y1_CDR3 = VDJdb_paired_['CDR3'].to_list()
        data_y2_CDR3 = VDJdb_paired_['CDR3_chain1'].to_list()
        data_MHC_seq = VDJdb_paired_['MHC A'].apply(lambda x: self.VDJdb_allele_hla_dict[x] if (str(x) != 'nan') else '').to_list()
        
        data_immunogenic = [1] * VDJdb_paired_.shape[0]
        return [data_x_epitopes, data_MHC_seq, data_y1_CDR3, data_y2_CDR3, data_immunogenic]

    def _prepare_testdata(self, test_df):
        original_data = self._process_BA_df(test_df)

        dataset = EpitopeMHCTCRDataset(original_data, tokenizer=self.tokenizer, max_seq_length=self.max_seq_length)
        return dataset

    def _prepare_dataset(self, epitope_BA_df):
        """
        original_data = self._process(iedb_df)
        # print('1:',len(original_data[0]))
        # drop duplicates 
        original_data_df = pd.DataFrame(original_data).T
        original_data_df.columns = ['epitope','MHC','chain1_CDR3','chain2_CDR3','immunogenic']
        original_data_df.drop_duplicates(subset=['epitope','MHC','immunogenic'],inplace=True)
        original_data_df = original_data_df[original_data_df['MHC'] != '']
        original_data_df.to_csv('iedb_processed.csv',index=None)

        iedb_processed_df = [original_data_df.iloc[:,0].to_list(), original_data_df.iloc[:,1].to_list(),
                            original_data_df.iloc[:,2].to_list(),original_data_df.iloc[:,3].to_list(),
                            original_data_df.iloc[:,4].to_list()]
        print('2:',len(iedb_processed_df[0]))
        non_immunogenic_data = self._process_RPIME_df(PRIME_df, self.cdr3b_healthy)
        print('IEDB dataset:', len(iedb_processed_df[0]))
        print('non_immunogenic_data:', len(non_immunogenic_data[0]))

        VDJ_db_data = self._process_VDJdb_df(VDJdb_df)
        VDJdb_idx_full = np.arange(len(VDJ_db_data[0]))
        np.random.shuffle(VDJdb_idx_full)
        VDJ_db_data_random = [[],[],[],[],[]]
        for i in VDJdb_idx_full[:6000]:
            VDJ_db_data_random[0].append(VDJ_db_data[0][i])
            VDJ_db_data_random[1].append(VDJ_db_data[1][i])
            VDJ_db_data_random[2].append(VDJ_db_data[2][i])
            VDJ_db_data_random[3].append(VDJ_db_data[3][i])
            VDJ_db_data_random[4].append(VDJ_db_data[4][i])


        ### merge iddb and PRIME data and VDJdb data### 
        for i in range(len(iedb_processed_df)):
            # print('data type:',type(original_data[i][0]))
            iedb_processed_df[i].extend(non_immunogenic_data[i])

        for i in range(len(iedb_processed_df)):
            iedb_processed_df[i].extend(VDJ_db_data_random[i])
        print('after:', len(iedb_processed_df[0]))
  
        dataset = EpitopeMHCTCRDataset(iedb_processed_df, tokenizer=self.tokenizer, max_seq_length=self.max_seq_length)
        """
        # only use PRIME dataset
        # PRIME_dataset = self._process_RPIME_df(PRIME_df, self.cdr3b_healthy)
        # dataset = EpitopeMHCTCRDataset(PRIME_dataset, tokenizer=self.tokenizer, max_seq_length=self.max_seq_length)

        Epitope_BA_dataset = self._process_BA_df(epitope_BA_df)
        dataset = EpitopeMHCTCRDataset(Epitope_BA_dataset, tokenizer=self.tokenizer, max_seq_length=self.max_seq_length)
        return dataset


    def _classify_epitopes(self, epitope_description):
        '''Return the classification of epitopes.
        
        This function is borrowed from desmondyuan. The epitopes can be classified as several classes
            +: if "+" in its description
            Discontinuous Peptide: if "," in its description and all its elements (split by ,) are in the 
                form as X123 (at least starts with a upper character)
            Peptide: if all the elements (split by " ") are in the form as X123 (at least starts with a
                upper character)
        Args:
            epitope_description (str): description of epitopes

        Returns:
            epitope_type (str): classification of epitopes
                                four classifications (+, Discontinuous Peptide, Peptide, and Others) 
        '''
        epitope_type = "Others"
        if '+' in epitope_description:
            epitope_type = '+'
        elif ',' in epitope_description:
            elements = epitope_description.split(',')
            if all([self._assert_aa(ele) for ele in elements]):
                epitope_type = "Discontinuous Peptide"
        else:
            if all([self._assert_aa(ele) for ele in epitope_description]):
                epitope_type = "Peptide"
        return epitope_type

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

    def _merge_to_curation_if_both_exist(self, col1, col2):
        merged = [i if str(i) != 'nan' else j for i, j in zip(col1, col2)]
        return merged