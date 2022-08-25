import numpy as np
import pandas as pd
from os.path import join
from torch.utils.data import DataLoader
from data.esm_dataset import EpitopeMHCTCRDataset
from transformers import ESMTokenizer




class EpitopeMHCDataset(object):
    def __init__(self,
                logger, 
                seed,
                batch_size,
                data_dir,
                use_part = 10000, 
                shuffle=True,
                max_seq_length=40):
        self.logger = logger
        self.seed = seed
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.use_part = use_part
        self.max_seq_length = max_seq_length
        self.rng = np.random.default_rng(seed=self.seed)
        self.pos_df, self.neg_df = self._load_data()
        self.logger.info("Load ESM-1b ...")
        self.tokenizer = ESMTokenizer.from_pretrained("facebook/esm-1b", do_lower_case=False, local_files_only=False)

        # '''Get tokenizer'''
        # self.logger.info('Creating tokenizer...')

    def _load_data(self):
        self.logger.info("Using binding affinity data to test ...")

        # Epitope_BA_df_test = pd.read_csv(join(self.data_dir, '20220821CD8_benchmark_data_1_500.csv'),dtype=str)
        # Epitope_BA_df_test = pd.read_csv(join(self.data_dir, 'PRIME_data_for_benchmark.csv'),dtype=str)
        Epitope_BA_df_test = pd.read_csv(join(self.data_dir, '20220825benchmark_data_from_DeepNetBim.csv'),dtype=str)


        Epitope_BA_df_test = Epitope_BA_df_test[['Epitope','Allele','Binder','HLA_pseudo_seq']]
        pos_df = Epitope_BA_df_test[Epitope_BA_df_test['Binder']== '1']
        neg_df = Epitope_BA_df_test[Epitope_BA_df_test['Binder']== '0']
        self.logger.info("Dataset shape {} with {} positive and {} negative".format(
                Epitope_BA_df_test.shape, len(pos_df), len(neg_df)
            ))
        
        return pos_df, neg_df

    def _random_sample_balance(self):
        self.logger.info('Create sampled data.')
        # pos_df = self.pos_df.sample(n=int(0.5*self.use_part), random_state=self.seed).reset_index(drop=True)
        # neg_df = self.neg_df.sample(n=int(0.5*self.use_part), random_state=self.seed).reset_index(drop=True)
        sample_df = pd.concat([self.pos_df, self.neg_df])
        sample_df = sample_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        return sample_df

    def get_binding_affinitiy_dataset(self):

        if self.use_part is not None:
            self.logger.info(f'Using {self.use_part} to test.')
            df = self._random_sample_balance()
        
        self.logger.info("{} positive and {} negative".format(
            len(df[df['Binder']== '1']), len(df[df['Binder']== '0'])))
        
        ba_dataset = EpitopeMHCTCRDataset(
            [list(df['Epitope']), list(df['HLA_pseudo_seq']), [int(i) for i in list(df['Binder'])]],
            tokenizer=self.tokenizer, 
            max_seq_length=self.max_seq_length
        )
        ba_dataloader = DataLoader(dataset=ba_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

        return ba_dataloader