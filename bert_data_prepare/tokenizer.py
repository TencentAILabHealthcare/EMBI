
from lib2to3.pgen2 import token
from os import openpty
import tempfile
import pandas as pd
import numpy as np
from abc import abstractmethod
from os.path import join
from transformers import BertTokenizer

class BaseTokenizer(object):
    def __init__(self, logger, tokenizer_name, vocab_dir, token_length_list=[2,3]):
        self.PAD = "$"
        self.MASK = "."
        self.UNK = "?"
        self.SEP = "|"
        self.CLS = "*"

        self.logger = logger
        self.logger.info(f'Using {tokenizer_name} tokenizer.')
        self.vocab_dir = vocab_dir
        self.token_length_list = token_length_list
        self.logger.info(f"Using token with length {self.token_length_list}")

        self.vocab_df, self.vocab_freq_dict = self._load_vocab()
        self.token_with_special_list, self.token2index_dict = self._get_vocab_dict()       
    
    def _load_vocab(self):
        df = pd.read_csv(self.vocab_dir, na_filter=False)
        self.logger.info('{} tokens in the vocab'.format(len(df)))
        vocab_dict = {row['toekn']: row['freq_z_normalized'] for _, row in df.iterrows()}
        return df, vocab_dict
    
    def _get_vocab_dict(self):
        amino_acids_list = list('ACDEFGHIKLMNPQRSTVWY')
        special_tokens = [self.PAD, self.MASK, self.UNK, self.SEP, self.CLS]
        token_list = list(self.vocab_df['token'])
        token_with_special_list = amino_acids_list + special_tokens + token_list
        token2index_dict = {t:i for i,t in enumerate(token_with_special_list)}

        return token_with_special_list, token2index_dict
    
    @abstractmethod
    def split(self, seq):
        raise NotImplementedError
    
    def get_bert_tokenizer(self, max_len=34, tokenizer_dir=None):
        if tokenizer_dir is not None:
            self.logger.info('Loading pre-trained tokenizer ...')
            tok = BertTokenizer.from_pretrained(
                tokenizer_dir,
                do_lower_case=False,
                do_basic_tokenize=True,
                tokenize_chinese_chars=False,
                pad_token=self.PAD,
                mask_token=self.MASK,
                unk_token=self.UNK,
                sep_token=self.SEP,
                cls_token=self.CLS,                
            )
            return tok
        
        with tempfile.TemporaryDirectory() as tempdir:
            vocab_fname = self._write_vocab(self.token2index_dict, join(tempdir,'vocab.txt'))
            tok = BertTokenizer.from_pretrained(
                vocab_fname,
                do_lower_case=False,
                do_basic_tokenize=True,
                tokenize_chinese_chars=False,
                pad_token=self.PAD,
                mask_token=self.MASK,
                unk_token=self.UNK,
                sep_token=self.SEP,
                cls_token=self.CLS,                
            ) 
        return tok
    
    def _write_vocab(self, vocab, fname):
        with open(fname, "w") as sink:
            for v in vocab:
                sink.write(v + "\n")
        return fname

                

class CommonTokenizer(object):
    def __init__(self, logger, tokenizer_name='common'):
        self.PAD = "$"
        self.MASK = "."
        self.UNK = "?"
        self.SEP = "|"
        self.CLS = "*"

        self.logger = logger
        self.logger.info(f'Using {tokenizer_name} tokenizer.')

        self.token_with_special_list, self.token2index_dict = self._get_vocab_dict()
    
    def _get_vocab_dict(self):
        amino_acids_list = list('ACDEFGHIKLMNPQRSTVWY')
        special_tokens = [self.PAD, self.MASK, self.UNK, self.SEP, self.CLS]
        # token_list = list(self.vocab_df['token'])
        token_list = amino_acids_list + special_tokens 
        token2index_dict = {t:i for i,t in enumerate(token_list)}

        return token_list, token2index_dict
    
    def get_bert_tokenizer(self, max_len=64, tokenizer_dir=None):
        if tokenizer_dir is not None:
            self.logger.info('Loading pre-trained tokenizer...')
            tok = BertTokenizer.from_pretrained(
                tokenizer_dir,
                do_lower_case=False,
                do_basic_tokenize=True,
                tokenize_chinese_chars=False,
                pad_token=self.PAD,
                mask_token=self.MASK,
                unk_token=self.UNK,
                sep_token=self.SEP,
                cls_token=self.CLS,
                padding_side="right")
            return tok

        with tempfile.TemporaryDirectory() as tempdir:
            vocab_fname = self._write_vocab(self.token2index_dict, join(tempdir, "vocab.txt"))
            tok = BertTokenizer(
                vocab_fname,
                do_lower_case=False,
                do_basic_tokenize=True,
                tokenize_chinese_chars=False,
                pad_token=self.PAD,
                mask_token=self.MASK,
                unk_token=self.UNK,
                sep_token=self.SEP,
                cls_token=self.CLS,
                model_max_len=max_len,
                padding_side="right")
        return tok   
    
    def split(self, seq):
        return list(seq)

    def _write_vocab(self, vocab, fname):
        with open(fname, "w") as sink:
            for v in vocab:
                sink.write(v + "\n")
        return fname

def get_tokenizer(tokenizer_name, logger):
    if tokenizer_name ==  "common":
        MyTokenizer = CommonTokenizer(logger=logger)
    
    return MyTokenizer