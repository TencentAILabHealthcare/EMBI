import pandas as pd
import os

def extractAAseq(file):
    data = pd.read_csv(file, sep='\t')
    AAseq_list = list(set(data['AASeq'].to_list()))
    return AAseq_list


PRJNA_file_dir = '/data/home/yixinguo/TcellEpitope/data/raw_data/TCRbFromTCRdb'
PRJNA_files = [os.path.join(PRJNA_file_dir, f) for f in os.listdir(PRJNA_file_dir)]

PRJNA_CDR3b = list(map(extractAAseq, PRJNA_files))
PRJNA_CDR3b_unique = [item for sublist in PRJNA_CDR3b for item in sublist]