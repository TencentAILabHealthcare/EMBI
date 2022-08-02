from data.dataset import EpitopeMHCTCRDataLoader

import logging

logger = logging.getLogger('test')

data_loader = EpitopeMHCTCRDataLoader(data_dir='/data/home/yixinguo/TcellEpitope/data/raw_data',batch_size=12,logger=logger)
