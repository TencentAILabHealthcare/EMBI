import sys
sys.path.append('../')
from os.path import join

import torch
import numpy as np
import data.test_ap_dataset_epitopeBert_MHC as module_data
import models.epitopebert_mhc as module_arch
from models.metric import correct_count, roc_auc, calculatePR
import pandas as pd
import argparse
import collections
from parse_config import ConfigParser

def main(config):
    logger = config.get_logger('train')

    # fix random seeds for reproducibility
    seed = config['data_loader']['args']['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    # setup data_loader instances
    config['data_loader']['args']['logger'] = logger
    data_loader = config.init_obj('data_loader', module_data)
    test_data_loader = data_loader.get_ap_dataset()

    # ntoken = 33
    model = config.init_obj('arch', module_arch)

    """Test."""
    logger = config.get_logger('test')

    # load best checkpoint
    # resume = '../Result/checkpoints/EpitopeBert-MHC-Pre-Debug/0830_143304/model_best.pth'
    logger.info('Loading checkpoint: {} ... '.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    model = model.to("cuda")

    model.eval()
    logger.info("Starting test.")
    correct_output = {'count': 0, 'num': 0}
    test_result = {'input': [], 'output':[], 'target': [], 'output_r':[]}       
    with torch.no_grad():
        for _, (epitope_tokenized, MHC_encoding, target) in enumerate(test_data_loader):
            epitope_tokenized = {k:v.to("cuda") for k,v in epitope_tokenized.items()}
            MHC_encoding = MHC_encoding.to("cuda")              
            target = target.to("cuda")

            output = model(epitope_tokenized, MHC_encoding)

            y_pred = output.cpu().detach().numpy()
            y_pred_r = np.round_(y_pred)
            # print()
            y_true = np.squeeze(target.cpu().detach().numpy())

            test_result['input'].append(epitope_tokenized['input_ids'].cpu().detach().numpy())
            test_result['output'].append(y_pred)
            test_result['target'].append(y_true)
            test_result['output_r'].append(y_pred_r)


            correct, num = correct_count(y_pred_r, y_true)
            correct_output['count'] += correct
            correct_output['num'] += num
    y_pred = np.concatenate(test_result['output'])
    y_true = np.concatenate(test_result['target'])
    y_pred_r = np.concatenate(test_result['output_r'])
    test_result_df = pd.DataFrame({'y_pred': list(y_pred.flatten()),
                                    'y_true': list(y_true.flatten()),
                                    'y_pred_r': list(y_pred_r.flatten())})
    precision, recall = calculatePR(test_result_df['y_pred_r'].to_list(), test_result_df['y_true'].to_list())
    auc = roc_auc(list(test_result_df['y_pred']), list(test_result_df['y_true']))                             

    logger.info("Accuracy {}, precision {}, recall {}, auc_roc {}".format(
        correct_output['count'] / correct_output['num'],
        precision,
        recall,
        auc
    ))
    test_result_df.to_csv(join(config.save_dir, '20220831_IEDB_ap_testdata.csv'), index=False)



 

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='config.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--s', '--seed'], type=int, target='data_loader;args;seed')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)