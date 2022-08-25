#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('../')

import argparse
import collections
import torch
import numpy as np
import pandas as pd
from os.path import join

import data.test_binding_affinity_dataset as module_data
import models.esm as module_arch
from models.metric import calculatePR, correct_count, roc_auc
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
    binding_affinity_dataloader = data_loader.get_binding_affinitiy_dataset()

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)

    """Test."""
    logger = config.get_logger('test')
    
    # load best checkpoint
    # resume = str(config.save_dir / 'model_best.pth')
    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    model = model.to("cuda")

    model.eval()
    logger.info("Starting test.")
    result_dict = {'y_true': [], 'y_pred': [],'y_pred_r':[]}
    correct_output = {'count': 0, 'num': 0}
    with torch.no_grad():
        for batch_idx, (x_input_ids, x_attention_mask, target) in enumerate(binding_affinity_dataloader):
            x_input_ids, x_attention_mask = x_input_ids.to("cuda"), x_attention_mask.to("cuda")
            target = target.to("cuda")

            output = model(x_input_ids, x_attention_mask)
            # print('loss.item:',loss.item())
            # print('output test,', output)
            y_pred = output.cpu().detach().numpy()
            y_pred_r = np.round_(y_pred)
            # print()
            y_true = np.squeeze(target.cpu().detach().numpy())
            # TP, FP, TN, FN = calculatePR(y_pred, y_true)
            result_dict['y_true'].append(y_true)
            result_dict['y_pred'].append(y_pred)
            result_dict['y_pred_r'].append(y_pred_r)

            correct, num = correct_count(y_pred_r, y_true)
            correct_output['count'] += correct
            correct_output['num'] += num
    y_pred = np.concatenate(result_dict['y_pred'])
    y_true = np.concatenate(result_dict['y_true'])
    y_pred_r = np.concatenate(result_dict['y_pred_r'])

    test_result_df = pd.DataFrame({'y_pred': list(y_pred.flatten()),
                                   'y_true': list(y_true.flatten()),
                                   'y_pred_r': list(y_pred_r.flatten())})
    precision, recall = calculatePR(test_result_df['y_pred_r'].to_list(), test_result_df['y_true'].to_list())

    logger.info("Accuracy {}, precision {}, recall {}, auc_roc {}".format(
        correct_output['count'] / correct_output['num'],
        precision,
        recall,
        roc_auc(list(test_result_df['y_pred']), list(test_result_df['y_true']))
    ))
    test_result_df.to_csv(join(config.save_dir, '20220825_ESM_DeepNetBim_benchmark_test_result.csv'), index=False)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='config.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-local_rank', '--local_rank', default=None, type=str,
                      help='local rank for nGPUs training')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
