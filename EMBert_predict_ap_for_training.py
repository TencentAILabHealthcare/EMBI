import sys
sys.path.append('../')
from os.path import join

import torch
import numpy as np
import data.predict_ap_dataset_EMBert as module_data
import models.epitope_mhc_bert as module_arch
import pandas as pd
import argparse
import collections
from parse_config import ConfigParser

def predictAP(modelfile):
    
    # logger = config.get_logger('train')

    # fix random seeds for reproducibility
    # seed = config['data_loader']['args']['seed']
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # np.random.seed(seed)

    # setup data_loader instances
    config['data_loader']['args']['logger'] = logger
    data_loader = config.init_obj('data_loader', module_data)
    test_data_loader = data_loader.get_ap_dataset()
    MHC_tokenizer = data_loader.get_MHC_tokenizer()
    epitope_tokenizer = data_loader.get_epitope_tokenizer()
    # print('log_step', log_step)
    model = config.init_obj('arch', module_arch)

    """Predict."""
    logger = config.get_logger('Predict')

    # load best checkpoint
    resume = modelfile
    logger.info('Loading checkpoint: {} ... '.format(resume))
    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    model = model.to("cuda")

    model.eval()
    logger.info("Starting predict.")
    with torch.no_grad():
        for (epitope_tokenized, MHC_tokenized, target) in test_data_loader:
            # print('batch_idx', batch_idx)
            print('target', target)
            epitope_tokenized = {k:v.to("cuda") for k,v in epitope_tokenized.items()}
            MHC_tokenized = {k:v.to("cuda") for k,v in MHC_tokenized.items()}                
            target = target.to("cuda")
            print('target', target)
            
            output = model(epitope_tokenized, MHC_tokenized)
            epitope_str = epitope_tokenizer.batch_decode(epitope_tokenized['input_ids'], skip_special_tokens=True)
            epitope_nospace = [s.replace(" ","") for s in epitope_str]
            MHC_str = MHC_tokenizer.batch_decode(MHC_tokenized['input_ids'], skip_special_tokens=True)
            MHC_nospace = [s.replace(" ","") for s in MHC_str]


            y_pred = output.cpu().detach().numpy()
            y_pred_r = np.round_(y_pred)
            # print('y_pred_r:', y_pred_r)
            y_true = np.squeeze(target.cpu().detach().numpy())
            print('y_true',y_true)

            test_result['input'].append(epitope_tokenized['input_ids'].cpu().detach().numpy())
            test_result['output'].append(y_pred)
            test_result['target'].append(y_true)
            test_result['output_r'].append(y_pred_r)
            test_result['Epitope'].append(epitope_nospace)
            test_result['MHC'].append(MHC_nospace)


            correct, num = correct_count(y_pred_r, y_true)
            correct_output['count'] += correct
            correct_output['num'] += num
          

    y_pred = np.concatenate(test_result['output'])
    y_true = np.concatenate(test_result['target'])
    y_pred_r = np.concatenate(test_result['output_r'])
    epitope_input = np.concatenate(test_result['Epitope'])
    MHC_input = np.concatenate(test_result['MHC'])

    test_result_df = pd.DataFrame({
        'Epitope':list(epitope_input.flatten()),
        'MHC':list(MHC_input.flatten()),
        'EMBert_y_pred': list(y_pred.flatten()),
        'y_true': list(y_true.flatten()),
        'EMBert_y_pred_r': list(y_pred_r.flatten())})

    test_result_df.to_csv(join(config.save_dir, 'predict.csv'), index=False)
 

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='config.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-rid', '--run_id', default=None, type=str,
                      help='run id (default:None)')
    

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--s', '--seed'], type=int, target='data_loader;args;seed'),
        CustomArgs(['--pf', '--predict_file'], type=str, target='data_loader;args;predict_file')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)