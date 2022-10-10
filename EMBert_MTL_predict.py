import sys
sys.path.append('../')
from os.path import join

import torch
import numpy as np
import data.predict_dataset_EMBert_MTL as module_data
import models.EMBert_MTL as module_arch
from models.metric import correct_count
import pandas as pd
import argparse
import collections
from parse_config import ConfigParser
from tqdm import tqdm

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
    predict_data_loader = data_loader.get_predict_dataloader()
    MHC_tokenizer = data_loader.get_MHC_tokenizer()
    epitope_tokenizer = data_loader.get_epitope_tokenizer()
    # print('log_step', log_step)
    model = config.init_obj('arch', module_arch)

    """Predict."""
    logger = config.get_logger('Predict')

    # load best checkpoint
    resume = '../Result/checkpoints/EMBert-MTL/0929_195300/model_best.pth'
    logger.info('Loading checkpoint: {} ... '.format(resume))
    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    model = model.to("cuda")

    model.eval()
    logger.info("Starting predict.")
    # correct_output = {'count': 0, 'num': 0}
    test_result = {
        'input': [], 
        'immu_output':[], 'immu_target': [], 'immu_pred_r':[],
        'BA_output':[], 'BA_target': [], 'BA_pred_r':[],
        'AP_output':[], 'AP_target': [], 'AP_pred_r':[], 
        'Epitope':[], 'MHC':[]
    } 

    with torch.no_grad():
        for (epitope_tokenized, MHC_tokenized, immu_target, BA_target, AP_target) in tqdm(predict_data_loader):
            # print('batch_idx', batch_idx)
            epitope_tokenized = {k:v.to("cuda") for k,v in epitope_tokenized.items()}
            MHC_tokenized = {k:v.to("cuda") for k,v in MHC_tokenized.items()}
            immu_target = immu_target.to("cuda")
            BA_target = BA_target.to("cuda")
            AP_target = AP_target.to("cuda")
            # print('target',target.shape)
            immu_output, BA_output, AP_output = model(epitope_tokenized, MHC_tokenized)
            
            epitope_str = epitope_tokenizer.batch_decode(epitope_tokenized['input_ids'], skip_special_tokens=True)
            epitope_nospace = [s.replace(" ","") for s in epitope_str]
            MHC_str = MHC_tokenizer.batch_decode(MHC_tokenized['input_ids'], skip_special_tokens=True)
            MHC_nospace = [s.replace(" ","") for s in MHC_str]


            immu_pred = immu_output.cpu().detach().numpy()
            immu_pred_r = np.round_(immu_pred)
            BA_pred = BA_output.cpu().detach().numpy()
            BA_pred_r = np.round_(BA_pred)
            AP_pred = AP_output.cpu().detach().numpy()
            AP_pred_r = np.round_(AP_pred)

            immu_true = np.squeeze(immu_target.cpu().detach().numpy())
            BA_true = np.squeeze(BA_target.cpu().detach().numpy())
            AP_true = np.squeeze(AP_target.cpu().detach().numpy())

            test_result['input'].append(epitope_tokenized['input_ids'].cpu().detach().numpy())
            test_result['immu_output'].append(immu_pred)
            test_result['immu_target'].append(immu_true)
            test_result['immu_pred_r'].append(immu_pred_r)
            test_result['BA_output'].append(BA_pred)
            test_result['BA_target'].append(BA_true)
            test_result['BA_pred_r'].append(BA_pred_r)
            test_result['AP_output'].append(AP_pred)
            test_result['AP_target'].append(AP_true)
            test_result['AP_pred_r'].append(AP_pred_r)  
            test_result['Epitope'].append(epitope_nospace)
            test_result['MHC'].append(MHC_nospace)
        
    immu_pred = np.concatenate(test_result['immu_output'])
    immu_true = np.concatenate(test_result['immu_target'])
    immu_pred_r = np.concatenate(test_result['immu_pred_r'])
    BA_pred = np.concatenate(test_result['BA_output'])
    BA_true = np.concatenate(test_result['BA_target'])
    BA_pred_r = np.concatenate(test_result['BA_pred_r'])
    AP_pred = np.concatenate(test_result['AP_output'])
    AP_true = np.concatenate(test_result['AP_target'])
    AP_pred_r = np.concatenate(test_result['AP_pred_r'])
    epitope_input = np.concatenate(test_result['Epitope'])
    MHC_input = np.concatenate(test_result['MHC'])

    test_result_df = pd.DataFrame({
            'Epitope':list(epitope_input.flatten()),
            'MHC':list(MHC_input.flatten()),
            'immu_pred': list(immu_pred.flatten()),
            'immu_true': list(immu_true.flatten()),
            'immu_pred_r': list(immu_pred_r.flatten()),
            'AP_pred': list(AP_pred.flatten()),
            'AP_true': list(AP_true.flatten()),
            'AP_pred_r': list(AP_pred_r.flatten()),
            'BA_pred': list(BA_pred.flatten()),
            'BA_true': list(BA_true.flatten()),
            'BA_pred_r': list(BA_pred_r.flatten())
            })

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