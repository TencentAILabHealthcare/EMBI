import sys
sys.path.append('../')
from os.path import join

import torch
import numpy as np
import data.predict_immu_multimodality_first_dataset as module_data
import models.epitope_mhc_bert_multimodality as module_arch
from models.metric import correct_count
import pandas as pd
import argparse
import collections
from parse_config import ConfigParser
from tqdm import tqdm
import models.epitope_mhc_bert as module_arch_


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
    test_data_loader = data_loader.get_predict_dataset()
    MHC_tokenizer = data_loader.get_MHC_tokenizer()
    epitope_tokenizer = data_loader.get_epitope_tokenizer()
    # print('log_step', log_step)
    model = config.init_obj('arch', module_arch)

    """Predict."""
    logger = config.get_logger('Predict')

    # load best checkpoint
    ## first trained model '../Result/checkpoints/EMBert-BA-AP-Immu/Multimodality/Retrain/1020_105319/model_best.pth'
    ## retarined model '../Result/checkpoints/EMBert-BA-AP-Immu/Multimodality/Retrain/1020_105319/model_best.pth'
    # fine-tuned on cancer data model '../Result/checkpoints/EMBert-BA-AP-Immu/Multimodality/Fine_tune_cancer/1116_170629/model_best.pth'
    # fine-tuned on covid-19 model '../Result/checkpoints/EMBert-BA-AP-Immu/Multimodality/Fine_tune_covid_19/1116_192013/model_best.pth'
    # fine_tuned on covid-19 new data 0.4/0.1/0.5 '../Result/checkpoints/EMBert-BA-AP-Immu/Multimodality/Covid_19/1117_171258/model_best.pth'
    # fine-tuned on cancer data 0.4/0.1/0.5 '../Result/checkpoints/EMBert-BA-AP-Immu/Multimodality/Fine_tune_cancer/1117_142620/model_best.pth'
    # fine-tuned on dbpepneo melenoma and tesla negative '../Result/checkpoints/EMBert-BA-AP-Immu/Multimodality/Fine_tune_dppepneo_tesla/1119_163304/model_best.pth'
    # fine-tuned on dbpepneo melenoma and tesla negative seed 3407 '../Result/checkpoints/EMBert-BA-AP-Immu/Multimodality/Fine_tune_dppepneo_tesla/1124_104841/model_best.pth'
    resume = 'EMBI_multimodality_model/model_best.pth'
    logger.info('Loading checkpoint: {} ... '.format(resume))
    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    model = model.to("cuda")

    # load ba and ap best checkpoint
    ba_model = config.init_obj('arch_ba', module_arch_)
    ap_model = config.init_obj('arch_ap', module_arch_)
    # original ap model '../Result/checkpoints/EMBert-Pre-Debug/0826_183701/model_best.pth'
    # ap semi model '../Result/checkpoints/EMBert-AP-Data-Augmentation/0914_144510/model_best.pth'
    ba_model_resume = "EMBI_BA_model/model_best.pth"
    ap_model_resume = "EMBI_AP_semi_model/model_best.pth"
    ba_model.load_state_dict(torch.load(ba_model_resume)['state_dict'])
    ap_model.load_state_dict(torch.load(ap_model_resume)['state_dict'])
    ba_model.to("cuda")
    ap_model.to("cuda")

    model.eval()
    logger.info("Starting predict.")
    correct_output = {'count': 0, 'num': 0}
    test_result = {
        'input': [], 'output':[], 'target': [], 'output_r':[],'Epitope':[], 'MHC':[],
        'ba_output':[], 'ap_output':[], 'HLA_name':[]

    }       
    with torch.no_grad():
        for (epitope_tokenized, MHC_tokenized, target, HLA_name) in tqdm(test_data_loader):
            # print('batch_idx', batch_idx)
            # print('target', target)
            epitope_tokenized = {k:v.to("cuda") for k,v in epitope_tokenized.items()}
            MHC_tokenized = {k:v.to("cuda") for k,v in MHC_tokenized.items()}                
            target = target.to("cuda")
            # print('target', target)

            ba_output,ba_ReLU_output = ba_model(epitope_tokenized, MHC_tokenized)
            ap_output, ap_ReLU_output = ap_model(epitope_tokenized, MHC_tokenized)

            output = model(ba_ReLU_output, ap_ReLU_output, epitope_tokenized, MHC_tokenized)

            epitope_str = epitope_tokenizer.batch_decode(epitope_tokenized['input_ids'], skip_special_tokens=True)
            epitope_nospace = [s.replace(" ","") for s in epitope_str]
            MHC_str = MHC_tokenizer.batch_decode(MHC_tokenized['input_ids'], skip_special_tokens=True)
            MHC_nospace = [s.replace(" ","") for s in MHC_str]


            y_pred = output.cpu().detach().numpy()
            y_pred_r = np.round_(y_pred)
            # print('y_pred_r:', y_pred_r)
            y_true = np.squeeze(target.cpu().detach().numpy())
            ba_output = np.squeeze(ba_output.cpu().detach().numpy())
            ap_output = np.squeeze(ap_output.cpu().detach().numpy())
            # print('y_true',y_true)

            test_result['input'].append(epitope_tokenized['input_ids'].cpu().detach().numpy())
            test_result['output'].append(y_pred)
            test_result['target'].append(y_true)
            test_result['output_r'].append(y_pred_r)
            test_result['Epitope'].append(epitope_nospace)
            test_result['MHC'].append(MHC_nospace)
            test_result['ba_output'].append(ba_output)
            test_result['ap_output'].append(ap_output)
            test_result['HLA_name'].append(HLA_name)

            correct, num = correct_count(y_pred_r, y_true)
            correct_output['count'] += correct
            correct_output['num'] += num
          

    y_pred = np.concatenate(test_result['output'])
    y_true = np.concatenate(test_result['target'])
    y_pred_r = np.concatenate(test_result['output_r'])
    epitope_input = np.concatenate(test_result['Epitope'])
    MHC_input = np.concatenate(test_result['MHC'])
    ba_p = np.concatenate(test_result['ba_output'])
    ap_p = np.concatenate(test_result['ap_output'])
    hla_name = np.concatenate(test_result['HLA_name'])

    test_result_df = pd.DataFrame({
        'Epitope':list(epitope_input.flatten()),
        'MHC':list(MHC_input.flatten()),
        'HLA':list(hla_name.flatten()),
        'ba_p':list(ba_p.flatten()),
        'ap_p':list(ap_p.flatten()),
        'Immu_pred': list(y_pred.flatten()),
        'y_true': list(y_true.flatten()),
        'Immu_y_pred_r': list(y_pred_r.flatten())})

    test_result_df.to_csv(join(config.save_dir, 'predict.csv'), index=False)
    print('correct_output', correct_output)
 

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