import torch
import numpy as np
import data.train_immu_multimodality_dataset as module_data
# import data.train_ap_dataset_EMBert as module_data
import models.epitope_mhc_bert_noise_train as module_arch
import models.loss as module_loss
import models.metric as module_metric
import transformers
from trainer.epitope_MHC_trainer import EpitopeMHCTraniner as Trainer
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
    # train_data_loader = data_loader.get_train_dataloader()
    valid_data_loader = data_loader.split_dataset(test=True)
    test_data_loader = data_loader.get_test_dataloader()

    
    logger.info('Number of pairs in train: {}, valid: {}, and test: {}'.format(
        data_loader.sampler.__len__(),
        valid_data_loader.sampler.__len__(),
        test_data_loader.sampler.__len__()
    ))

    # ntoken = 33
    model = config.init_obj('arch', module_arch)

    # get function handles of loss and metrics

    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]   
    logger.info(model)

    trainable_params = model.parameters()
    optimizer = config.init_obj('optimizer', transformers, trainable_params)
    # lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    if config["lr_scheduler"]["type"] == "StepLR":
        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    else:
        lr_scheduler = config.init_obj('lr_scheduler', transformers, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      test_data_loader=test_data_loader,
                      lr_scheduler=lr_scheduler)
    trainer.train()

    """Test."""
    logger = config.get_logger('test')
    logger.info(model)
    test_metrics = [getattr(module_metric, met) for met in config['metrics']]

    # load best checkpoint
    resume = str(config.save_dir / 'model_best.pth')
    # resume = '../Result/checkpoints/EMBert-Pre-Debug/0826_183701/model_best.pth'
    logger.info('Loading checkpoint: {} ... '.format(resume))
    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

    test_output = trainer.test()
    log={
        'total_accuracy': test_output['accuracy'],
        'precision':test_output['precision'],
        'recall': test_output['recall'],
        'roc_auc': test_output['roc_auc']
    }
    logger.info(log)

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
        CustomArgs(['--sep', '--add_noise_sep'], type=int, target='arch;args;Add_noise_sep'),
        CustomArgs(['--d', '--dropout'], type=float, target='arch;args;dropout'),
        CustomArgs(['--wd', '--weight_decay'], type=float, target='optimizer;args;weight_decay'),
        CustomArgs(['--ks', '--kernel_set'], type=int, target='arch;args;kernel_set'),
        
        
    ]
    config = ConfigParser.from_args(args, options)
    main(config)