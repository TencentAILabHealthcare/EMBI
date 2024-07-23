from imp import source_from_cache
import pickle
import torch
from .base_trainer import BaseTrainer
from utils.utility import inf_loop, MetricTracker
from models.metric import correct_count, calculatePR, roc_auc, calculate_AUPRC, calculateMCC
import numpy as np
from os.path import join
import pandas as pd



class EpitopeMHCTraniner(BaseTrainer):
    """"
    Trainer class
    """
    def __init__(self, model, criterion, metric_fns, optimizer, config,
                data_loader, valid_data_loader=None, test_data_loader=None,
                lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_fns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        

        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based traning
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        
        self.valid_data_loader = valid_data_loader
        self.test_data_loader = test_data_loader

        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))       

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_fns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_fns], writer=self.writer)


    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        correct_output = {'count':0, 'num':0}
        predict_record = {'y_pred':[],'target':[]}
        print(self.data_loader)
        for batch_idx, (epitope_tokenized, MHC_tokenized, target) in enumerate(self.data_loader):
            epitope_tokenized = {k:v.to(self.device) for k,v in epitope_tokenized.items()}
            MHC_tokenized = {k:v.to(self.device) for k,v in MHC_tokenized.items()}
            target = target.to(self.device)

            # print('target',target.shape)
            self.optimizer.zero_grad()

            # input for noise student self-training
            # output, _ = self.model(epitope_tokenized, MHC_tokenized, batch_idx)
            output, _ = self.model(epitope_tokenized, MHC_tokenized)
            # output = torch.unsqueeze(output, 1)
            # print('output',output.shape)
            # target shape: [batch_size,], output shape: [batch_size, 920, 1]
            
            loss = self.criterion(output, target)
            # loss = loss.to(self.device)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.writer.add_scalar(tag = 'learning_rate',data = self.optimizer.param_groups[0]['lr'])
            self.train_metrics.update('loss', loss.item())
            with torch.no_grad():
                y_pred = output.cpu().detach().numpy()
                predict_record['y_pred'].append(y_pred)
                y_pred = np.round_(y_pred)
                y_true = np.squeeze(target.cpu().detach().numpy())
                predict_record['target'].append(y_true)
                # print('y_pred',y_pred.shape)
                # print('y_true',y_true.shape)
               
                for met in self.metric_fns:
                    self.train_metrics.update(met.__name__, met(y_pred, y_true))

                # compute the total correct predictions
                correct, num = correct_count(y_pred, y_true)
                correct_output['count'] += correct
                correct_output['num'] += num   
            
            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(epoch, self._progress(batch_idx), loss.item()))
            
            if batch_idx == self.len_epoch:
                break
        
        log = self.train_metrics.result()
        log['train'] = self.train_metrics.result()
        log['train']['total_accuracy'] = correct_output['count'] / correct_output['num']

        # predict_record
        predict_record['y_pred'] = np.concatenate(predict_record['y_pred'])
        predict_record['target'] = np.concatenate(predict_record['target'])
        log['train']['ROC_AUC'] = roc_auc(predict_record['y_pred'],predict_record['target'])
        log['train']['AUPRC'] = calculate_AUPRC(predict_record['y_pred'],predict_record['target'])
        log['train']['precision'], log['train']['recall'] = calculatePR(np.round(predict_record['y_pred']), predict_record['target'])
        log['train']['MCC'] = calculateMCC(np.round(predict_record['y_pred']), predict_record['target'])

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})
            log['validation'] = {'val_' +k : v for k,v in val_log.items()}
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

            test_log = None
        if epoch % 1 == 0:
            test_output = self.test()
            test_log={
                'test_result_of_epoch':epoch,
                'test_loss':test_output['total_loss'] / test_output['n_samples'],
                'total_accuracy': test_output['accuracy'],
                'precision':test_output['precision'],
                'recall': test_output['recall'],
                'ROC_AUC': test_output['roc_auc'],
                'AUPRC': test_output['prc_auc'],
                'MCC': test_output['MCC']
            }
        return log,test_log
    
    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """  
        self.model.eval()
        self.valid_metrics.reset()
        correct_output = {'count': 0, 'num': 0}
        predict_record = {'y_pred':[],'target':[]}
        with torch.no_grad():
            for batch_idx, (epitope_tokenized, MHC_tokenized, target) in enumerate(self.valid_data_loader):  
                epitope_tokenized = {k:v.to(self.device) for k,v in epitope_tokenized.items()}
                MHC_tokenized = {k:v.to(self.device) for k,v in MHC_tokenized.items()}
                target = target.to(self.device)

                output, _ = self.model(epitope_tokenized, MHC_tokenized)
                loss = self.criterion(output, target)      

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())

                y_pred = output.cpu().detach().numpy()
                y_pred = np.array([y_pred]) if y_pred.shape == () else y_pred
                predict_record['y_pred'].append(y_pred)
                y_pred = np.round_(y_pred)
                y_true = np.squeeze(target.cpu().detach().numpy())
                y_true = np.array([y_true]) if y_true.shape == () else y_true
                predict_record['target'].append(y_true)
                for met in self.metric_fns:
                    self.valid_metrics.update(met.__name__, met(y_pred, y_true))

                # compute the total correct predictions
                correct, num = correct_count(y_pred, y_true)
                correct_output['count'] += correct
                correct_output['num'] += num

        valid_metrics = self.valid_metrics.result()
        test_accuracy = correct_output['count'] / correct_output['num']
        valid_metrics['total_accuracy'] = test_accuracy
        
        predict_record['y_pred'] = np.concatenate(predict_record['y_pred'])
        predict_record['target'] = np.concatenate(predict_record['target'])
        valid_metrics['ROC_AUC'] = roc_auc(predict_record['y_pred'],predict_record['target'])
        valid_metrics['AUPRC'] = calculate_AUPRC(predict_record['y_pred'],predict_record['target'])
        valid_metrics['precision'], valid_metrics['recall'] = calculatePR(np.round(predict_record['y_pred']), predict_record['target'])
        valid_metrics['MCC'] = calculateMCC(np.round(predict_record['y_pred']), predict_record['target'])


        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return valid_metrics
    
    def test(self):
        self.model.eval()
        total_loss = 0.0

        correct_output = {'count': 0, 'num': 0}
        test_result = {'input': [], 'output':[], 'target': [], 'output_r':[]}       

        with torch.no_grad():
            for _, (epitope_tokenized, MHC_tokenized, target) in enumerate(self.test_data_loader):
                epitope_tokenized = {k:v.to(self.device) for k,v in epitope_tokenized.items()}
                MHC_tokenized = {k:v.to(self.device) for k,v in MHC_tokenized.items()}                
                target = target.to(self.device)

                output,_ = self.model(epitope_tokenized, MHC_tokenized)
                loss = self.criterion(output, target)
                # print('loss.item:',loss.item())
                # print('output test,', output)

                # batch_size = torch.squeeze(target).shape[0]
                batch_size = target.flatten().shape[0]
                total_loss += loss.item() * batch_size

                y_pred = output.flatten().cpu().detach().numpy()
                y_pred_r = np.round(y_pred)
                # print(np.squeeze(target.cpu().detach().numpy()))
                y_true = np.squeeze(target.cpu().detach().numpy()).flatten()
                # print(y_true.shape)
                

                test_result['input'].append(epitope_tokenized['input_ids'].cpu().detach().numpy())
                test_result['output'].append(y_pred)
                test_result['target'].append(np.asarray(y_true))
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

        test_result_df.to_csv(join(self.config._save_dir, 'testdata_predict.csv'), index=False)
        
        precision, recall = calculatePR(test_result_df['y_pred_r'].to_list(), test_result_df['y_true'].to_list())

        auc = roc_auc(list(test_result_df['y_pred']), list(test_result_df['y_true']))    

        prc = calculate_AUPRC(list(test_result_df['y_pred']), list(test_result_df['y_true'])) 

        MCC = calculateMCC(test_result_df['y_pred_r'].to_list(), test_result_df['y_true'].to_list())

        with open(join(self.config._save_dir, 'test_result.pkl'),'wb') as f:
            pickle.dump(test_result, f)

        test_output = {'n_samples': len(self.test_data_loader.sampler),
                       'total_loss': total_loss,
                       'accuracy': correct_output['count'] / correct_output['num'],
                       'precision': precision,
                       'recall': recall,
                       'roc_auc':auc,
                       'prc_auc':prc,
                       'MCC':MCC,
#                        Added
                       'test_result_df':test_result_df,
                       }
        return test_output          

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)       
