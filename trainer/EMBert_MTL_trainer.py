import pickle
import torch
from .base_trainer import BaseTrainer
from utils.utility import inf_loop, MetricTracker
from models.metric import correct_count, calculatePR, roc_auc
import numpy as np
from os.path import join
import pandas as pd



class EMBertMTLTraniner(BaseTrainer):
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
        for batch_idx, (epitope_tokenized, MHC_tokenized, immu_target, BA_target, AP_target) in enumerate(self.data_loader):
            epitope_tokenized = {k:v.to(self.device) for k,v in epitope_tokenized.items()}
            MHC_tokenized = {k:v.to(self.device) for k,v in MHC_tokenized.items()}
            immu_target = immu_target.to(self.device)
            BA_target = BA_target.to(self.device)
            AP_target = AP_target.to(self.device)
            # print('target',target.shape)
            self.optimizer.zero_grad()
            immu_output, BA_output, AP_output = self.model(epitope_tokenized, MHC_tokenized)
            
            # output = torch.unsqueeze(output, 1)
            # print('output',output.shape)
            # target shape: [batch_size,], output shape: [batch_size, 920, 1]
            immu_loss = self.criterion(immu_output, immu_target)
            BA_loss = self.criterion(BA_output, BA_target)
            AP_loss = self.criterion(AP_output, AP_target)

            # weighted loss
            loss = immu_loss * 1/3 + BA_loss * 1/3 + AP_loss * 1/3
            # loss = loss.to(self.device)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
         
            with torch.no_grad():
                immu_pred = immu_output.cpu().detach().numpy()
                immu_pred_r = np.round_(immu_pred)
                BA_pred = BA_output.cpu().detach().numpy()
                BA_pred_r = np.round_(BA_pred)
                AP_pred = AP_output.cpu().detach().numpy()
                AP_pred_r = np.round_(AP_pred)

                # y_pred = np.round_(y_pred)
                immu_true = np.squeeze(immu_target.cpu().detach().numpy())
                BA_true = np.squeeze(BA_target.cpu().detach().numpy())
                AP_true = np.squeeze(AP_target.cpu().detach().numpy())
                
                # print('immu_pred',immu_pred.shape)
                # print('immu_pred',type(immu_pred_r))
                # print('y_true',y_true.shape)
                pred_df = pd.DataFrame({
                    'immu_pred_r':immu_pred_r,
                    'BA_pred_r':BA_pred_r,
                    'AP_pred_r':AP_pred_r
                    })
                pred_merge_array = np.array(pred_df['immu_pred_r'].map(str) + pred_df['BA_pred_r'].map(str) + pred_df['AP_pred_r'].map(str))
                true_df = pd.DataFrame({
                    'immu_true':immu_true,
                    'BA_true':BA_true,
                    'AP_true':AP_true
                    })
                true_merge_array = np.array(true_df['immu_true'].map(str) + true_df['BA_true'].map(str) + true_df['AP_true'].map(str))


               
                for met in self.metric_fns:
                    self.train_metrics.update(met.__name__, met(pred_merge_array, true_merge_array))

                # compute the total correct predictions
                # print('predict',np.array([immu_pred_r, BA_pred_r, AP_pred_r]))
                correct, num = correct_count(pred_merge_array, true_merge_array)
                correct_output['count'] += correct
                correct_output['num'] += num  
            
            
            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(epoch, self._progress(batch_idx), loss.item()))
            
            if batch_idx == self.len_epoch:
                break
        
        log = self.train_metrics.result()
        log['train'] = self.train_metrics.result()
        log['train']['total_accuracy'] = correct_output['count'] / correct_output['num']

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})
            log['validation'] = {'val_' +k : v for k,v in val_log.items()}
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log
    
    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """  
        self.model.eval()
        self.valid_metrics.reset()
        correct_output = {'count': 0, 'num': 0}
        with torch.no_grad():
            for batch_idx, (epitope_tokenized, MHC_tokenized, immu_target, BA_target, AP_target) in enumerate(self.valid_data_loader):  
                epitope_tokenized = {k:v.to(self.device) for k,v in epitope_tokenized.items()}
                MHC_tokenized = {k:v.to(self.device) for k,v in MHC_tokenized.items()}
                immu_target = immu_target.to(self.device)
                BA_target = BA_target.to(self.device)
                AP_target = AP_target.to(self.device)
                # print('target',target.shape)
                self.optimizer.zero_grad()
                immu_output, BA_output, AP_output = self.model(epitope_tokenized, MHC_tokenized)
                # output = torch.unsqueeze(output, 1)
                # print('output',output.shape)
                # target shape: [batch_size,], output shape: [batch_size, 920, 1]
                immu_loss = self.criterion(immu_output, immu_target)
                BA_loss = self.criterion(BA_output, BA_target)
                AP_loss = self.criterion(AP_output, AP_target)

                loss = immu_loss * 0.5 + BA_loss * 0.25 + AP_loss * 0.25    

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())

                immu_pred = immu_output.cpu().detach().numpy()
                immu_pred_r = np.round_(immu_pred)
                BA_pred = BA_output.cpu().detach().numpy()
                BA_pred_r = np.round_(BA_pred)
                AP_pred = AP_output.cpu().detach().numpy()
                AP_pred_r = np.round_(AP_pred)

                immu_true = np.squeeze(immu_target.cpu().detach().numpy())
                BA_true = np.squeeze(BA_target.cpu().detach().numpy())
                AP_true = np.squeeze(AP_target.cpu().detach().numpy())
                pred_df = pd.DataFrame({
                    'immu_pred_r':immu_pred_r,
                    'BA_pred_r':BA_pred_r,
                    'AP_pred_r':AP_pred_r
                    })
                pred_merge_array = np.array(pred_df['immu_pred_r'].map(str) + pred_df['BA_pred_r'].map(str) + pred_df['AP_pred_r'].map(str))
                true_df = pd.DataFrame({
                    'immu_true':immu_true,
                    'BA_true':BA_true,
                    'AP_true':AP_true
                    })
                true_merge_array = np.array(true_df['immu_true'].map(str) + true_df['BA_true'].map(str) + true_df['AP_true'].map(str))
                for met in self.metric_fns:
                    self.valid_metrics.update(met.__name__, met(pred_merge_array, true_merge_array))

                # compute the total correct predictions
                correct, num = correct_count(pred_merge_array, true_merge_array)
                correct_output['count'] += correct
                correct_output['num'] += num

        valid_metrics = self.valid_metrics.result()
        test_accuracy = correct_output['count'] / correct_output['num']
        valid_metrics['total_accuracy'] = test_accuracy   

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return valid_metrics
    
    def test(self):
        self.model.eval()
        total_loss = 0.0

        correct_output = {'count': 0, 'num': 0}
        test_result = {
            'input': [], 
            'immu_output':[], 'immu_target': [], 'immu_pred_r':[],
            'BA_output':[], 'BA_target': [], 'BA_pred_r':[],
            'AP_output':[], 'AP_target': [], 'AP_pred_r':[]
            }       

        with torch.no_grad():
            for _, (epitope_tokenized, MHC_tokenized, immu_target, BA_target, AP_target) in enumerate(self.test_data_loader):
                epitope_tokenized = {k:v.to(self.device) for k,v in epitope_tokenized.items()}
                MHC_tokenized = {k:v.to(self.device) for k,v in MHC_tokenized.items()}
                immu_target = immu_target.to(self.device)
                BA_target = BA_target.to(self.device)
                AP_target = AP_target.to(self.device)
                # print('target',target.shape)
                immu_output, BA_output, AP_output = self.model(epitope_tokenized, MHC_tokenized)

                immu_loss = self.criterion(immu_output, immu_target)
                BA_loss = self.criterion(BA_output, BA_target)
                AP_loss = self.criterion(AP_output, AP_target)

                loss = immu_loss * 0.5 + BA_loss * 0.25 + AP_loss * 0.25
                # print('loss.item:',loss.item())
                # print('output test,', output)

                batch_size = torch.squeeze(immu_target).shape[0]
                total_loss += loss.item() * batch_size

                immu_pred = immu_output.cpu().detach().numpy()
                immu_pred_r = np.round_(immu_pred)
                BA_pred = BA_output.cpu().detach().numpy()
                BA_pred_r = np.round_(BA_pred)
                AP_pred = AP_output.cpu().detach().numpy()
                AP_pred_r = np.round_(AP_pred)


                # y_pred_r = np.round_(y_pred)
                # print()
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
                pred_df = pd.DataFrame({
                    'immu_pred_r':immu_pred_r,
                    'BA_pred_r':BA_pred_r,
                    'AP_pred_r':AP_pred_r
                    })
                pred_merge_array = np.array(pred_df['immu_pred_r'].map(str) + pred_df['BA_pred_r'].map(str) + pred_df['AP_pred_r'].map(str))
                true_df = pd.DataFrame({
                    'immu_true':immu_true,
                    'BA_true':BA_true,
                    'AP_true':AP_true
                    })
                true_merge_array = np.array(true_df['immu_true'].map(str) + true_df['BA_true'].map(str) + true_df['AP_true'].map(str))

                correct, num = correct_count(pred_merge_array, true_merge_array)
                correct_output['count'] += correct
                correct_output['num'] += num

        immu_pred = np.concatenate(test_result['immu_output'])
        immu_true = np.concatenate(test_result['immu_target'])
        immu_pred_r = np.concatenate(test_result['immu_pred_r'])
        BA_pred = np.concatenate(test_result['BA_output'])
        BA_true = np.concatenate(test_result['BA_target'])
        BA_pred_r = np.concatenate(test_result['BA_pred_r'])
        AP_pred = np.concatenate(test_result['AP_output'])
        AP_true = np.concatenate(test_result['AP_target'])
        AP_pred_r = np.concatenate(test_result['AP_pred_r'])

        test_result_df = pd.DataFrame({
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

        test_result_df.to_csv(join(self.config._save_dir, 'testdata_predict.csv'), index=False)
        
        # precision, recall = calculatePR(
        #     [test_result_df['immu_pred_r'].to_list(), test_result_df['BA_pred_r'].to_list(), test_result_df['AP_pred_r'].to_list()], 
        #     [test_result_df['immu_true'].to_list(), test_result_df['BA_true'].to_list(), test_result_df['AP_true'].to_list()])

        # auc = roc_auc(
        #     [test_result_df['immu_pred_r'].to_list(), test_result_df['BA_pred_r'].to_list(), test_result_df['AP_pred_r'].to_list()], 
        #     [test_result_df['immu_true'].to_list(), test_result_df['BA_true'].to_list(), test_result_df['AP_true'].to_list()]
        # )    

        with open(join(self.config._save_dir, 'test_result.pkl'),'wb') as f:
            pickle.dump(test_result, f)

        test_output = {'n_samples': len(self.test_data_loader.sampler),
                       'total_loss': total_loss,
                       'accuracy': correct_output['count'] / correct_output['num'],
                    #    'precision': precision,
                    #    'recall': recall,
                    #    'roc_auc':auc
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
