import pickle
import torch
from .base_trainer import BaseTrainer
from utils.utility import inf_loop, MetricTracker
from models.metric import correct_count, calculatePR, roc_auc
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

        self.MHC_tokenizer = data_loader.get_MHC_tokenizer()
        self.epitope_tokenizer = data_loader.get_epitope_tokenizer()

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
        for batch_idx, (epitope_tokenized, MHC_tokenized, target, source) in enumerate(self.data_loader):
            epitope_tokenized = {k:v.to(self.device) for k,v in epitope_tokenized.items()}
            MHC_tokenized = {k:v.to(self.device) for k,v in MHC_tokenized.items()}
            target = target.to(self.device)

            # source = source.to_numpy()

            # print('target',target.shape)
            self.optimizer.zero_grad()
            # output = self.model(epitope_tokenized, MHC_tokenized)
            # output = torch.unsqueeze(output, 1)
            # print('output',output.shape)
            # target shape: [batch_size,], output shape: [batch_size, 920, 1]
            immu_output, BA_output, AP_output = self.model(epitope_tokenized, MHC_tokenized)

            loss0 = self.criterion(BA_output[source==0], target[source==0])
            loss1 = self.criterion(AP_output[source==1], target[source==1])
            loss2 = self.criterion(immu_output[source==2], target[source==2])

            loss = loss0+loss1+loss2

            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            with torch.no_grad():
                immu_pred = immu_output[source==2].cpu().detach().numpy()
                immu_pred_r = np.round_(immu_pred)
                BA_pred = BA_output[source==0].cpu().detach().numpy()
                BA_pred_r = np.round_(BA_pred)
                AP_pred = AP_output[source==1].cpu().detach().numpy()
                AP_pred_r = np.round_(AP_pred)

                # y_pred = np.round_(y_pred)
                immu_true = np.squeeze(target[source==2].cpu().detach().numpy())
                BA_true = np.squeeze(target[source==0].cpu().detach().numpy())
                AP_true = np.squeeze(target[source==1].cpu().detach().numpy())
                # print('y_pred',y_pred.shape)
                # print('y_true',y_true.shape)

                for met in self.metric_fns:
                    self.train_metrics.update(met.__name__, met(BA_pred_r, BA_true))
                    self.train_metrics.update(met.__name__, met(AP_pred_r, AP_true))
                    self.train_metrics.update(met.__name__, met(immu_pred_r, immu_true))

                # compute the total correct predictions
                correct_ba, num_ba = correct_count(BA_pred_r, BA_true)
                correct_ap, num_ap = correct_count(AP_pred_r, AP_true)
                correct_immu, num_immu = correct_count(immu_pred_r, immu_true)
                correct_output['count'] += correct_ba + correct_ap + correct_immu
                correct_output['num'] += num_ba + num_ap + num_immu
            
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
            for batch_idx, (epitope_tokenized, MHC_tokenized, target,source) in enumerate(self.valid_data_loader):  
                epitope_tokenized = {k:v.to(self.device) for k,v in epitope_tokenized.items()}
                MHC_tokenized = {k:v.to(self.device) for k,v in MHC_tokenized.items()}
                target = target.to(self.device)

                immu_output, BA_output, AP_output = self.model(epitope_tokenized, MHC_tokenized)

                loss0 = self.criterion(BA_output[source==0], target[source==0])
                loss1 = self.criterion(AP_output[source==1], target[source==1])
                loss2 = self.criterion(immu_output[source==2], target[source==2])

                loss = loss0+loss1+loss2      

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())

                immu_pred = immu_output[source==2].cpu().detach().numpy()
                immu_pred_r = np.round_(immu_pred)
                BA_pred = BA_output[source==0].cpu().detach().numpy()
                BA_pred_r = np.round_(BA_pred)
                AP_pred = AP_output[source==1].cpu().detach().numpy()
                AP_pred_r = np.round_(AP_pred)

                immu_true = np.squeeze(target[source==2].cpu().detach().numpy())
                BA_true = np.squeeze(target[source==0].cpu().detach().numpy())
                AP_true = np.squeeze(target[source==1].cpu().detach().numpy())

                for met in self.metric_fns:
                    self.valid_metrics.update(met.__name__, met(BA_pred_r, BA_true))
                    self.valid_metrics.update(met.__name__, met(AP_pred_r, AP_true))
                    self.valid_metrics.update(met.__name__, met(immu_pred_r, immu_true))

                # compute the total correct predictions
                correct_ba, num_ba = correct_count(BA_pred_r, BA_true)
                correct_ap, num_ap = correct_count(AP_pred_r, AP_true)
                correct_immu, num_immu = correct_count(immu_pred_r, immu_true)
                correct_output['count'] += correct_ba + correct_ap + correct_immu
                correct_output['num'] += num_ba + num_ap + num_immu

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
            'ba_output':[], 'ba_target': [], 'ba_output_r':[], 
            'ap_output':[], 'ap_target': [], 'ap_output_r':[],
            'immu_output':[], 'immu_target': [], 'immu_output_r':[],
            'Epitope':[], 'MHC':[], 'ba_source':[], 'ap_source':[], 'immu_source':[]
            }


        with torch.no_grad():
            for _, (epitope_tokenized, MHC_tokenized, target, source) in enumerate(self.test_data_loader):
                epitope_tokenized = {k:v.to(self.device) for k,v in epitope_tokenized.items()}
                MHC_tokenized = {k:v.to(self.device) for k,v in MHC_tokenized.items()}                
                target = target.to(self.device)

                epitope_str = self.epitope_tokenizer.batch_decode(epitope_tokenized['input_ids'], skip_special_tokens=True)
                epitope_nospace = [s.replace(" ","") for s in epitope_str]
                MHC_str = self.MHC_tokenizer.batch_decode(MHC_tokenized['input_ids'], skip_special_tokens=True)
                MHC_nospace = [s.replace(" ","") for s in MHC_str]

                immu_output, BA_output, AP_output = self.model(epitope_tokenized, MHC_tokenized)

                loss0 = self.criterion(BA_output[source==0], target[source==0])
                loss1 = self.criterion(AP_output[source==1], target[source==1])
                loss2 = self.criterion(immu_output[source==2], target[source==2])

                loss = loss0+loss1+loss2 
                # print('loss.item:',loss.item())
                # print('output test,', output)

                batch_size = torch.squeeze(target).shape[0]
                total_loss += loss.item() * batch_size

                immu_pred = immu_output[source==2].cpu().detach().numpy()
                immu_pred_r = np.round_(immu_pred)
                BA_pred = BA_output[source==0].cpu().detach().numpy()
                BA_pred_r = np.round_(BA_pred)
                AP_pred = AP_output[source==1].cpu().detach().numpy()
                AP_pred_r = np.round_(AP_pred)

                immu_true = np.squeeze(target[source==2].cpu().detach().numpy())
                BA_true = np.squeeze(target[source==0].cpu().detach().numpy())
                AP_true = np.squeeze(target[source==1].cpu().detach().numpy())
                # print()

                test_result['input'].append(epitope_tokenized['input_ids'].cpu().detach().numpy())
                test_result['ba_output'].append(BA_pred)
                test_result['ba_target'].append(BA_true)
                test_result['ba_output_r'].append(BA_pred_r)
                test_result['ap_output'].append(AP_pred)
                test_result['ap_target'].append(AP_true)
                test_result['ap_output_r'].append(AP_pred_r)
                test_result['immu_output'].append(immu_pred)
                test_result['immu_target'].append(immu_true)
                test_result['immu_output_r'].append(immu_pred_r)

                test_result['Epitope'].append(epitope_nospace)
                test_result['MHC'].append(MHC_nospace)
                test_result['ba_source'].append(source[source==0].cpu().detach().numpy())
                test_result['ap_source'].append(source[source==1].cpu().detach().numpy())
                test_result['immu_source'].append(source[source==2].cpu().detach().numpy())


                correct_ba, num_ba = correct_count(BA_pred_r, BA_true)
                correct_ap, num_ap = correct_count(AP_pred_r, AP_true)
                correct_immu, num_immu = correct_count(immu_pred_r, immu_true)
                correct_output['count'] += correct_ba + correct_ap + correct_immu
                correct_output['num'] += num_ba + num_ap + num_immu

        ba_pred = np.concatenate(test_result['ba_output'])
        ba_true = np.concatenate(test_result['ba_target'])
        ba_pred_r = np.concatenate(test_result['ba_output_r'])
        ba_source = np.concatenate(test_result['ba_source'])
        # print('len ba_source', ba_source)

        ap_pred = np.concatenate(test_result['ap_output'])
        ap_true = np.concatenate(test_result['ap_target'])
        ap_pred_r = np.concatenate(test_result['ap_output_r'])
        ap_source = np.concatenate(test_result['ap_source'])

        immu_pred = np.concatenate(test_result['immu_output'])
        immu_true = np.concatenate(test_result['immu_target'])
        immu_pred_r = np.concatenate(test_result['immu_output_r'])
        immu_source = np.concatenate(test_result['immu_source'])


        epitope_input = np.concatenate(test_result['Epitope'])
        MHC_input = np.concatenate(test_result['MHC'])

        ba_test_result_df = pd.DataFrame({
                                        # 'Epitope':list(epitope_input.flatten()),
                                        # 'MHC':list(MHC_input.flatten()),
                                        'ba_pred': list(ba_pred.flatten()),
                                        'ba_true': list(ba_true.flatten()),
                                        'ba_pred_r': list(ba_pred_r.flatten()),
                                        'ba_source':list(ba_source.flatten())})

        ap_test_result_df = pd.DataFrame({
                                        # 'Epitope':list(epitope_input.flatten()),
                                        # 'MHC':list(MHC_input.flatten()),
                                        'ap_pred': list(ap_pred.flatten()),
                                        'ap_true': list(ap_true.flatten()),
                                        'ap_pred_r': list(ap_pred_r.flatten()),
                                        'ap_source':list(ap_source.flatten())})

        immu_test_result_df = pd.DataFrame({
                                        # 'Epitope':list(epitope_input.flatten()),
                                        # 'MHC':list(MHC_input.flatten()),
                                        'immu_pred': list(immu_pred.flatten()),
                                        'immu_true': list(immu_true.flatten()),
                                        'immu_pred_r': list(immu_pred_r.flatten()),
                                        'immu_source':list(immu_source.flatten())})


        ba_test_result_df.to_csv(join(self.config._save_dir, 'ba_testdata_predict.csv'), index=False)
        ap_test_result_df.to_csv(join(self.config._save_dir, 'ap_testdata_predict.csv'), index=False)
        immu_test_result_df.to_csv(join(self.config._save_dir, 'immu_testdata_predict.csv'), index=False)

        
        precision_ba, recall_ba = calculatePR(ba_test_result_df['ba_pred_r'].to_list(), ba_test_result_df['ba_true'].to_list())
        precision_ap, recall_ap = calculatePR(ap_test_result_df['ap_pred_r'].to_list(), ap_test_result_df['ap_true'].to_list())
        precision_immu, recall_immu = calculatePR(immu_test_result_df['immu_pred_r'].to_list(), immu_test_result_df['immu_true'].to_list())


        auc_ba = roc_auc(list(ba_test_result_df['ba_pred']), list(ba_test_result_df['ba_true']))
        auc_ap = roc_auc(list(ap_test_result_df['ap_pred']), list(ap_test_result_df['ap_true']))
        auc_immu = roc_auc(list(immu_test_result_df['immu_pred']), list(immu_test_result_df['immu_true']))
          

        with open(join(self.config._save_dir, 'test_result.pkl'),'wb') as f:
            pickle.dump(test_result, f)

        test_output = {'n_samples': len(self.test_data_loader.sampler),
                       'total_loss': total_loss,
                       'accuracy': correct_output['count'] / correct_output['num'],
                       'ba_ap_immu_precision': [precision_ba, precision_ap, precision_immu],
                       'ba_ap_immu_recall': [recall_ba, recall_ap, recall_immu],
                       'ba_ap_immu_roc_auc':[auc_ba, auc_ap, auc_immu]
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
