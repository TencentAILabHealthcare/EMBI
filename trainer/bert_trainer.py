
from .base_trainer import BaseTrainer
from utils.utility import inf_loop, MetricTracker
from models.metric import correct_count
import numpy as np
import torch
import pickle
from os.path import join
import numpy as np
from models.metric import correct_count

class BERTTrainer(BaseTrainer):

    def __init__(self, model, ntoken, criterion, metric_fns, optimizer, config,
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

        self.ntoken = ntoken

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
        for batch_idx, (x_input_ids, x_attention_mask, target) in enumerate(self.data_loader):
            x_input_ids, x_attention_mask = x_input_ids.to(self.device), x_attention_mask.to(self.device)
            target = target.to(self.device)

            # print('target',target.shape)

            output = self.model(x_input_ids, x_attention_mask)
            # output = torch.unsqueeze(output, 1)
            # print('output',output.shape)
            # target shape: [batch_size,], output shape: [batch_size, 920, 1]
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            with torch.no_grad():
                y_pred = output.cpu().detach().numpy()
                y_pred = np.round_(y_pred)
                y_true = np.squeeze(target.cpu().detach().numpy())
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
            for batch_idx, (x_input_ids, x_attention_mask, target) in enumerate(self.valid_data_loader):  
                x_input_ids, x_attention_mask = x_input_ids.to(self.device), x_attention_mask.to(self.device)
                target = target.to(self.device)

                output = self.model(x_input_ids, x_attention_mask)
                loss = self.criterion(output, target)      

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())

                y_pred = output.cpu().detach().numpy()
                y_pred = np.round_(y_pred)
                y_true = np.squeeze(target.cpu().detach().numpy())
                for met in self.metric_fns:
                    self.valid_metrics.update(met.__name__, met(y_pred, y_true))

                # compute the total correct predictions
                correct, num = correct_count(y_pred, y_true)
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
        test_result = {'input': [], 'output':[], 'target': []}       

        with torch.no_grad():
            for batch_idx, (x_input_ids, x_attention_mask, target) in enumerate(self.test_data_loader):
                x_input_ids, x_attention_mask = x_input_ids.to(self.device), x_attention_mask.to(self.device)
                target = target.to(self.device)

                output = self.model(x_input_ids, x_attention_mask)
                loss = self.criterion(output, target)

                print('loss.item:',loss.item())
                # print('output test,', output)

                batch_size = torch.squeeze(target).shape[0]
                total_loss += loss.item() * batch_size

                y_pred = output.cpu().detach().numpy()
                y_pred = np.round_(y_pred)
                # print()
                y_true = np.squeeze(target.cpu().detach().numpy())

                test_result['input'].append(x_input_ids.cpu().detach().numpy())
                test_result['output'].append(y_pred)
                test_result['target'].append(y_true)

                correct, num = correct_count(y_pred, y_true)
                correct_output['count'] += correct
                correct_output['num'] += num
        with open(join(self.config._save_dir, 'test_result.pkl'),'wb') as f:
            pickle.dump(test_result, f)

        test_output = {'n_samples': len(self.test_data_loader.sampler),
                       'total_loss': total_loss,
                       'accuracy': correct_output['count'] / correct_output['num']}
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