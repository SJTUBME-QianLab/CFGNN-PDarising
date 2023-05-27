from __future__ import print_function
import os
import time
import yaml
import pickle
from shutil import copytree, ignore_patterns
import argparse
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from tools.utils import seed_torch, str2bool, str2list, get_graph_name, import_class, coef_list, Prior
from settle_results import SettleResults
TF_ENABLE_ONEDNN_OPTS = 0
torch.use_deterministic_algorithms(True)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--exp_name', default='')
    parser.add_argument('--save_dir', default='./results', help='the work folder for storing results')
    parser.add_argument('--data_dir', default='./')
    parser.add_argument('--config', default='./train_causal.yaml', help='path to the configuration file')
    parser.add_argument('--seed', default=1, type=int, help='seed for random')
    parser.add_argument('--split_seed', default=1, type=int)
    parser.add_argument('--fold', default=0, type=int, help='0-4, fold idx for cross-validation')
    parser.add_argument('--data_name', default='arising_2_0n0', type=str)
    parser.add_argument('--patch_size', default=5, type=int)

    # visualize and debug
    parser.add_argument('--save_score', type=str2bool, default=True,
                        help='if ture, the classification score will be stored')
    parser.add_argument('--log_interval', type=int, default=100, help='the interval for printing messages (#iteration)')
    parser.add_argument('--save_interval', type=int, default=5, help='the interval for storing models (#iteration)')
    parser.add_argument('--eval_interval', type=int, default=5, help='the interval for evaluating models (#iteration)')
    parser.add_argument('--print_log', type=str2bool, default=True, help='print logging or not')

    # feeder
    parser.add_argument('--feeder', default='tools.feeder.Feeder', help='data loader will be used')
    parser.add_argument('--num_worker', type=int, default=0, help='the number of worker for data loader')
    parser.add_argument('--train_feeder_args', default=dict(), help='the arguments of data loader for training')
    parser.add_argument('--test_feeder_args', default=dict(), help='the arguments of data loader for test')

    # model
    parser.add_argument('--graph_args', default=dict(), help='the arguments of model')  # type=dict,
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument('--model_args', default=dict(), help='the arguments of model')  # type=dict,

    # hyper-parameters
    parser.add_argument('--pre_epoch', type=int, default=0)
    parser.add_argument('--inc_mode', type=str, default='lin')
    parser.add_argument('--LO', type=float, default=0)
    parser.add_argument('--LR', type=float, default=0)
    parser.add_argument('--LG', type=float, default=0)

    # optimizer
    parser.add_argument('--base_lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--base_lr_mask', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--scheduler', type=str, default='step', help='scheduler')
    parser.add_argument('--stepsize', type=int, default=30, help='scheduler step size')
    parser.add_argument('--gamma', type=float, default=0.5, help='scheduler shrinking rate')
    parser.add_argument('--device', type=int, default=0, nargs='+', help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument('--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay for optimizer')
    parser.add_argument('--batch_size', type=int, default=50, help='training batch size')
    parser.add_argument('--test_batch_size', type=int, default=50, help='test batch size')
    parser.add_argument('--start_epoch', type=int, default=0, help='start training from which epoch')
    parser.add_argument('--num_epoch', type=int, default=80, help='stop training in which epoch')
    return parser


class Processor:
    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        self.load_data()
        self.train_writer = SummaryWriter(os.path.join(self.work_dir, 'train'), 'train')
        self.val_writer = SummaryWriter(os.path.join(self.work_dir, 'train_val'), 'train_val')
        self.test_writer = SummaryWriter(os.path.join(self.work_dir, 'test'), 'test')
        self.global_step = 0
        self.load_model()
        self.load_optimizer()
        self.best_acc = 0

    def save_arg(self):
        self.num_class = int(self.arg.data_name.split('_')[1])

        self.reg = coef_list(init=0.01, final=1, pre_epoch=self.arg.pre_epoch,
                             inc_epoch=self.arg.num_epoch, num_epoch=self.arg.num_epoch, kind=self.arg.inc_mode)

        self.arg.graph_args = str2list(self.arg.graph_args, flag='simple')
        self.arg.model_args = str2list(self.arg.model_args, flag='deep')
        self.data_path = os.path.join(self.arg.data_dir, self.arg.data_name.split('_pw')[0], self.arg.data_name)
        self.graph_name = get_graph_name(**self.arg.graph_args)
        netw = 'C{}k{}G{}'.format(
            '.'.join([str(s) for s in self.arg.model_args['hidden1']]),
            '.'.join([str(s) for s in self.arg.model_args['kernels']]),
            '.'.join([str(s) for s in self.arg.model_args['hidden2']]),
        )
        losses = 'lr{:g}m{:g}_{:d}.{:d}{}_O{:g}R{:g}G{:g}S0'.format(
            self.arg.base_lr, self.arg.base_lr_mask,
            self.arg.pre_epoch, self.arg.num_epoch, self.arg.inc_mode,
            self.arg.LO, self.arg.LR, self.arg.LG)
        self.arg.exp_name = '__'.join([losses, self.arg.exp_name])

        self.model_name = 't{}__{}'.format(time.strftime('%Y%m%d%H%M%S'), self.arg.exp_name)
        self.work_dir = os.path.join(self.arg.save_dir, self.arg.data_name, f'split{self.arg.split_seed}',
                                     f'seed{self.arg.seed}', f'fold{self.arg.fold}', self.model_name)
        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(os.path.join(self.work_dir, 'epoch'), exist_ok=True)  # save pt, pkl
        self.output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.print_log(',\t'.join([self.graph_name, netw, self.arg.exp_name]))

        # copy all files
        pwd = os.path.dirname(os.path.realpath(__file__))
        copytree(pwd, os.path.join(self.work_dir, 'code'), symlinks=False, ignore=ignore_patterns('__pycache__'), dirs_exist_ok=True)
        arg_dict = vars(self.arg)
        with open(os.path.join(self.work_dir, 'config.yaml'), 'w') as f:
            yaml.dump(arg_dict, f)

    def load_data(self):
        self.print_log('Load data.')
        Feeder = import_class(self.arg.feeder)
        train_set = Feeder(fold=self.arg.fold, split_seed=self.arg.split_seed,
                           out_dir=self.data_path, mode='train', graph_arg=self.arg.graph_args, **self.arg.train_feeder_args)
        test_set = Feeder(fold=self.arg.fold, split_seed=self.arg.split_seed,
                          out_dir=self.data_path, mode='test', graph_arg=self.arg.graph_args, **self.arg.test_feeder_args)
        self.DiffNode = Prior(train_set, device=self.output_device)
        self.print_log('prior knowledge sparsity:\nNode: {}/{:d}={:.4f}'.format(
            self.DiffNode.sum(), self.DiffNode.shape[0], self.DiffNode.sum() / self.DiffNode.shape[0],
        ))

        self.data_loader = dict()
        # train_sampler = WeightedRandomSampler(weights=train_set.samples_weights, num_samples=len(train_set))
        # self.data_loader['train'] = DataLoader(
        #     dataset=train_set, batch_size=self.arg.batch_size, num_workers=self.arg.num_worker,
        #     shuffle=False, drop_last=True, sampler=train_sampler)
        self.data_loader['train'] = DataLoader(
            dataset=train_set, batch_size=self.arg.batch_size, num_workers=self.arg.num_worker,
            shuffle=True, drop_last=True)
        self.data_loader['train_val'] = DataLoader(
            dataset=train_set, batch_size=self.arg.batch_size, num_workers=self.arg.num_worker,
            shuffle=False, drop_last=False)
        self.data_loader['test'] = DataLoader(
            dataset=test_set, batch_size=self.arg.test_batch_size, num_workers=self.arg.num_worker,
            shuffle=False, drop_last=False)

    def load_model(self):
        self.loss = nn.CrossEntropyLoss(reduction="mean")
        self.losses = nn.CrossEntropyLoss(reduction="none")
        self.mask = import_class(self.arg.model).CausalMask(
            patch_num=self.data_loader['train'].dataset.P, channel=self.arg.model_args['hidden1'][-1],
        ).cuda(self.output_device)
        self.model = import_class(self.arg.model).CausalNet(
            num_class=self.num_class, **self.arg.model_args
        ).cuda(self.output_device)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer_mask = optim.SGD(self.mask.parameters(),
                                            lr=self.arg.base_lr_mask, weight_decay=self.arg.weight_decay,
                                            momentum=0.9, nesterov=self.arg.nesterov)
            self.optimizer = optim.SGD(self.model.parameters(),
                                       lr=self.arg.base_lr, weight_decay=self.arg.weight_decay,
                                       momentum=0.9, nesterov=self.arg.nesterov)
        elif self.arg.optimizer == 'Adam':
            self.optimizer_mask = optim.Adam(self.mask.parameters(),
                                             lr=self.arg.base_lr_mask, weight_decay=self.arg.weight_decay)
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=self.arg.base_lr, weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

        if self.arg.scheduler == 'auto':
            self.lr_scheduler = ReduceLROnPlateau(self.optimizer, verbose=True,
                                                  patience=self.arg.stepsize, factor=self.arg.gamma)
            self.lr_scheduler_mask = ReduceLROnPlateau(self.optimizer_mask, verbose=True,
                                                       patience=self.arg.stepsize, factor=self.arg.gamma)
        elif self.arg.scheduler == 'step':
            self.lr_scheduler = StepLR(self.optimizer, step_size=self.arg.stepsize, gamma=self.arg.gamma)
            self.lr_scheduler_mask = StepLR(self.optimizer_mask, step_size=self.arg.stepsize, gamma=self.arg.gamma)
        else:
            raise ValueError()

    def start(self):
        for epoch in range(self.arg.start_epoch, self.arg.pre_epoch):
            # save_model = ((epoch + 1) % self.arg.save_interval == 0) or (epoch + 1 == self.arg.num_epoch)
            save_model = False
            self.train_emb(epoch, save_model=save_model)
            with torch.no_grad():
                self.eval_emb(epoch, save_score=self.arg.save_score, loader_name=['train_val', 'test'])

        for epoch in range(self.arg.pre_epoch, self.arg.num_epoch):
            # save_model = ((epoch + 1) % self.arg.save_interval == 0) or (epoch + 1 == self.arg.num_epoch)
            save_model = False
            self.train(epoch, save_model=save_model)
            with torch.no_grad():
                self.eval(epoch, save_score=self.arg.save_score, loader_name=['train_val', 'test'])

        self.print_log(f'best acc: {self.best_acc}, model_name: {self.model_name}')

        # settle results
        ss = SettleResults(self.data_loader['test'].dataset.out_dir, self.work_dir, self.arg.exp_name)
        ss.concat_trend_scores(num_epoch=self.arg.num_epoch, start_epoch=0,
                               metrics=['acc', 'pre', 'sen', 'spe', 'f1'], phase='test', ave='micro')
        ss.merge_pkl(num_epoch=self.arg.num_epoch, start_epoch=0,
                     type_list=['test_score', 'train_val_score'])
        ss.confusion_matrix(out_path=os.path.join(self.work_dir, 'CM.png'))
        self.print_log(f'finish: {self.work_dir}')

    def train_emb(self, epoch, save_model=False):
        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']
        self.train_writer.add_scalar(self.model_name + '/epoch', epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        try:
            process = tqdm(loader, ncols=50)
        except KeyboardInterrupt:
            process.close()
            raise
        process.close()

        loss_value = []
        scores, true = [], []
        self.print_log('\tReg coefficient: {:.4f}.'.format(self.reg[epoch]))
        for batch_idx, (data, edges, label, index) in enumerate(process):
            self.global_step += 1

            x_node, edge, label = self.converse2tensor(data, edges, label)
            timer['dataloader'] += self.split_time()

            x_new = self.model.emb(x_node)
            yw = self.model.prediction_whole(x_new, edge)
            lossC = self.losses(yw, label)
            lossAll = lossC.mean()
            self.optimizer.zero_grad()
            lossAll.backward()
            self.optimizer.step()

            loss_value.append(lossC.mean().item())
            timer['model'] += self.split_time()

            true.extend(self.data_loader['train'].dataset.label[index])
            scores.extend(yw.view(-1).data.cpu().numpy())

            value, predict_label = torch.max(yw.data, 1)
            acc = torch.mean((predict_label == label.data).float())
            self.train_writer.add_scalar(self.model_name + '/acc', acc.item(), self.global_step)
            self.train_writer.add_scalar(self.model_name + '/loss_causal', lossC.mean().item(), self.global_step)
            self.train_writer.add_scalar(self.model_name + '/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
            timer['statistics'] += self.split_time()

        # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log('\tMean training loss: {:.4f}.'.format(np.mean(loss_value)))
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}, [Evaluate]{statistics}'.format(**proportion))

        if save_model:
            state_dict = self.model.state_dict()
            torch.save(state_dict, os.path.join(self.work_dir, 'epoch', 'epoch-' + str(epoch + 1) + '.pt'))

    def eval_emb(self, epoch, save_score=False, loader_name=['test']):
        self.model.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        for ln in loader_name:
            loss_value = []
            score_dict = {}

            try:
                process = tqdm(self.data_loader[ln], ncols=50)
            except KeyboardInterrupt:
                process.close()
                raise
            # process.close()

            for batch_idx, (data, edges, label, index) in enumerate(process):
                x_node, edge, label = self.converse2tensor(data, edges, label)

                x_new = self.model.emb(x_node)
                yw = self.model.prediction_whole(x_new, edge)
                lossC = self.losses(yw, label)

                loss_value.extend(lossC.data.cpu().numpy())
                sub_list = self.data_loader[ln].dataset.sample_name[index] if len(index) > 1 \
                    else [self.data_loader[ln].dataset.sample_name[index]]
                score_dict.update(dict(zip(sub_list, yw.data.cpu().numpy())))

            loss = np.mean(loss_value)
            accuracy = self.data_loader[ln].dataset.top_k(np.array(list(score_dict.values())), 1)
            if accuracy > self.best_acc and ln == 'test':
                self.best_acc = accuracy
            self.print_log(f'ln: {ln}, acc: {accuracy}, model: {self.model_name}')
            if ln == 'train_val':
                if self.arg.scheduler == 'auto':
                    self.lr_scheduler.step(loss)
                elif self.arg.scheduler == 'step':
                    self.lr_scheduler.step()
                else:
                    raise ValueError()
                self.val_writer.add_scalar(self.model_name + '/acc', accuracy, self.global_step)
                self.val_writer.add_scalar(self.model_name + '/loss_causal', lossC.mean().item(), self.global_step)
            elif ln == 'test':
                self.test_writer.add_scalar(self.model_name + '/acc', accuracy, self.global_step)
                self.test_writer.add_scalar(self.model_name + '/loss_causal', lossC.mean().item(), self.global_step)

            self.print_log('\tMean {} loss of {} batches: {}.'.format(
                ln, len(self.data_loader[ln]), np.mean(loss_value)))

            if save_score:
                with open(os.path.join(self.work_dir, 'epoch', 'epoch{}_{}_score.pkl'.format(epoch + 1, ln)), 'wb') as f:
                    pickle.dump(score_dict, f)

    def train(self, epoch, save_model=False):
        self.mask.train()
        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']
        self.train_writer.add_scalar(self.model_name + '/epoch', epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        try:
            process = tqdm(loader, ncols=50)
        except KeyboardInterrupt:
            process.close()
            raise
        process.close()

        scores, true = [], []
        self.print_log('\tReg coefficient: {:.4f}.'.format(self.reg[epoch]))
        for batch_idx, (data, edges, label, index) in enumerate(process):
            self.global_step += 1

            x_node, edge, label = self.converse2tensor(data, edges, label)
            timer['dataloader'] += self.split_time()

            # fix gcn, update mask
            masks, sparsity = self.mask(train=True)
            for name, p in self.model.named_parameters():
                p.requires_grad = False

            x_new = self.model.emb(x_node)
            yc = self.model.prediction_causal(x_new, edge, masks)
            lossC = self.losses(yc, label)
            if self.arg.LR > 0:
                yr = self.model.prediction_combine(x_new, edge, masks)
                lossR = torch.stack([self.losses(yr[i], label[i].repeat(len(index))) for i in range(len(index))], dim=0)
                lossAll = lossC.mean() + self.arg.LR * lossR.mean()
                self.train_writer.add_scalar(self.model_name + '/loss_comb', lossR.mean().item(), self.global_step)
            else:
                lossAll = lossC.mean()

            if self.arg.LO > 0:
                yo = self.model.prediction_counterfactual(x_new, edge, masks)
                lossO = - self.losses(yo, label)
                lossAll += self.arg.LO * lossO.mean()
                self.train_writer.add_scalar(self.model_name + '/loss_opp', lossO.mean().item(), self.global_step)
            if self.reg[epoch] < 1 and self.arg.LG > 0:
                if self.arg.LR > 0:
                    guide_node = self.guide(M=masks[0] * masks[1])
                else:
                    guide_node = self.guide(M=masks[0])
                lossAll += (1 - self.reg[epoch]) * self.arg.LG * guide_node
                self.train_writer.add_scalar(self.model_name + '/guide', guide_node.item(), self.global_step)

            self.optimizer_mask.zero_grad()
            lossAll.backward()
            self.optimizer_mask.step()

            for name, p in self.model.named_parameters():
                p.requires_grad = True

            # update gcn
            masks, sparsity = self.mask(train=False)
            masks = [mm.detach() for mm in masks]

            x_new = self.model.emb(x_node)
            yc = self.model.prediction_causal(x_new, edge, masks)
            lossC = self.losses(yc, label)
            if self.arg.LR > 0:
                yr = self.model.prediction_combine(x_new, edge, masks)
                lossR = torch.stack([self.losses(yr[i], label[i].repeat(len(index))) for i in range(len(index))], dim=0)
                lossAll = lossC.mean() + self.arg.LR * lossR.mean()
                self.train_writer.add_scalar(self.model_name + '/loss_comb', lossR.mean().item(), self.global_step)
            elif self.arg.LR == 0:
                lossAll = lossC.mean()
            else:
                raise ValueError

            self.optimizer.zero_grad()
            lossAll.backward()
            self.optimizer.step()

            timer['model'] += self.split_time()

            true.extend(self.data_loader['train'].dataset.label[index])
            scores.extend(yc.view(-1).data.cpu().numpy())

            value, predict_label = torch.max(yc.data, 1)
            acc = torch.mean((predict_label == label.data).float())
            self.train_writer.add_scalar(self.model_name + '/acc', acc.item(), self.global_step)
            if self.arg.LR > 0:
                M1_node, M2_node, M1_edge, M2_edge = masks
                self.train_writer.add_scalar(self.model_name + '/spar_node1', M1_node.mean().item(), self.global_step)
                self.train_writer.add_scalar(self.model_name + '/spar_edge1', M1_edge.mean().item(), self.global_step)
                self.train_writer.add_scalar(self.model_name + '/spar_node12', (M1_node * M2_node).mean().item(), self.global_step)
                self.train_writer.add_scalar(self.model_name + '/spar_edge12', (M1_edge * M2_edge).mean().item(), self.global_step)
            else:
                M1_node, M1_edge = masks
                self.train_writer.add_scalar(self.model_name + '/spar_node1', M1_node.mean().item(), self.global_step)
                self.train_writer.add_scalar(self.model_name + '/spar_edge1', M1_edge.mean().item(), self.global_step)

            self.train_writer.add_scalar(self.model_name + '/loss_causal', lossC.mean().item(), self.global_step)
            self.train_writer.add_scalar(self.model_name + '/sparsity', sparsity.item(), self.global_step)
            self.train_writer.add_scalar(self.model_name + '/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
            self.train_writer.add_scalar(self.model_name + '/lr_mask', self.optimizer_mask.param_groups[0]['lr'], self.global_step)
            timer['statistics'] += self.split_time()

        # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}, [Evaluate]{statistics}'.format(**proportion))

        if save_model and epoch in [20, 24, self.arg.num_epoch-1]:
            state_dict = self.mask.state_dict()
            torch.save(state_dict, os.path.join(self.work_dir, 'epoch', f'save{epoch + 1}_mask.pt'))

    def eval(self, epoch, save_score=False, loader_name=['test']):
        self.mask.eval()
        self.model.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        for ln in loader_name:
            loss_value1, loss_value2 = [], []
            score_dict = {}

            try:
                process = tqdm(self.data_loader[ln], ncols=50)
            except KeyboardInterrupt:
                process.close()
                raise
            # process.close()

            for batch_idx, (data, edges, label, index) in enumerate(process):
                x_node, edge, label = self.converse2tensor(data, edges, label)

                masks, sparsity = self.mask(train=False)

                x_new = self.model.emb(x_node)
                yc = self.model.prediction_causal(x_new, edge, masks)
                lossC = self.losses(yc, label)

                yo = self.model.prediction_counterfactual(x_new, edge, masks)
                lossO = - self.losses(yo, label)

                if self.arg.LR > 0:
                    guide_node = self.guide(M=masks[0] * masks[1])
                    yr = self.model.prediction_combine(x_new, edge, masks)
                    lossR = torch.stack([self.losses(yr[i], label[i].repeat(len(index))) for i in range(len(index))], dim=0)
                    loss1 = lossC + torch.mean(lossR, dim=1) + self.arg.LO * lossO + \
                            (1 - self.reg[epoch]) * self.arg.LG * guide_node
                    loss2 = lossC + torch.mean(lossR, dim=1)
                elif self.arg.LR == 0:
                    guide_node = self.guide(M=masks[0])
                    loss1 = lossC + self.arg.LO * lossO + \
                            (1 - self.reg[epoch]) * self.arg.LG * guide_node
                    loss2 = lossC
                else:
                    raise ValueError

                loss_value1.extend(loss1.data.cpu().numpy())
                loss_value2.extend(loss2.data.cpu().numpy())
                sub_list = self.data_loader[ln].dataset.sample_name[index] if len(index) > 1 \
                    else [self.data_loader[ln].dataset.sample_name[index]]
                score_dict.update(dict(zip(sub_list, yc.data.cpu().numpy())))

            loss1 = np.mean(loss_value1)
            loss2 = np.mean(loss_value2)
            accuracy = self.data_loader[ln].dataset.top_k(np.array(list(score_dict.values())), 1)
            if accuracy > self.best_acc and ln == 'test':
                self.best_acc = accuracy
            self.print_log(f'ln: {ln}, acc: {accuracy}, model: {self.model_name}')
            if ln == 'train_val':
                if self.arg.scheduler == 'auto':
                    self.lr_scheduler_mask.step(loss2)
                    self.lr_scheduler.step(loss1)
                elif self.arg.scheduler == 'step':
                    self.lr_scheduler_mask.step()
                    self.lr_scheduler.step()
                else:
                    raise ValueError()
                self.val_writer.add_scalar(self.model_name + '/acc', accuracy, self.global_step)
                self.val_writer.add_scalar(self.model_name + '/loss_causal', lossC.mean().item(), self.global_step)
                self.val_writer.add_scalar(self.model_name + '/loss_opp', lossO.mean().item(), self.global_step)
                if self.arg.LR > 0:
                    self.val_writer.add_scalar(self.model_name + '/loss_comb', lossR.mean().item(), self.global_step)
                self.val_writer.add_scalar(self.model_name + '/guide', guide_node.item(), self.global_step)
            elif ln == 'test':
                self.test_writer.add_scalar(self.model_name + '/acc', accuracy, self.global_step)
                self.test_writer.add_scalar(self.model_name + '/loss_causal', lossC.mean().item(), self.global_step)
                self.test_writer.add_scalar(self.model_name + '/loss_opp', lossO.mean().item(), self.global_step)
                if self.arg.LR > 0:
                    self.test_writer.add_scalar(self.model_name + '/loss_comb', lossR.mean().item(), self.global_step)
                self.test_writer.add_scalar(self.model_name + '/guide', guide_node.item(), self.global_step)

            if save_score:
                with open(os.path.join(self.work_dir, 'epoch', 'epoch{}_{}_score.pkl'.format(epoch + 1, ln)), 'wb') as f:
                    pickle.dump(score_dict, f)

    def converse2tensor(self, data, edges, label):
        data = torch.FloatTensor(data.float()).cuda(self.output_device)
        label = torch.LongTensor(label.long()).cuda(self.output_device)
        all_edge = torch.FloatTensor(edges.float()).cuda(self.output_device)
        return data, all_edge, label

    def guide(self, M):
        aa = torch.norm(self.DiffNode.squeeze(1) - M[:, 0], p=2)
        return aa

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open(os.path.join(self.work_dir, 'log.txt'), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time


if __name__ == '__main__':
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    seed_torch(arg.seed)
    processor = Processor(arg)
    processor.start()

