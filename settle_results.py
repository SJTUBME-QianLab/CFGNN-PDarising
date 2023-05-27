import numpy as np
import pandas as pd
import os
import re
import yaml
import pickle
import platform
import nibabel as nib
from tools.utils import eval_metric_cl2, get_auc_cl2, get_CM, plot_confusion_matrix
import sklearn
import seaborn as sns
import itertools
from itertools import cycle, product
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
plt.rc('font', family='Times New Roman')
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300

raw_dirs = {
    'windows': {
        'data_path': '',
        'save_path': ''
    },
    'linux': {
        'data_path': '',
        'save_path': '',
    },
}
raw_dirs = raw_dirs[platform.system().lower()]
metrics = ['acc', 'pre', 'sen', 'spe', 'f1', 'auc']
ave = ''


class SettleResults:
    def __init__(self, data_dir, exp_dir, para_name=''):
        self.data_dir = data_dir
        self.exp_dir = exp_dir
        self.data_name = os.path.split(self.data_dir)[-1]
        self.para_name = '__'.join(para_name.split('__')[1:])
        self.patch_size = int(re.search('pw\d+', self.data_name).group().split('pw')[1])

        self.coor = np.load(os.path.join(self.data_dir, 'coordinates.npy'))
        with open(os.path.join(self.data_dir, 'label.pkl'), 'rb') as f:
            label_all, sub_name_all = pickle.load(f)
        label_dict = dict(zip(sub_name_all, label_all))
        self.label_df = pd.DataFrame(label_dict, index=['true_label'])
        if self.data_name.split('_')[1].startswith('pw'):
            self.num_class = 1
        else:
            self.num_class = int(self.data_name.split('_')[1])
            assert self.num_class == len(np.unique(label_all))

    def concat_trend_scores(self, num_epoch, metrics, start_epoch=0, phase='test', ave=''):
        with open(os.path.join(self.data_dir, 'label.pkl'), 'rb') as f:
            label_all, sub_name_all = pickle.load(f)
        label_dict = dict(zip(sub_name_all, label_all))
        label_df = pd.DataFrame(label_dict, index=['true_label'])
        if os.path.isfile(os.path.join(self.exp_dir, 'epoch', 'epoch1_test_score.pkl')):
            if not os.path.isfile(os.path.join(self.exp_dir, 'epoch', 'epoch%d_%s_score.pkl' % (num_epoch, phase))):
                print('------------------%s does not complete!' % self.exp_dir)
                return True
            evals_list = []
            for epo in range(1+start_epoch, 1+num_epoch):
                with open(os.path.join(self.exp_dir, 'epoch', 'epoch%d_%s_score.pkl' % (epo, phase)), 'rb') as f:
                    score_pred = pickle.load(f)
                score_pred_df = pd.DataFrame(score_pred, index=range(list(score_pred.values())[0].shape[0]))
                scores = pd.concat([score_pred_df, label_df], axis=0).dropna(axis=1)
                if self.num_class == 2:
                    evals = eval_metric_cl2(true=scores.iloc[-1, :], prob=scores.iloc[:-1, :])
                    evals_list.append([evals[kk] for kk in metrics])
                    evals_df = pd.DataFrame(evals_list, columns=metrics)
                else:
                    evals = eval_metric(true=scores.iloc[-1, :], prob=scores.iloc[:-1, :])
                    evals_list.append([evals[ave][kk] for kk in metrics])
                    evals_df = pd.DataFrame(evals_list, columns=[kk if kk[:3] == 'acc' else f'{kk}_{ave}' for kk in metrics])
            evals_df.to_csv(os.path.join(self.exp_dir, "trend_metrics_%s.csv" % phase), index=False)
        elif os.path.isfile(os.path.join(self.exp_dir, 'epoch', 'allepo_%d_test_score.pkl' % num_epoch)):
            with open(os.path.join(self.exp_dir, 'epoch', 'allepo_%d_%s_score.pkl' % (num_epoch, phase)), 'rb') as f:
                all_score_pred = pickle.load(f)
            evals_list = []
            for epo, score_pred in all_score_pred.items():
                score_pred_df = pd.DataFrame(score_pred)
                scores = pd.concat([score_pred_df, label_df], axis=0).dropna(axis=1)
                if self.num_class == 2:
                    evals = eval_metric_cl2(true=scores.iloc[-1, :], prob=scores.iloc[:-1, :])
                    evals_list.append([evals[kk] for kk in metrics])
                    evals_df = pd.DataFrame(evals_list, columns=metrics)
                else:
                    evals = eval_metric(true=scores.iloc[-1, :], prob=scores.iloc[:-1, :])
                    evals_list.append([evals[ave][kk] for kk in metrics])
                    evals_df = pd.DataFrame(evals_list, columns=[kk if kk[:3] == 'acc' else f'{kk}_{ave}' for kk in metrics])
            evals_df.to_csv(os.path.join(self.exp_dir, "trend_metrics_%s.csv" % phase), index=False)
        elif not os.path.isfile(os.path.join(self.exp_dir, 'log.txt')):
            print('------------------%s missing log.txt!' % self.exp_dir)
            return True  # failed

    def merge_pkl(self, num_epoch, type_list, start_epoch=0):
        # type_list = ['svm', 'test_score', 'train_val_score']
        for tt in type_list:
            if os.path.isfile(os.path.join(self.exp_dir, 'epoch', 'allepo_%d_%s.pkl' % (num_epoch - start_epoch, tt))):
                print('------------------%s have merged!' % tt)
                continue
            elif not os.path.isfile(os.path.join(self.exp_dir, 'epoch', 'epoch1_%s.pkl' % tt)):
                if not os.path.isfile(os.path.join(self.exp_dir, 'log.txt')):
                    print('------------------%s missing log.txt!' % tt)
                    return True  # failed
            if not os.path.isfile(os.path.join(self.exp_dir, 'epoch', 'epoch%d_%s.pkl' % (num_epoch, tt))):
                print('------------------%s does not complete!' % tt)
                return False
            tt_all = {}
            for epo in range(1+start_epoch, 1+num_epoch):
                with open(os.path.join(self.exp_dir, 'epoch', 'epoch%d_%s.pkl' % (epo, tt)), 'rb') as f:
                    score_pred = pickle.load(f)
                tt_all[epo] = score_pred
            with open(os.path.join(self.exp_dir, 'epoch', 'allepo_%d_%s.pkl' % (len(tt_all), tt)), 'wb') as f:
                pickle.dump(tt_all, f)
        assert set([kk for kk in os.listdir(os.path.join(self.exp_dir, 'epoch')) if kk[:3] == 'all']) \
               == set(['allepo_%d_%s.pkl' % (num_epoch - start_epoch, tt) for tt in type_list])
        if platform.system().lower() == 'linux' and os.path.isfile(os.path.join(self.exp_dir, 'epoch', 'epoch1_test_score.pkl')):
            os.system('rm %s/epoch*.pkl' % os.path.join(self.exp_dir, 'epoch'))
        elif platform.system().lower() == 'windows' and os.path.isfile(os.path.join(self.exp_dir, 'epoch', 'epoch1_test_score.pkl')):
            os.system('del \"%s\epoch*.pkl\"' % os.path.join(self.exp_dir, 'epoch'))
        print('finish merge: ', self.exp_dir)

    def get_final_score(self):
        file = [kk for kk in os.listdir(os.path.join(self.exp_dir, 'epoch'))
                if re.search('allepo_\d+_test_score.pkl', kk) is not None]
        if not (len(file) == 1 and file[0][:6] == 'allepo'):
            print(os.path.join(self.exp_dir, 'epoch'), str(file))
            return None
        with open(os.path.join(self.exp_dir, 'epoch', file[0]), 'rb') as f:
            score_pred_df = pickle.load(f)
        score_pred_df = pd.DataFrame(score_pred_df[len(score_pred_df)])
        with open(os.path.join(self.data_dir, 'label.pkl'), 'rb') as f:
            label_all, sub_name_all = pickle.load(f)
        label_dict = dict(zip(sub_name_all, label_all))
        label_df = pd.DataFrame(label_dict, index=['true_label'])
        scores = pd.concat([score_pred_df, label_df], axis=0).dropna(axis=1)
        return scores

    def confusion_matrix(self, out_path):
        scores = self.get_final_score()
        cm = get_CM(scores.iloc[-1, :], scores.iloc[:-1, :])
        classes = ['0', 'n0'] if self.data_name.split('_')[2] == '0n0' else [str(kk) for kk in self.data_name.split('_')[2]]
        plot_confusion_matrix(self.data_name + '\n' + self.para_name, out_path, cm=cm, classes=classes)

