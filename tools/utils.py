import os
import random
import argparse
import numpy as np
import pandas as pd
import torch
import sklearn
from sklearn.metrics import roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scipy import interp, stats
from scipy.sparse import coo_matrix
from scipy.spatial.distance import cdist
import seaborn as sns
import itertools
from itertools import cycle
import matplotlib.pyplot as plt
plt.rc('font', family='Times New Roman')
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2list(args, flag='simple'):
    if isinstance(args, str):
        args_new = eval(args)
        if flag == 'simple':
            return args_new
        elif flag == 'deep':
            return str2list(args_new, flag=flag)
    elif isinstance(args, dict):
        if flag == 'simple':
            return args
        elif flag == 'deep':
            for i, v in args.items():
                if isinstance(v, str):
                    args[i] = eval(v)
            return args
    else:
        raise ValueError('wrong arg type')


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def coef_list(init, final, pre_epoch, inc_epoch, num_epoch, kind='lin'):
    if final > 0:
        if pre_epoch == inc_epoch:  # 前0，后恒定；若pre_epoch=0则始终恒定
            reg = np.zeros(num_epoch)
            reg[inc_epoch:] = [final] * (num_epoch - inc_epoch)
        elif kind == 'log':  # 前0，后增长epoch ** step，终恒定
            reg = np.zeros(num_epoch)
            step = np.log(final / init) / np.log(inc_epoch - pre_epoch) if final > 0 else 0
            reg[pre_epoch - 1:inc_epoch] = [init * (epoch ** step) for epoch in range(inc_epoch - pre_epoch + 1)]
            reg[inc_epoch:] = [final] * (num_epoch - inc_epoch)
        elif kind == 'power':  # 前0，后增长step ** epoch，终恒定
            reg = np.zeros(num_epoch)
            reg[pre_epoch:inc_epoch] = np.logspace(start=np.log(init) / np.log(10), stop=np.log(final) / np.log(10),
                                                   num=inc_epoch - pre_epoch, endpoint=True)
            reg[inc_epoch:] = [final] * (num_epoch - inc_epoch)
        elif kind == 'lin':  # 前0，后增长epoch * step，终恒定
            reg = np.zeros(num_epoch)
            reg[pre_epoch:inc_epoch] = np.linspace(start=init, stop=final,
                                                   num=inc_epoch - pre_epoch, endpoint=True)
            reg[inc_epoch:] = [final] * (num_epoch - inc_epoch)
        else:
            raise ValueError('coef list kind?')
    else:  # final == 0
        reg = np.zeros(num_epoch)
    return reg


def get_graph_name(node_type=None, edge_type=None, dist_type='gau', sparse=None, adj_norm='DAD'):
    if sparse is not None:
        assert isinstance(sparse, dict)
        sparse_name = f"{sparse['level']}" + \
                      (f"th{sparse['thresh']}" if sparse['rate'] is None else f"ra{sparse['rate']}")
        name = f"N{node_type}_E{edge_type}_D{dist_type}_S{sparse_name}_{adj_norm}"
    else:
        name = f"N{node_type}_E{edge_type}_D{dist_type}_S{sparse}_{adj_norm}"
    return name


def Prior(train, device=None):
    pos = np.where(train.label == 1)[0]
    neg = np.where(train.label == 0)[0]
    nodes_data = train.node.reshape(len(train.label), train.P, -1)

    NodeMean = np.mean(nodes_data, axis=-1)  # [N,P,d]->[N,P]
    DiffNode = np.zeros((train.P, 20))
    for i in range(20):
        pos_i = random.sample(list(pos), int(np.floor(len(pos) * 0.5)))
        neg_i = random.sample(list(neg), int(np.floor(len(neg) * 0.5)))
        diff = abs(NodeMean[pos_i, :].mean(axis=0) - NodeMean[neg_i, :].mean(axis=0))
        DiffNode[:, i] = diff > diff.mean()
    DiffNode = np.mean(DiffNode, axis=-1, keepdims=True) > 0.5

    if device is not None:
        DiffNode = torch.FloatTensor(DiffNode).cuda(device)

    return DiffNode


def onehot_code(Y, num_class):
    Y = np.array(Y)
    Yc_onehot = np.zeros((len(Y), num_class))
    for i in range(num_class):
        Yc_onehot[np.where(Y == i)[0], i] = 1.0
    return Yc_onehot


def eval_metric_cl2(true, prob):
    num_class = int(max(true) + 1)
    assert num_class in prob.shape
    assert num_class == 2
    pred = list(prob.values.argmax(axis=0 if num_class == prob.shape[0] else 1))
    assert len(true) == len(pred)

    # acc
    acc = sum([pred[i] == true[i] for i in range(len(true))]) / len(true)

    # confusion matrix
    con_matrix = np.array(
        [[sum([pred[i] == k1 and true[i] == k2 for i in range(len(true))]) for k1 in range(num_class)] for k2 in range(num_class)])
    con_arr = con_matrix.ravel()

    SEN_cal = lambda tn, fp, fn, tp: tp / (tp + fn) if (tp + fn) != 0 else 0
    PRE_cal = lambda tn, fp, fn, tp: tp / (tp + fp) if (tp + fp) != 0 else 0
    SPE_cal = lambda tn, fp, fn, tp: tn / (tn + fp) if (tn + fp) != 0 else 0
    NPV_cal = lambda tn, fp, fn, tp: tn / (tn + fn) if (tn + fn) != 0 else 0

    sen = SEN_cal(*con_arr)
    pre = PRE_cal(*con_arr)
    spe = SPE_cal(*con_arr)
    npv = NPV_cal(*con_arr)
    f1 = 2 * pre * sen / (pre + sen)

    evals = {
        'confusion_matrix': con_matrix,
        'acc': acc, 'accuracy': acc,
        'pre': pre, 'precision': pre, 'ppv': pre,
        'npv': npv,
        'sen': sen, 'sensitivity': sen, 'recall': sen, 'tpr': sen,
        'spe': spe, 'specificity': spe, 'tnr': spe,
        'fpr': 1-spe,
        'f1': f1, 'f1_score': f1, 'f1score': f1,
    }

    return evals


def get_auc_cl2(true, prob):
    num_class = int(max(true) + 1)
    assert num_class == 2
    if isinstance(prob, pd.DataFrame):
        prob = prob.values
    if num_class == prob.shape[0]:  # num_class*num_sample -> num_sample*num_class
        prob = prob.T
    assert len(true) == prob.shape[0]
    prob = np.exp(prob) / np.sum(np.exp(prob), axis=1, keepdims=True).repeat(prob.shape[-1], axis=1)

    fpr, tpr, _ = roc_curve(true, prob[:, 1])
    roc_auc = auc(fpr, tpr)
    return roc_auc, fpr, tpr


def get_CM(true, prob):
    num_class = int(max(true) + 1)
    assert num_class in prob.shape
    pred = list(prob.values.argmax(axis=0 if num_class == prob.shape[0] else 1))
    assert len(true) == len(pred)

    con_matrix = sklearn.metrics.confusion_matrix(true, pred)
    acc = sklearn.metrics.accuracy_score(true, pred)
    print(con_matrix)
    print(acc)

    return con_matrix


def plot_confusion_matrix(name, out_path, cm, classes, cmap=plt.cm.GnBu):  # PuBu
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    num_cm = cm
    f, ax = plt.subplots()
    # if normalize:
    #     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #     print("Normalized confusion matrix")
    # else:
    #     print('Confusion matrix, without normalization')
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    ax.set_ylabel('True Label', family="Times New Roman", weight="bold", size=20)
    ax.set_xlabel('Predicted Label', family="Times New Roman", weight="bold", size=20)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    cb = f.colorbar(im, ax=ax)
    cb.ax.tick_params(labelsize=24)
    labels = cb.ax.get_xticklabels() + cb.ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks, classes)
    ax.set_yticks(tick_marks, classes, rotation=90)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.xaxis.set_ticks_position('top')

    # add number
    ax.set_ylim(len(classes) - 0.5, -0.5)
    # fmt = '.2f' if normalize else 'd'
    thresh = (cm.max() + cm.min()) / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(num_cm[i, j], 'd'),
                horizontalalignment="center", verticalalignment="bottom",
                color="white" if cm[i, j] > thresh else "black",
                family="Times New Roman", fontsize=24)
        ax.text(j, i, '({:.2f})'.format(cm[i, j]),
                horizontalalignment="center", verticalalignment="top",
                color="white" if cm[i, j] > thresh else "black",
                family="Times New Roman", fontsize=18)

    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)

    ax.set_title(name, fontsize=16)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    # plt.show()


# if __name__ == '__main__':
#     main()
