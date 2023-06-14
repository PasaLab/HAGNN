import datetime
import dgl
import errno
import numpy as np
import os
import pickle
import random
import torch
import copy
import torch as th

from dgl.data.utils import download, get_download_dir, _get_dgl_url
from pprint import pprint
from scipy import sparse
from scipy import io as sio

import sys

sys.path.append('../../')
from scripts.data_loader import data_loader,data_loader4dag,data_loader4largefreebase


def set_random_seed(seed=0):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def mkdir_p(path, log=True):
    """Create a directory for the specified path.
    Parameters
    ----------
    path : str
        Path name
    log : bool
        Whether to print result for directory creation
    """
    try:
        os.makedirs(path)
        if log:
            print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print('Directory {} already exists.'.format(path))
        else:
            raise


def get_date_postfix():
    """Get a date based postfix for directory name.
    Returns
    -------
    post_fix : str
    """
    dt = datetime.datetime.now()
    post_fix = '{}_{:02d}-{:02d}-{:02d}'.format(
        dt.date(), dt.hour, dt.minute, dt.second)

    return post_fix


def setup_log_dir(args, sampling=False):
    """Name and create directory for logging.
    Parameters
    ----------
    args : dict
        Configuration
    Returns
    -------
    log_dir : str
        Path for logging directory
    sampling : bool
        Whether we are using sampling based training
    """
    date_postfix = get_date_postfix()
    log_dir = os.path.join(
        args['log_dir'],
        '{}_{}'.format(args['dataset'], date_postfix))

    if sampling:
        log_dir = log_dir + '_sampling'

    mkdir_p(log_dir)
    return log_dir


# The configuration below is from the paper.
default_configure = {
    'lr': 0.005,  # Learning rate
    'num_heads': [8],  # Number of attention heads for node-level attention 层数和[]的长度有关
    'hidden_units': 8,
    'dropout': 0.6,
    'weight_decay': 0.001,
}

sampling_configure = {
    'batch_size': 20
}


def setup(args):
    args.update(default_configure)
    set_random_seed(args['seed'])
    # args['log_dir'] = setup_log_dir(args)
    return args


def setup_for_sampling(args):
    args.update(default_configure)
    args.update(sampling_configure)
    set_random_seed()
    args['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args['log_dir'] = setup_log_dir(args, sampling=True)
    return args


def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask.byte()


def load_acm(feat_type=0):
    dl = data_loader('../../data/ACM')
    link_type_dic = {0: 'pp', 1: '-pp', 2: 'pa', 3: 'ap', 4: 'ps', 5: 'sp', 6: 'pt', 7: 'tp'}
    paper_num = dl.nodes['count'][0]
    data_dic = {}
    for link_type in dl.links['data'].keys():
        src_type = str(dl.links['meta'][link_type][0])
        dst_type = str(dl.links['meta'][link_type][1])
        data_dic[(src_type, link_type_dic[link_type], dst_type)] = dl.links['data'][link_type].nonzero()
    hg = dgl.heterograph(data_dic)

    # paper feature
    if feat_type == 0:
        '''preprocessed feature'''
        features = th.FloatTensor(dl.nodes['attr'][0])
    else:
        '''one-hot'''
        features = th.FloatTensor(np.eye(paper_num))

    # author labels

    labels = dl.labels_test['data'][:paper_num] + dl.labels_train['data'][:paper_num]
    labels = [np.argmax(l) for l in labels]  # one-hot to value
    labels = th.LongTensor(labels)

    num_classes = 3

    train_valid_mask = dl.labels_train['mask'][:paper_num]
    test_mask = dl.labels_test['mask'][:paper_num]
    train_valid_indices = np.where(train_valid_mask == True)[0]
    split_index = int(0.7 * np.shape(train_valid_indices)[0])
    train_indices = train_valid_indices[:split_index]
    valid_indices = train_valid_indices[split_index:]
    train_mask = copy.copy(train_valid_mask)
    valid_mask = copy.copy(train_valid_mask)
    train_mask[valid_indices] = False
    valid_mask[train_indices] = False
    test_indices = np.where(test_mask == True)[0]

    meta_paths = [['pp', 'ps', 'sp'], ['-pp', 'ps', 'sp'], ['pa', 'ap'], ['ps', 'sp'], ['pt', 'tp']]
    return hg, features, labels, num_classes, train_indices, valid_indices, test_indices, \
           th.BoolTensor(train_mask), th.BoolTensor(valid_mask), th.BoolTensor(test_mask), meta_paths, \
           dl

def load_acm2(feat_type=0):
    dl = data_loader4dag('../../data/ACM2')
    link_type_dic = {0: 'pp', 1: 'ap',3:'pa'}
    paper_num = dl.nodes['count'][0]
    data_dic = {}
    for link_type in dl.links['data'].keys():
        src_type = str(dl.links['meta'][link_type][0])
        dst_type = str(dl.links['meta'][link_type][1])
        data_dic[(src_type, link_type_dic[link_type], dst_type)] = dl.links['data'][link_type].nonzero()
    hg = dgl.heterograph(data_dic)

    # paper feature
    if feat_type == 0:
        '''preprocessed feature'''
        features = th.FloatTensor(dl.nodes['attr'][0])
    else:
        '''one-hot'''
        features = th.FloatTensor(np.eye(paper_num))

    # author labels

    labels = dl.labels_test['data'][:paper_num] + dl.labels_train['data'][:paper_num]
    labels = [np.argmax(l) for l in labels]  # one-hot to value
    labels = th.LongTensor(labels)

    num_classes = 3

    train_valid_mask = dl.labels_train['mask'][:paper_num]
    test_mask = dl.labels_test['mask'][:paper_num]
    train_valid_indices = np.where(train_valid_mask == True)[0]
    split_index = int(0.7 * np.shape(train_valid_indices)[0])
    train_indices = train_valid_indices[:split_index]
    valid_indices = train_valid_indices[split_index:]
    train_mask = copy.copy(train_valid_mask)
    valid_mask = copy.copy(train_valid_mask)
    train_mask[valid_indices] = False
    valid_mask[train_indices] = False
    test_indices = np.where(test_mask == True)[0]

    meta_paths = [['pp', 'ps', 'sp'], ['-pp', 'ps', 'sp'], ['pa', 'ap'], ['ps', 'sp'], ['pt', 'tp']]
    return hg, features, labels, num_classes, train_indices, valid_indices, test_indices, \
           th.BoolTensor(train_mask), th.BoolTensor(valid_mask), th.BoolTensor(test_mask), meta_paths, \
           dl

def load_freebase(feat_type=1):
    dl = data_loader4largefreebase('../../data/Freebase')
    link_type_dic = {8: '22', 16: '42', 32: '72', 7: '21', 6: '20', 5: '11', 15: '41',
                     18: '44', 17: '43', 20: '46', 19: '45', 13: '35', 22: '51', 14: '40',
                     1: '01', 30: '70', 12: '33', 25: '62', 10: '25', 9: '23', 4: '06',
                     31: '71', 11: '31', 21: '47', 23: '55', 29: '67', 33: '73', 3: '05',
                     27: '65', 2: '03', 24: '61', 34: '75', 35: '77', 26: '63', 28: '66', 0: '00'}
    book_num = dl.nodes['count'][0]
    data_dic = {}
    for link_type in dl.links['data'].keys():
        src_type = str(dl.links['meta'][link_type][0])
        dst_type = str(dl.links['meta'][link_type][1])
        data_dic[(src_type, link_type_dic[link_type], dst_type)] = dl.links['data'][link_type].nonzero()
    hg = dgl.heterograph(data_dic)


    return hg

    # 返回的的feature只是author的，其他的都在dl里，和源数据文件夹一致
def load_dblp(feat_type=0):
    prefix = '../../data/DBLP'
    dl = data_loader(prefix)
    link_type_dic = {0: 'ap', 1: 'pc', 2: 'pt', 3: 'pa', 4: 'cp', 5: 'tp'}
    author_num = dl.nodes['count'][0]
    data_dic = {}
    # [data]本身是一个稀疏图 但是这边没用这个稀疏图 而是用了dgl
    for link_type in dl.links['data'].keys():
        src_type = str(dl.links['meta'][link_type][0])
        dst_type = str(dl.links['meta'][link_type][1])
        data_dic[(src_type, link_type_dic[link_type], dst_type)] = dl.links['data'][link_type].nonzero()
    hg = dgl.heterograph(data_dic)

    # author feature
    if feat_type == 0:
        '''preprocessed feature'''
        features = th.FloatTensor(dl.nodes['attr'][0]) # N_tar * F
    else:
        '''one-hot'''
        # indices = np.vstack((np.arange(author_num), np.arange(author_num)))
        # indices = th.LongTensor(indices)
        # values = th.FloatTensor(np.ones(author_num))
        # features = th.sparse.FloatTensor(indices, values, th.Size([author_num,author_num]))
        features = th.FloatTensor(np.eye(author_num))

    # author labels

    labels = dl.labels_test['data'][:author_num] + dl.labels_train['data'][:author_num]
    labels = [np.argmax(l) for l in labels]  # one-hot to value
    labels = th.LongTensor(labels)

    num_classes = 4

    train_valid_mask = dl.labels_train['mask'][:author_num]
    test_mask = dl.labels_test['mask'][:author_num]
    train_valid_indices = np.where(train_valid_mask == True)[0]
    split_index = int(0.7 * np.shape(train_valid_indices)[0])
    train_indices = train_valid_indices[:split_index]
    valid_indices = train_valid_indices[split_index:]
    train_mask = copy.copy(train_valid_mask)
    valid_mask = copy.copy(train_valid_mask)
    train_mask[valid_indices] = False
    valid_mask[train_indices] = False
    test_indices = np.where(test_mask == True)[0]

    meta_paths = [['ap', 'pa'], ['ap', 'pt', 'tp', 'pa'], ['ap', 'pc', 'cp', 'pa']]
    # meta_paths = [['ap', 'pt', 'tp', 'pa']]
    # meta_paths = [['ap', 'pa'], ['ap', 'pc', 'cp', 'pa']]
    return hg, features, labels, num_classes, train_indices, valid_indices, test_indices, \
           th.BoolTensor(train_mask), th.BoolTensor(valid_mask), th.BoolTensor(test_mask), meta_paths, \
           dl

def load_dblp2(feat_type=0):
    prefix = '../../data/DBLP2'
    dl = data_loader4dag(prefix) #{3: (1, 0), 5: (3, 1), 4: (2, 1), 2: (1, 1), 1: (1, 1), 0: (0, 0), 9: (0, 1), 11: (1, 3), 10: (1, 2)}
    # link_type_dic = {0: 'ap', 1: 'pc', 2: 'pt', 3: 'pa', 4: 'cp', 5: 'tp'}
    link_type_dic = {3: '10', 5: '31', 4: '21', 2: '11', 1: '11',0: '00', 9: '01', 11: '13', 10: '12'}
    author_num = dl.nodes['count'][0]
    data_dic = {}
    # [data]本身是一个稀疏图 但是这边没用这个稀疏图 而是用了dgl
    for link_type in dl.links['data'].keys():
        src_type = str(dl.links['meta'][link_type][0])
        dst_type = str(dl.links['meta'][link_type][1])
        data_dic[(src_type, link_type_dic[link_type], dst_type)] = dl.links['data'][link_type].nonzero()
    hg = dgl.heterograph(data_dic)

    # author feature
    if feat_type == 0:
        '''preprocessed feature'''
        features = th.FloatTensor(dl.nodes['attr'][0]) # N_tar * F
    else:
        '''one-hot'''
        # indices = np.vstack((np.arange(author_num), np.arange(author_num)))
        # indices = th.LongTensor(indices)
        # values = th.FloatTensor(np.ones(author_num))
        # features = th.sparse.FloatTensor(indices, values, th.Size([author_num,author_num]))
        features = th.FloatTensor(np.eye(author_num))

    # author labels

    labels = dl.labels_test['data'][:author_num] + dl.labels_train['data'][:author_num]
    labels = [np.argmax(l) for l in labels]  # one-hot to value
    labels = th.LongTensor(labels)

    num_classes = 4

    train_valid_mask = dl.labels_train['mask'][:author_num]
    test_mask = dl.labels_test['mask'][:author_num]
    train_valid_indices = np.where(train_valid_mask == True)[0]
    split_index = int(0.7 * np.shape(train_valid_indices)[0])
    train_indices = train_valid_indices[:split_index]
    valid_indices = train_valid_indices[split_index:]
    train_mask = copy.copy(train_valid_mask)
    valid_mask = copy.copy(train_valid_mask)
    train_mask[valid_indices] = False
    valid_mask[train_indices] = False
    test_indices = np.where(test_mask == True)[0]

    meta_paths = [['ap', 'pa'], ['ap', 'pt', 'tp', 'pa'], ['ap', 'pc', 'cp', 'pa']]
    # meta_paths = [['ap', 'pt', 'tp', 'pa']]
    # meta_paths = [['ap', 'pa'], ['ap', 'pc', 'cp', 'pa']]
    return hg, features, labels, num_classes, train_indices, valid_indices, test_indices, \
           th.BoolTensor(train_mask), th.BoolTensor(valid_mask), th.BoolTensor(test_mask), meta_paths, \
           dl

def load_imdb(feat_type=0):
    prefix = '../../data/IMDB'
    dl = data_loader(prefix)
    link_type_dic = {0: 'md', 1: 'dm', 2: 'ma', 3: 'am', 4: 'mk', 5: 'km'}
    movie_num = dl.nodes['count'][0]
    data_dic = {}
    for link_type in dl.links['data'].keys():
        src_type = str(dl.links['meta'][link_type][0])
        dst_type = str(dl.links['meta'][link_type][1])
        data_dic[(src_type, link_type_dic[link_type], dst_type)] = dl.links['data'][link_type].nonzero() #建图只需要右边的 所以Nonzero
    hg = dgl.heterograph(data_dic)

    # author feature
    if feat_type == 0:
        '''preprocessed feature'''
        features = th.FloatTensor(dl.nodes['attr'][0])
        # print(features.shape())
    else:
        '''one-hot'''
        # indices = np.vstack((np.arange(author_num), np.arange(author_num)))
        # indices = th.LongTensor(indices)
        # values = th.FloatTensor(np.ones(author_num))
        # features = th.sparse.FloatTensor(indices, values, th.Size([author_num,author_num]))
        features = th.FloatTensor(np.eye(movie_num))

    # author labels

    labels = dl.labels_test['data'][:movie_num] + dl.labels_train['data'][:movie_num]
    labels = th.FloatTensor(labels)

    num_classes = 5

    train_valid_mask = dl.labels_train['mask'][:movie_num]
    test_mask = dl.labels_test['mask'][:movie_num]
    train_valid_indices = np.where(train_valid_mask == True)[0]
    split_index = int(0.7 * np.shape(train_valid_indices)[0])
    train_indices = train_valid_indices[:split_index]
    valid_indices = train_valid_indices[split_index:]
    train_mask = copy.copy(train_valid_mask)
    valid_mask = copy.copy(train_valid_mask)
    train_mask[valid_indices] = False
    valid_mask[train_indices] = False
    test_indices = np.where(test_mask == True)[0]

    meta_paths = [['md', 'dm'], ['ma', 'am'], ['mk', 'km']]
    return hg, features, labels, num_classes, train_indices, valid_indices, test_indices, \
           th.BoolTensor(train_mask), th.BoolTensor(valid_mask), th.BoolTensor(test_mask), meta_paths, dl

def load_PubMed(feat_type=0):
    prefix = '../../data/PubMed'
    dl = data_loader4dag(prefix)
    link_type_dic = {0: 'gg', 1: 'gd', 2: 'dd', 3: 'cg', 4: 'cd', 5: 'cc', 6 :'cs', 7 :'sg', 8 :'sd', 9 :'ss',
                     11: 'dg',  13: 'gc', 14: 'dc',  16 :'sc', 17 :'gs', 18 :'ds'}
    author_num = dl.nodes['count'][0]
    data_dic = {}
    # [data]本身是一个稀疏图 但是这边没用这个稀疏图 而是用了dgl
    for link_type in dl.links['data'].keys():
        src_type = str(dl.links['meta'][link_type][0])
        dst_type = str(dl.links['meta'][link_type][1])
        data_dic[(src_type, link_type_dic[link_type], dst_type)] = dl.links['data'][link_type].nonzero()
    hg = dgl.heterograph(data_dic)
    return hg

def load_lastfm(feat_type=0):
    prefix = '../../data/LastFM'
    dl = data_loader4dag(prefix) #{0: (0, 1), 1: (0, 0), 2: (1, 2), 3: (1, 0), 5: (2, 1)}
    link_type_dic = {0: '01', 1: '00', 2: '12', 3: '10', 5: '21'}
    author_num = dl.nodes['count'][0]
    data_dic = {}
    # [data]本身是一个稀疏图 但是这边没用这个稀疏图 而是用了dgl
    for link_type in dl.links['data'].keys():
        src_type = str(dl.links['meta'][link_type][0])
        dst_type = str(dl.links['meta'][link_type][1])
        data_dic[(src_type, link_type_dic[link_type], dst_type)] = dl.links['data'][link_type].nonzero()
    hg = dgl.heterograph(data_dic)


    return hg

def load_data(dataset, feat_type=0):
    load_fun = None
    if dataset == 'ACM':
        load_fun = load_acm
    elif dataset == 'Freebase':
        feat_type = 1
        load_fun = load_freebase
    elif dataset == 'DBLP':
        load_fun = load_dblp
    elif dataset == 'IMDB':
        load_fun = load_imdb
    elif dataset == 'PubMed':
        load_fun = load_PubMed
    elif dataset == 'ACM2':
        load_fun = load_acm2
    elif dataset == 'DBLP2':
        load_fun = load_dblp2
    elif dataset == 'LastFM':
        load_fun = load_lastfm
    return load_fun(feat_type=feat_type)


class EarlyStopping(object):
    def __init__(self, patience=10):
        dt = datetime.datetime.now()
        self.filename = 'early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(
            dt.date(), dt.hour, dt.minute, dt.second)
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, acc, model):
        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
            self.save_checkpoint(model)
        elif (loss > self.best_loss) and (acc <= self.best_acc):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (loss <= self.best_loss) and (acc >= self.best_acc):
                self.save_checkpoint(model)
            self.best_loss = np.min((loss, self.best_loss))
            self.best_acc = np.max((acc, self.best_acc))
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        model.load_state_dict(torch.load(self.filename))
