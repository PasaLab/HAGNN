import sys
sys.path.append('../../')
import time
import argparse
import random
from torch.backends import cudnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from utils.pytorchtools import EarlyStopping
from utils.data import load_data
from hgutils import load_data as load_data_hetero
#from utils.tools import index_generator, evaluate_results_nc, parse_minibatch
from data_playground import multi_metapath_graph,multi_metapath_graph_edgevalue_premeta
from GNN import myGAT
import dgl
def set_random_seed(seed, is_cuda):
    # random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if is_cuda:
        # logger.info('Using CUDA')
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.enabled = True
        cudnn.benchmark = False
        cudnn.deterministic = True
        # cudnn.benchmark = True
def sp_to_spt(mat):
    coo = mat.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

def mat2tensor(mat):
    if type(mat) is np.ndarray:
        return torch.from_numpy(mat).type(torch.FloatTensor)
    return sp_to_spt(mat)

def run_model_DBLP(args,seednow):
    feats_type = args.feats_type
    features_list, adjM, labels, train_val_test_idx, dl = load_data(args.dataset)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    features_list = [mat2tensor(features).to(device) for features in features_list]
    if feats_type == 0:
        in_dims = [features.shape[1] for features in features_list]
    elif feats_type == 1 or feats_type == 5:
        save = 0 if feats_type == 1 else 2
        in_dims = []#[features_list[0].shape[1]] + [10] * (len(features_list) - 1)
        for i in range(0, len(features_list)):
            if i == save:
                in_dims.append(features_list[i].shape[1])
            else:
                in_dims.append(10)
                features_list[i] = torch.zeros((features_list[i].shape[0], 10)).to(device)
    elif feats_type == 2 or feats_type == 4:
        save = feats_type - 2
        in_dims = [features.shape[0] for features in features_list]
        for i in range(0, len(features_list)):
            if i == save:
                in_dims[i] = features_list[i].shape[1]
                continue
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
    elif feats_type == 3:
        in_dims = [features.shape[0] for features in features_list]
        for i in range(len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
    labels = torch.FloatTensor(labels).to(device)
    train_idx = train_val_test_idx['train_idx']
    train_idx = np.sort(train_idx)
    val_idx = train_val_test_idx['val_idx']
    val_idx = np.sort(val_idx)
    test_idx = train_val_test_idx['test_idx']
    test_idx = np.sort(test_idx)
    
    # edge2type = {}
    # for k in dl.links['data']:
    #     for u,v in zip(*dl.links['data'][k].nonzero()):
    #         edge2type[(u,v)] = k
    # for i in range(dl.nodes['total']):
    #     edge2type[(i,i)] = len(dl.links['count'])

    g = dgl.DGLGraph(adjM)
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g = g.to(device)
    # e_feat = []
    # for u, v in zip(*g.edges()):
    #     u = u.cpu().item()
    #     v = v.cpu().item()
    #     e_feat.append(edge2type[(u,v)])
    # e_feat = torch.tensor(e_feat, dtype=torch.long).to(device)
    # print(len(dl.links['count']))
    hgorigin, _, _, _, _, _, _, _, _, _, _, _ = load_data_hetero(args.dataset, feat_type=0)
    # metapathes = [  # 为什么papap不收敛 两层 [['ma', 'am', 'ma', 'am']],
    #               [['am', 'md', 'dm', 'ma'],['am','ma']],
    #               [['km', 'ma', 'am','mk'],['km','mk']]]  #
    metapathes = [  # 为什么papap不收敛 两层 [['ma', 'am', 'ma', 'am']],
                  [['am', 'md', 'dm', 'ma']],
                  [['km', 'ma', 'am','mk']]]  #
    select = ['2', '3']
    # 不带先验的图
    hgs = []
    for meta in metapathes:
        hg, num_dic = multi_metapath_graph_edgevalue_premeta(hgorigin, meta)
        hgs.append(hg.to(device))
    split_list = list(num_dic.values())

    # # h2gcn
    # hg2,_ = multi_metapath_graph(hgorigin,metapathes2hop)
    # print("out")

    # 带先验的图
    # hg, num_dic = multi_metapath_graph_edgevalue(args.dataset, metapathes)
    hg = hg.to(device)

    # # h2gcn
    # hg2 = hg2.to(device)

    split_list = list(num_dic.values())

    loss = nn.BCELoss()
    num_classes = dl.labels_train['num_classes']
    heads = [args.num_heads] * args.num_layers + [1]
    net = myGAT(g,hgs, in_dims, args.hidden_dim, num_classes, args.num_layers,args.intralayers, heads, F.elu, args.dropout, args.dropout, args.slope, True, 0.00)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # training loop
    net.train()
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, save_path='checkpoint/checkpoint_{}_{}_{}.pt'.format(args.dataset, args.num_layers,seednow))
    for epoch in range(args.epoch):
        t_start = time.time()
        # training
        net.train()

        logits = net(select,split_list,features_list)
        logp = F.sigmoid(logits)
        train_loss = loss(logp[train_idx], labels[train_idx])

        # autograd
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        t_end = time.time()

        # print training info
        print('Epoch {:05d} | Train_Loss: {:.4f} | Time: {:.4f}'.format(epoch, train_loss.item(), t_end-t_start))

        t_start = time.time()
        # validation
        net.eval()
        with torch.no_grad():
            logits = net(select,split_list,features_list)
            logp = F.sigmoid(logits)
            val_loss = loss(logp[val_idx], labels[val_idx])
        t_end = time.time()
        # print validation info
        print('Epoch {:05d} | Val_Loss {:.4f} | Time(s) {:.4f}'.format(
            epoch, val_loss.item(), t_end - t_start))
        # early stopping
        early_stopping(val_loss, net)
        if early_stopping.early_stop:
            print('Early stopping!')
            break

    # testing with evaluate_results_nc
    net.load_state_dict(torch.load('checkpoint/checkpoint_{}_{}_{}.pt'.format(args.dataset, args.num_layers,seednow)))
    net.eval()
    test_logits = []
    with torch.no_grad():
        logits = net(select,split_list,features_list)
        test_logits = logits[test_idx]
        pred = (test_logits.cpu().numpy()>0).astype(int)
        filepath = "../../submit_simpleHGNresultall/" + 'beta0.5'
        if os.path.exists(filepath):
            dl.gen_file_for_evaluate(test_idx=test_idx, label=pred, file_path= filepath+'/'+f"{args.dataset}_{cur_repeat + 1}.txt", mode='multi')
        else:
            os.mkdir(filepath)
            dl.gen_file_for_evaluate(test_idx=test_idx, label=pred,
                                     file_path=filepath + '/' + f"{args.dataset}_{cur_repeat + 1}.txt", mode='multi')
        print(dl.evaluate(pred))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='MRGNN testing for the DBLP dataset')
    ap.add_argument('--feats-type', type=int, default=3,
                    help='Type of the node features used. ' +
                         '0 - loaded features; ' +
                         '1 - only target node features (zero vec for others); ' +
                         '2 - only target node features (id vec for others); ' +
                         '3 - all id vec. Default is 2;' +
                        '4 - only term features (id vec for others);' + 
                        '5 - only term features (zero vec for others).')
    ap.add_argument('--hidden-dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--num-heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
    ap.add_argument('--epoch', type=int, default=300, help='Number of epochs.')
    ap.add_argument('--patience', type=int, default=30, help='Patience.')
    ap.add_argument('--repeat', type=int, default=5, help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument('--num-layers', type=int, default=5)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--weight-decay', type=float, default=2e-4)
    ap.add_argument('--slope', type=float, default=0.1)
    ap.add_argument('--dataset', type=str)
    ap.add_argument('--edge-feats', type=int, default=64)
    ap.add_argument('--run', type=int, default=1)
    ap.add_argument('--intralayers', type=int, default=2)
    ap.add_argument('--seedlist', type=str, default='4321,2000,2023,321,2020')

    args = ap.parse_args()
    seedlist = args.seedlist
    # seedlist = seedlist.split(',')
    SEED_LIST = []
    if seedlist == 'res':
        for seed in range(100, 160):
            for i in range(5):
                SEED_LIST.append(seed)
            for cur_repeat in range(args.repeat):
                set_random_seed(SEED_LIST[cur_repeat], is_cuda=True)
                run_model_DBLP(args, SEED_LIST[cur_repeat])
    else:

        seedlist = seedlist.split(',')
        for i in seedlist:
            SEED_LIST.append(int(i))
        print(SEED_LIST)
        for cur_repeat in range(args.repeat):
            set_random_seed(SEED_LIST[cur_repeat], is_cuda=True)
            run_model_DBLP(args,SEED_LIST[cur_repeat])
    # run_model_DBLP(args)
