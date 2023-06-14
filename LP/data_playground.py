from sklearn.metrics import f1_score
import dgl
import torch
import random
from torch.backends import cudnn
import numpy as np
import os
from scipy.sparse import csr_matrix
from hgutils import load_data, EarlyStopping
import torch.nn.functional as F
# link_type_dic = {0: 'ap', 1: 'pc', 2: 'pt', 3: 'pa', 4: 'cp', 5: 'tp'}

def prune(adj):
    # maxinadj = adj.max()
    # length = adj.shape[0]
    # removeself = csr_matrix(([maxinadj]*length, (list(range(length)), list(range(length)))))
    # adj = adj - removeself
    # print(adj)

    res_adj = 0
    num = 600 #每个元路径取最大先验权重的num个点
    for i in range(num):
        # print(i)
        col = adj.argmax(1).getA1()
        row = list(range(len(col)))
        data = adj.toarray()[(row, col)]
        # print(len(data.shape),len(row),len(col))
        newadj = csr_matrix((data, (row, col)),shape=(len(col), len(col)))
        # print(newadj)
        newadj.eliminate_zeros()
        # print(newadj.shape)
        adj = adj - newadj
        res_adj = res_adj +newadj
    # print(res_adj)
    # exit(0)
    return res_adj

def multi_metapath_graph_edgevalue_premeta(hg,metapaths):
    edge_list = []
    value_list = []
    num_dic = {}
    sub = 0
    for type in hg.ntypes:
        num = hg.num_nodes(type)
        num_dic[type] = num - sub
        sub = num
    res_adj = 0
    for metapath in metapaths:
        adj = 1
        for etype in metapath:
            adj = adj * hg.adj(etype=etype, scipy_fmt='csr')
        # adj = prune(adj)
        # print("conducting",metapath)
        # print(adj)
        res_adj = res_adj + adj
    res_adj = (res_adj).tocsr()
    value_list.append(torch.tensor(res_adj.toarray(),device=hg.device,dtype=torch.float32)[res_adj.nonzero()].flatten())
    edge_list.append(res_adj.nonzero())
    value_res = torch.cat(value_list,0)
    # print(value_res)
    edge_list0 = []
    edge_list1 = []
    for item in edge_list:
        edge_list0.append(torch.tensor(item[0],device=hg.device))
    for item in edge_list:
        edge_list1.append(torch.tensor(item[1],device=hg.device))
    edge0 = torch.cat(edge_list0,0)
    edge1 = torch.cat(edge_list1,0)
    edge_res = (edge0,edge1)
    # print(edge_res)
    new_g = dgl.graph(edge_res,device=hg.device)
    # new_g = dgl.graph(edge_list[0],device=hg.device)
    new_g.edata['w'] = value_res
    # print(new_g)
    # for i in range(1,len(edge_list)):
    #     # print(edge_list[i].nonzero())
    #     new_g = dgl.add_edges(new_g,edge_list[i][0],edge_list[i][1])
    #     print(new_g)
    # print(new_g)
    # exit(0)
    return new_g,num_dic


def multi_metapath_graph_edgevalue(hg,metapaths):
    edge_list = []
    value_list = []
    num_dic = {}
    sub = 0
    for type in hg.ntypes:
        num = hg.num_nodes(type)
        num_dic[type] = num - sub
        sub = num
    for per_typr_metapaths in metapaths:
        res_adj = 0
        for metapath in per_typr_metapaths:
            adj = 1
            for etype in metapath:
                adj = adj * hg.adj(etype=etype, scipy_fmt='csr')
            adj = prune(adj)
            # print(adj)
            res_adj = res_adj + adj
        res_adj = (res_adj).tocsr()
        value_list.append(torch.tensor(res_adj.toarray(),device=hg.device,dtype=torch.float32)[res_adj.nonzero()].flatten())
        edge_list.append(res_adj.nonzero())
    value_res = torch.cat(value_list,0)
    # print(value_res)
    edge_list0 = []
    edge_list1 = []
    for item in edge_list:
        edge_list0.append(torch.tensor(item[0],device=hg.device))
    for item in edge_list:
        edge_list1.append(torch.tensor(item[1],device=hg.device))
    edge0 = torch.cat(edge_list0,0)
    edge1 = torch.cat(edge_list1,0)
    edge_res = (edge0,edge1)
    # print(edge_res)
    new_g = dgl.graph(edge_res,device=hg.device)
    # new_g = dgl.graph(edge_list[0],device=hg.device)
    new_g.edata['w'] = value_res
    # print(new_g)
    # for i in range(1,len(edge_list)):
    #     # print(edge_list[i].nonzero())
    #     new_g = dgl.add_edges(new_g,edge_list[i][0],edge_list[i][1])
    #     print(new_g)
    # print(new_g)
    # exit(0)
    return new_g,num_dic

    # res_adj = 0
    # for metapath in metapaths:
    #     adj = 1
    #     for etype in metapath:
    #         adj = adj * hg.adj(etype=etype, scipy_fmt='csr')
    #     res_adj = res_adj + adj
    # res_adj = (res_adj != 0).tocsr()
    # # 传入的metapaths必须是一个类型点的 首尾要一致
    # srctype = hg.to_canonical_etype(metapaths[0][0])[0]
    # dsttype = hg.to_canonical_etype(metapaths[0][-1])[2]
    # # dgl的特色 蛮有意思的 g这个东西只有结构 信息 没有特征信息
    # new_g = dgl.heterograph({(srctype, '_E', dsttype): res_adj.nonzero()},
    #                         {srctype: res_adj.shape[0], dsttype: res_adj.shape[1]},
    #                         idtype=hg.idtype, device=hg.device)
    # new_g.nodes[srctype].data.update(hg.nodes[srctype].data)
    # return new_g

def metapath_graph_egdevalue(g, metapath):
    adj0 = 1  # apa
    for etype in metapath:
        adj0 = adj0 * g.adj(etype=etype, scipy_fmt='csr')  # 这个地方文档有问题 默认
    srctype = g.to_canonical_etype(metapath[0])[0]
    dsttype = g.to_canonical_etype(metapath[-1])[2]
    # dgl的特色 蛮有意思的 g这个东西只有结构 信息 没有特征信息
    new_g = dgl.heterograph({(srctype, '_E', dsttype): adj0.nonzero()},
                            {srctype: adj0.shape[0], dsttype: adj0.shape[1]},
                            idtype=g.idtype, device=g.device)
    # 挺有意思 dgl图的特征值必须是float
    new_g.edata['w'] = torch.tensor(adj0.toarray(),device=g.device,dtype=torch.float32)[adj0.nonzero()].flatten()
    ## ok了只是不知道有没有简单方法
    # # copy srcnode features
    new_g.nodes[srctype].data.update(g.nodes[srctype].data)
    return new_g

def multi_metapath_graph(hg,metapaths):
    edge_list = []
    num_dic = {}
    sub = 0
    for type in hg.ntypes:
        num = hg.num_nodes(type)
        num_dic[type] = num - sub
        sub = num
    for per_typr_metapaths in metapaths:
        res_adj = 0
        for metapath in per_typr_metapaths:
            adj = 1
            for etype in metapath:
                adj = adj * hg.adj(etype=etype, scipy_fmt='csr')
            # adj = prune(adj)
            res_adj = res_adj + adj
        res_adj = (res_adj != 0).tocsr()
        edge_list.append(res_adj.nonzero())
        # srctype = hg.to_canonical_etype(per_typr_metapaths[0][0])[0]
        # dsttype = hg.to_canonical_etype(per_typr_metapaths[0][-1])[2]
        # edge_dic[(srctype, srctype+'_'+dsttype, dsttype)] = res_adj.nonzero()
        # edge_list.append(res_adj.nonzero())
        # num_dic[srctype] = res_adj.shape[0]
        # num_dic[dsttype] = res_adj.shape[1]
        # pre_src = res_adj.shape[0]
        # pre_dst = res_adj.shape[1]
    # new_g = dgl.heterograph(edge_dic,num_dic,idtype=hg.idtype, device=hg.device)
    edge_list0 = []
    edge_list1 = []
    for item in edge_list:
        edge_list0.append(torch.tensor(item[0],device=hg.device))
    for item in edge_list:
        edge_list1.append(torch.tensor(item[1],device=hg.device))
    edge0 = torch.cat(edge_list0,0)
    edge1 = torch.cat(edge_list1,0)
    edge_res = (edge0,edge1)
    # print(edge_res)
    new_g = dgl.graph(edge_res,device=hg.device)
    # new_g = dgl.graph(edge_list[0],device=hg.device)
    # print(new_g)
    # for i in range(1,len(edge_list)):
    #     # print(edge_list[i].nonzero())
    #     new_g = dgl.add_edges(new_g,edge_list[i][0],edge_list[i][1])
    #     print(new_g)
    # print(new_g)
    # exit(0)
    return new_g,num_dic

    # res_adj = 0
    # for metapath in metapaths:
    #     adj = 1
    #     for etype in metapath:
    #         adj = adj * hg.adj(etype=etype, scipy_fmt='csr')
    #     res_adj = res_adj + adj
    # res_adj = (res_adj != 0).tocsr()
    # # 传入的metapaths必须是一个类型点的 首尾要一致
    # srctype = hg.to_canonical_etype(metapaths[0][0])[0]
    # dsttype = hg.to_canonical_etype(metapaths[0][-1])[2]
    # # dgl的特色 蛮有意思的 g这个东西只有结构 信息 没有特征信息
    # new_g = dgl.heterograph({(srctype, '_E', dsttype): res_adj.nonzero()},
    #                         {srctype: res_adj.shape[0], dsttype: res_adj.shape[1]},
    #                         idtype=hg.idtype, device=hg.device)
    # new_g.nodes[srctype].data.update(hg.nodes[srctype].data)
    # return new_g

def multi_metapath_graph_2order(hg,metapaths):
    edge_list = []
    num_dic = {}
    sub = 0
    for type in hg.ntypes:
        num = hg.num_nodes(type)
        num_dic[type] = num - sub
        sub = num
    for per_typr_metapaths in metapaths:
        res_adj = 0
        for metapath in per_typr_metapaths:
            adj = 1
            for etype in metapath:
                adj = adj * hg.adj(etype=etype, scipy_fmt='csr')
                print(adj)
                print("=======================================")
            adj = (adj != 0).tocsr()
            print(adj*adj)
            exit(0)
            res_adj = res_adj + adj
        res_adj = (res_adj != 0).tocsr()
        edge_list.append(res_adj.nonzero())
        # srctype = hg.to_canonical_etype(per_typr_metapaths[0][0])[0]
        # dsttype = hg.to_canonical_etype(per_typr_metapaths[0][-1])[2]
        # edge_dic[(srctype, srctype+'_'+dsttype, dsttype)] = res_adj.nonzero()
        # edge_list.append(res_adj.nonzero())
        # num_dic[srctype] = res_adj.shape[0]
        # num_dic[dsttype] = res_adj.shape[1]
        # pre_src = res_adj.shape[0]
        # pre_dst = res_adj.shape[1]
    # new_g = dgl.heterograph(edge_dic,num_dic,idtype=hg.idtype, device=hg.device)
    edge_list0 = []
    edge_list1 = []
    for item in edge_list:
        edge_list0.append(torch.tensor(item[0],device=hg.device))
    for item in edge_list:
        edge_list1.append(torch.tensor(item[1],device=hg.device))
    edge0 = torch.cat(edge_list0,0)
    edge1 = torch.cat(edge_list1,0)
    edge_res = (edge0,edge1)
    new_g = dgl.graph(edge_res,device=hg.device)
    # new_g = dgl.graph(edge_list[0],device=hg.device)
    # for i in range(1,len(edge_list)):
    #     # print(edge_list[i].nonzero())
    #     new_g = dgl.add_edges(new_g,edge_list[i][0],edge_list[i][1])
    #     print(new_g)
    # print(new_g)
    # exit(0)
    return new_g,num_dic

    # res_adj = 0
    # for metapath in metapaths:
    #     adj = 1
    #     for etype in metapath:
    #         adj = adj * hg.adj(etype=etype, scipy_fmt='csr')
    #     res_adj = res_adj + adj
    # res_adj = (res_adj != 0).tocsr()
    # # 传入的metapaths必须是一个类型点的 首尾要一致
    # srctype = hg.to_canonical_etype(metapaths[0][0])[0]
    # dsttype = hg.to_canonical_etype(metapaths[0][-1])[2]
    # # dgl的特色 蛮有意思的 g这个东西只有结构 信息 没有特征信息
    # new_g = dgl.heterograph({(srctype, '_E', dsttype): res_adj.nonzero()},
    #                         {srctype: res_adj.shape[0], dsttype: res_adj.shape[1]},
    #                         idtype=hg.idtype, device=hg.device)
    # new_g.nodes[srctype].data.update(hg.nodes[srctype].data)
    # return new_g

def multitype_metapath_graph(hg,metapaths):
    hgs = []
    for metapath in metapaths:
        hgs.append(multi_metapath_graph(hg,metapath))
    return dgl.merge(hgs)
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# dataset = DglNodePropPredDataset(name = "ogbn-mag")
#
# split_idx = dataset.get_idx_split()
# train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
# graph, label = dataset[0]
# print(graph)
# PubMed的这个数据集（KDD）和CKD的不一样 这个点好像是乱序的
# hg= load_data('Freebase', feat_type=0)
# print(hg)
# # #
#
# metapathes = [ #为什么papap不收敛 两层
#                     [['ap', 'pt', 'tp', 'pa'], ['ap', 'ps', 'sp', 'pa']],
#                   [['tp', '-pp', 'pt'],['tp', 'pp', 'pt']]]  #
# # my_meta_paths = [[['ap', 'pa'], ['ap', 'pt', 'tp', 'pa']],[['pa', 'ap'], ['pt',  'tp']]] # ACM
# # my_meta_paths = [[['ap','pa'],['ap', 'pt', 'tp', 'pa']],[['pa', 'ap'], ['pt',  'tp']]] # DBLP
#
# newg,dic = multi_metapath_graph(hg,metapathes)
# # 数不对 因为p的id是从4057开始的
# print(newg)

# metapath0 = meta_paths[0]
# metapath1 = meta_paths[1]
# adj0 = 1 # apa
# adj1 = 1 #aptpa
# for etype in metapath0:
#     adj0 = adj0 * hg.adj(etype=etype, scipy_fmt='csr') #这个地方文档有问题 默认
# for etype in metapath1:
#     adj1 = adj1 * hg.adj(etype=etype, scipy_fmt='csr') #这个地方文档有问题 默认为true
# # adj0 = (adj0 != 0).tocsr() #没有管value
# # adj1 = (adj1 != 0).tocsr()
# # print(adj1.toarray()[0][0])
# # print('0')
# # print(adj0.toarray()[0][0])
# # print('1')
# # print(adj1.toarray()[0][0])
# # print('和')
# # print((adj0+adj1).toarray()[0][0])
# # print(adj1)
# # print(adj1.shape)
# # 这个api是将 'ap'变成'0 ap 1'的
# srctype = hg.to_canonical_etype(metapath0[0])[0]
# dsttype = hg.to_canonical_etype(metapath0[-1])[2]
# # dgl的特色 蛮有意思的 g这个东西只有结构 信息 没有特征信息
# new_g = dgl.heterograph({(srctype, '_E', dsttype): adj0.nonzero()},
#                             {srctype: adj0.shape[0], dsttype: adj0.shape[1]},
#                             idtype=hg.idtype, device=hg.device)
# new_g.edata['w'] = torch.tensor(adj0.toarray())[adj0.nonzero()].flatten()
# ## ok了只是不知道有没有简单方法
# # # copy srcnode features
# new_g.nodes[srctype].data.update(g.nodes[srctype].data)
# # # copy dstnode features
# # if srctype != dsttype:
# #     new_g.nodes[dsttype].data.update(g.nodes[dsttype].data)
# #
# # print(new_g)
#
# # 为什么返回的是pa呢？？
# # adj = hg.adj(etype='ap', scipy_fmt='csr',transpose=True)
# #
# # print(adj)