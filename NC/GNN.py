import torch
import torch.nn as nn
import dgl
from dgl.nn.pytorch import GraphConv

import dgl.function as fn
from dgl.nn.pytorch import edge_softmax, GATConv
from conv import myGATConv,preGATConv,FAGCN,preGATConvHereo,preGATConvPrior,preGATConvMixHop,preGATConvMixHopCutNode
from dgl.nn.pytorch import edge_softmax
# class preGAT(nn.Module):
#     def __init__(self):

class myGAT(nn.Module):
    def __init__(self,
                 g,
                 hgs,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 intalayer,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 alpha):
        super(myGAT, self).__init__()
        self.g = g
        self.hgs = hgs
        # self.hg2 = hg2
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.intraconvs = nn.ModuleList()
        # self.convs = nn.ModuleList()
        # self.convs2 = nn.ModuleList()
        self.activation = activation
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        # h2gcn异配性
        # self.h2gcnfc = nn.Linear(num_hidden*3, num_hidden)
        # nn.init.xavier_normal_(self.h2gcnfc.weight, gain=1.414)
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        # 每个元路径图一个gat
        # for i in range(len(self.hgs)):
        #     self.convs.append(preGATConv(num_hidden, num_hidden, heads[-1], 0.5, 0.5, negative_slope, residual=True))
        # for i in range(len(self.hgs)):
        #     self.convs2.append(preGATConv(num_hidden, num_hidden, heads[-1], 0.5, 0.5, negative_slope, residual=True))
        for i in range(intalayer):
            temp = nn.ModuleList()
            for i in range(len(self.hgs)):
                temp.append(
                    preGATConv(num_hidden, num_hidden, heads[-1], feat_drop, attn_drop, negative_slope, residual=True))
            self.intraconvs.append(temp)
        # input projection (no residual)
        # 双层异配性
        # self.preconv = preGATConvHereo(num_hidden,num_hidden,heads[0],feat_drop,attn_drop,negative_slope,residual=True)
        # self.preconv_out = preGATConvHereo(num_hidden* heads[0],num_hidden,heads[-1],feat_drop,attn_drop,negative_slope,residual=True)
        # 单层异配性
        # self.preconv_one = preGATConvHereo(num_hidden,num_hidden,heads[-1],feat_drop,attn_drop,negative_slope,residual=True)
        # 一层GAT  原ACM
        # self.preconvacm = preGATConv(num_hidden,num_hidden,heads[-1],feat_drop,attn_drop,negative_slope,residual=True)

        # 重新实现的普通GAT
        # self.preconvacm = preGATConv(num_hidden, num_hidden, heads[0], feat_drop, attn_drop, negative_slope,
        #                           residual=True)
        # self.preconvacm1 = preGATConv(num_hidden * heads[0], num_hidden, heads[-1], feat_drop, attn_drop, negative_slope,
        #                            residual=True)

        # 普通GAT并且舍弃一些点
        # self.preconvacm = preGATConvMixHopCutNode(num_hidden, num_hidden, heads[0], feat_drop, attn_drop, negative_slope,
        #                           allow_zero_in_degree=True,residual=True)
        # self.preconvacm1 = preGATConvMixHopCutNode(num_hidden * heads[0], num_hidden, heads[-1], feat_drop, attn_drop, negative_slope,
        #                            allow_zero_in_degree=True,residual=True)

        # 利用先验卷积一层
        # self.preconvprior = preGATConvPrior(num_hidden,num_hidden,heads[-1],feat_drop,attn_drop,negative_slope,allow_zero_in_degree=True)
        # 两层先验卷积
        # self.preconvprior1 = preGATConvPrior(num_hidden, num_hidden, heads[0], feat_drop, attn_drop, negative_slope,
        #                           allow_zero_in_degree=True,residual=True)
        # self.preconvprior2 = preGATConvPrior(num_hidden * heads[0], num_hidden, heads[-1], feat_drop, attn_drop, negative_slope,
        #                            allow_zero_in_degree=True,residual=True)
        # 直接吧FAGCN生拉硬套过来
        # self.preGNN = FAGCN(self.hg,num_hidden,num_hidden,feat_drop,eps=0.3)
        # H2GCN的异配性
        # self.h2gcnconv1 = preGATConvMixHop(num_hidden,num_hidden,heads[-1],feat_drop,attn_drop,negative_slope)
        # self.h2gcnconv2 = preGATConvMixHop(num_hidden,num_hidden,heads[-1],feat_drop,attn_drop,negative_slope)
        # H2GCN删掉部分低节点
        # self.h2gcnconv1 = preGATConvMixHopCutNode(num_hidden, num_hidden, heads[-1], feat_drop, attn_drop, negative_slope)
        # self.h2gcnconv2 = preGATConvMixHopCutNode(num_hidden, num_hidden, heads[-1], feat_drop, attn_drop, negative_slope)

        self.gat_layers.append(myGATConv(
            num_hidden, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation, alpha=alpha))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(myGATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation, alpha=alpha))
        # output projection
        self.gat_layers.append(myGATConv( #* heads[-2]
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None, alpha=alpha))
        self.epsilon = torch.FloatTensor([1e-12]).cuda()

    def forward(self,select, split_list,features_list): #
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))

        # intra-type
        for i in range(len(self.hgs)):
            preh = h
            # all
            ph = []
            s = int(select[i])
            h = torch.cat(h[:s + 1], 0)  # +1是因为右边是开区间
            # print(self.hgs[i])
            # print(h.shape)
            # exit(0)
            for j in range(len(self.intraconvs)):
                h = self.intraconvs[j][i](self.hgs[i], h).flatten(1)
            # h = self.convs[i](self.hgs[i], h).flatten(1)
            # # for 2 layers
            # h = self.convs2[i](self.hgs[i], h).flatten(1)
            ph = torch.split(h, split_list[:s + 1], 0)
            preh[s] = ph[s]
            h = preh
        h = torch.cat(h, 0)
        h = h.squeeze()

        # h = torch.cat(h,0)
        # # pre层实验
        # preh = h
        # # all
        # ph = []
        # for s in select:
        #     ph.append(h[int(s)])
        # h = torch.cat(h[:int(select[-1])+1],0) # +1是因为右边是开区间
        # # H2GCN式的聚合
        # # h1 = self.h2gcnconv1(self.hg,h).squeeze()
        # # h2 = self.h2gcnconv2(self.hg2,h).squeeze()
        # # 异配性双层ACM
        # # h = self.preconv(self.hg, h).flatten(1)
        # # h = self.preconv_out(self.hg, h).squeeze()
        # # 异配性单层ACM
        # # h = self.preconv_one(self.hg, h).squeeze()
        #
        # #单层普通GAT
        # # h = self.preconvacm(self.hg,h).squeeze()
        #
        #
        # #重新实现的普通GAT
        # # h = self.preconvacm(self.hg, h).flatten(1)
        # # h = self.preconvacm1(self.hg, h).squeeze()
        #
        # # 先验一层
        # h = self.preconvacm(self.hg,h).squeeze()
        #
        # # 先验两层
        # # h = self.preconvprior1(self.hg, h).flatten(1)
        # # h = self.preconvprior2(self.hg, h).squeeze()
        #
        # # 硬套FAGCN
        # # h = self.preGNN(h)
        #
        # # 普通的聚合
        # ph = torch.split(h,split_list[:int(select[-1])+1],0)
        # for i in select:
        #     preh[int(i)] = ph[int(i)]
        # h = torch.cat(preh, 0)
        # # H2GCN式的聚合
        # # h_origin = []
        # # h_1 = []
        # # h_2 = []
        # # ph_1 = torch.split(h1, split_list[:int(select[-1]) + 1], 0)
        # # ph_2 = torch.split(h2, split_list[:int(select[-1]) + 1], 0)
        # # for i in select:
        # #     h_origin.append(preh[int(i)])
        # #     h_1.append(ph_1[int(i)])
        # #     h_2.append(ph_2[int(i)])
        # # h_origin = torch.cat(h_origin, 0)
        # # h_1 = torch.cat(h_1, 0)
        # # h_2 = torch.cat(h_2, 0)
        # # h_h2gcn = torch.cat((h_origin,h_1,h_2),1)
        # # h_h2gcn = self.h2gcnfc(h_h2gcn)
        # # # print(h_h2gcn.shape)
        # # h2gcnsplitlist = []
        # # for i in select:
        # #     h2gcnsplitlist.append(split_list[int(i)])
        # # # print(h2gcnsplitlist) #5959 1902
        # # h_h2gcn = torch.split(h_h2gcn, h2gcnsplitlist, 0)
        # # # print(h_h2gcn[0],h_h2gcn[1]) #5959 1902
        # # for index,i in enumerate(select):
        # #     # print(index,i) #0 1 // 1 3
        # #     preh[int(i)] = h_h2gcn[index]
        # # h = torch.cat(preh, 0)
        #
        # # print(h.shape)
        # # 问题是 传入pre的 h必须得有最大的那个
        # # DBLP
        # # h = torch.cat(h,0)
        # # h = self.preconv(self.hg,h).flatten(1)
        # # h = self.preconv_out(self.hg,h).squeeze()
        #
        # # ACM
        # # ph = torch.cat(h[0:2],0)
        # # ph = self.preconv(self.hg,ph).flatten(1)
        # # ph = self.preconv_out(self.hg,ph).squeeze()
        # # h = torch.cat(h, 0)
        # # h = torch.cat((ph,h[ph.shape[0]:]),0)
        # # print(h.shape)
        # # exit(0)

        res_attn = None
        for l in range(self.num_layers):
            h, res_attn = self.gat_layers[l](self.g, h, res_attn=res_attn)
            h = h.flatten(1)
        # output projection
        logits, _ = self.gat_layers[-1](self.g, h, res_attn=None)
        logits = logits.mean(1)
        # This is an equivalent replacement for tf.l2_normalize, see https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/math/l2_normalize for more information.
        logits = logits / (torch.max(torch.norm(logits, dim=1, keepdim=True), self.epsilon))
        return logits

class RGAT(nn.Module):
    def __init__(self,
                 gs,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()
        self.gs = gs
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList([nn.ModuleList() for i in range(len(gs))])
        self.activation = activation
        self.weights = nn.Parameter(torch.zeros((len(in_dims), num_layers+1, len(gs))))
        self.sm = nn.Softmax(2)
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        for i in range(len(gs)):
            # input projection (no residual)
            self.gat_layers[i].append(GATConv(
                num_hidden, num_hidden, heads[0],
                feat_drop, attn_drop, negative_slope, False, self.activation))
            # hidden layers
            for l in range(1, num_layers):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.gat_layers[i].append(GATConv(
                    num_hidden * heads[l-1], num_hidden, heads[l],
                    feat_drop, attn_drop, negative_slope, residual, self.activation))
            # output projection
            self.gat_layers[i].append(GATConv(
                num_hidden * heads[-2], num_classes, heads[-1],
                feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, features_list):
        nums = [feat.size(0) for feat in features_list]
        weights = self.sm(self.weights)
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        for l in range(self.num_layers):
            out = []
            for i in range(len(self.gs)):
                out.append(torch.split(self.gat_layers[i][l](self.gs[i], h).flatten(1), nums))
            h = []
            for k in range(len(nums)):
                tmp = []
                for i in range(len(self.gs)):
                    tmp.append(out[i][k]*weights[k,l,i])
                h.append(sum(tmp))
            h = torch.cat(h, 0)
        out = []
        for i in range(len(self.gs)):
            out.append(torch.split(self.gat_layers[i][-1](self.gs[i], h).mean(1), nums))
        logits = []
        for k in range(len(nums)):
            tmp = []
            for i in range(len(self.gs)):
                tmp.append(out[i][k]*weights[k,-1,i])
            logits.append(sum(tmp))
        logits = torch.cat(logits, 0)
        return logits

class GAT(nn.Module):
    def __init__(self,
                 g,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            num_hidden, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, features_list):
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](self.g, h).mean(1)
        return logits

class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        # input layer
        self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation, weight=False))
        # hidden layers
        for i in range(num_layers - 1):
            self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(num_hidden, num_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features_list):
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        for i, layer in enumerate(self.layers):
            h = self.dropout(h)
            h = layer(self.g, h)
        return h
