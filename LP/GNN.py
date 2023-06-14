import torch
import torch.nn as nn
import dgl
from dgl.nn.pytorch import GraphConv

import dgl.function as fn
from dgl.nn.pytorch import edge_softmax, GATConv
from conv import myGATConv,preGATConv

class DistMult(nn.Module):
    def __init__(self, num_rel, dim):
        super(DistMult, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(size=(num_rel, dim, dim)))
        nn.init.xavier_normal_(self.W, gain=1.414)

    def forward(self, left_emb, right_emb, r_id):
        thW = self.W[r_id]
        left_emb = torch.unsqueeze(left_emb, 1)
        right_emb = torch.unsqueeze(right_emb, 2)
        return torch.bmm(torch.bmm(left_emb, thW), right_emb).squeeze()

class Dot(nn.Module):
    def __init__(self):
        super(Dot, self).__init__()
    def forward(self, left_emb, right_emb, r_id):
        left_emb = torch.unsqueeze(left_emb, 1)
        right_emb = torch.unsqueeze(right_emb, 2)
        return torch.bmm(left_emb, right_emb).squeeze()

class myGAT(nn.Module):
    def __init__(self,
                 g,
                 hgs,
                 num_etypes,
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
                 alpha,
                 decode='distmult'):
        super(myGAT, self).__init__()
        self.g = g
        self.hgs = hgs
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.intraconvs = nn.ModuleList()
        self.activation = activation
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        # for i in range(len(self.hgs)):
        #     self.convs.append(preGATConv(num_hidden, num_hidden, 1, feat_drop, attn_drop, negative_slope, residual=True))
        # for i in range(len(self.hgs)):
        #     self.convs2.append(preGATConv(num_hidden, num_hidden,1, feat_drop,attn_drop, negative_slope, residual=True))
        for i in range(intalayer):
            temp = nn.ModuleList()
            for i in range(len(self.hgs)):
                temp.append(
                    preGATConv(num_hidden, num_hidden, 1, feat_drop, attn_drop, negative_slope, residual=True))
            self.intraconvs.append(temp)
        # input projection (no residual)
        self.gat_layers.append(myGATConv(
            num_hidden, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation, alpha=alpha))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(myGATConv(num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation, alpha=alpha))
        # output projection
        self.gat_layers.append(myGATConv(num_hidden* heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None, alpha=alpha))
        self.epsilon = torch.FloatTensor([1e-12]).cuda()
        if decode == 'distmult':
            self.decoder = DistMult(num_etypes, num_classes*(num_layers+2))
        elif decode == 'dot':
            self.decoder = Dot()

    def l2_norm(self, x):
        # This is an equivalent replacement for tf.l2_normalize, see https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/math/l2_normalize for more information.
        return x / (torch.max(torch.norm(x, dim=1, keepdim=True), self.epsilon))

    def forward(self, select,split_list,features_list, left, right, mid):
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
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
            ph = torch.split(h, split_list[:s + 1], 0)
            preh[s] = ph[s]
            h = preh
        h = torch.cat(h, 0)
        h = h.squeeze()
        emb = [self.l2_norm(h)]
        res_attn = None
        for l in range(self.num_layers):
            h, res_attn = self.gat_layers[l](self.g, h, res_attn=res_attn)
            emb.append(self.l2_norm(h.mean(1)))
            h = h.flatten(1)
        # output projection
        logits, _ = self.gat_layers[-1](self.g, h, res_attn=res_attn)#None)
        logits = logits.mean(1)
        logits = self.l2_norm(logits)
        emb.append(logits)
        logits = torch.cat(emb, 1)
        left_emb = logits[left]
        right_emb = logits[right]
        return self.decoder(left_emb, right_emb, mid)

