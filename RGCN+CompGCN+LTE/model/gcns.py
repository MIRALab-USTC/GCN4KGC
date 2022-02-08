import torch
from torch import nn
import dgl
from model.rgcn_layer import RelGraphConv
from model.compgcn_layer import CompGCNCov
import torch.nn.functional as F


class GCNs(nn.Module):
    def __init__(self, num_ent, num_rel, num_base, init_dim, gcn_dim, embed_dim, n_layer, edge_type, edge_norm,
                 conv_bias=True, gcn_drop=0., opn='mult', wni=False, wsi=False, encoder='compgcn', use_bn=True, ltr=True):
        super(GCNs, self).__init__()
        self.act = torch.tanh
        self.loss = nn.BCELoss()
        self.num_ent, self.num_rel, self.num_base = num_ent, num_rel, num_base
        self.init_dim, self.gcn_dim, self.embed_dim = init_dim, gcn_dim, embed_dim
        self.conv_bias = conv_bias
        self.gcn_drop = gcn_drop
        self.opn = opn
        self.edge_type = edge_type  # [E]
        self.edge_norm = edge_norm  # [E]
        self.n_layer = n_layer

        self.wni = wni

        self.encoder = encoder

        self.init_embed = self.get_param([self.num_ent, self.init_dim])
        self.init_rel = self.get_param([self.num_rel * 2, self.init_dim])

        if encoder == 'compgcn':
            if n_layer < 3:
                self.conv1 = CompGCNCov(self.init_dim, self.gcn_dim, self.act, conv_bias, gcn_drop, opn, num_base=-1,
                                        num_rel=self.num_rel, wni=wni, wsi=wsi, use_bn=use_bn, ltr=ltr)
                self.conv2 = CompGCNCov(self.gcn_dim, self.embed_dim, self.act, conv_bias, gcn_drop,
                                        opn, wni=wni, wsi=wsi, use_bn=use_bn, ltr=ltr) if n_layer == 2 else None
            else:
                self.conv1 = CompGCNCov(self.init_dim, self.gcn_dim, self.act, conv_bias, gcn_drop, opn, num_base=-1,
                                        num_rel=self.num_rel, wni=wni, wsi=wsi, use_bn=use_bn, ltr=ltr)
                self.conv2 = CompGCNCov(self.gcn_dim, self.gcn_dim, self.act, conv_bias, gcn_drop, opn, num_base=-1,
                                        num_rel=self.num_rel, wni=wni, wsi=wsi, use_bn=use_bn, ltr=ltr)
                self.conv3 = CompGCNCov(self.gcn_dim, self.embed_dim, self.act, conv_bias, gcn_drop,
                                        opn, wni=wni, wsi=wsi, use_bn=use_bn, ltr=ltr)
        elif encoder == 'rgcn':
            self.conv1 = RelGraphConv(self.init_dim, self.gcn_dim, self.num_rel*2, "bdd",
                                      num_bases=self.num_base, activation=self.act, self_loop=(not wsi), dropout=gcn_drop, wni=wni)
            self.conv2 = RelGraphConv(self.gcn_dim, self.embed_dim, self.num_rel*2, "bdd", num_bases=self.num_base,
                                      activation=self.act, self_loop=(not wsi), dropout=gcn_drop, wni=wni) if n_layer == 2 else None

        self.bias = nn.Parameter(torch.zeros(self.num_ent))

    def get_param(self, shape):
        param = nn.Parameter(torch.Tensor(*shape))
        nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('relu'))
        return param

    def calc_loss(self, pred, label):
        return self.loss(pred, label)

    def forward_base(self, g, subj, rel, drop1, drop2):
        """
        :param g: graph
        :param sub: subjects in a batch [batch]
        :param rel: relations in a batch [batch]
        :param drop1: dropout rate in first layer
        :param drop2: dropout rate in second layer
        :return: sub_emb: [batch, D]
                 rel_emb: [num_rel*2, D]
                 x: [num_ent, D]
        """
        x, r = self.init_embed, self.init_rel  # embedding of relations

        if self.n_layer > 0:
            if self.encoder == 'compgcn':
                if self.n_layer < 3:
                    x, r = self.conv1(g, x, r, self.edge_type, self.edge_norm)
                    x = drop1(x)  # embeddings of entities [num_ent, dim]
                    x, r = self.conv2(
                        g, x, r, self.edge_type, self.edge_norm) if self.n_layer == 2 else (x, r)
                    x = drop2(x) if self.n_layer == 2 else x
                else:
                    x, r = self.conv1(g, x, r, self.edge_type, self.edge_norm)
                    x = drop1(x)  # embeddings of entities [num_ent, dim]
                    x, r = self.conv2(g, x, r, self.edge_type, self.edge_norm)
                    x = drop1(x)  # embeddings of entities [num_ent, dim]
                    x, r = self.conv3(g, x, r, self.edge_type, self.edge_norm)
                    x = drop2(x)
            elif self.encoder == 'rgcn':
                x = self.conv1(g, x, self.edge_type,
                               self.edge_norm.unsqueeze(-1))
                x = drop1(x)  # embeddings of entities [num_ent, dim]
                x = self.conv2(
                    g, x, self.edge_type, self.edge_norm.unsqueeze(-1)) if self.n_layer == 2 else x
                x = drop2(x) if self.n_layer == 2 else x

        # filter out embeddings of subjects in this batch
        sub_emb = torch.index_select(x, 0, subj)
        # filter out embeddings of relations in this batch
        rel_emb = torch.index_select(r, 0, rel)

        return sub_emb, rel_emb, x


class GCN_TransE(GCNs):
    def __init__(self, num_ent, num_rel, num_base, init_dim, gcn_dim, embed_dim, n_layer, edge_type, edge_norm,
                 bias=True, gcn_drop=0., opn='mult', hid_drop=0., gamma=9., wni=False, wsi=False, encoder='compgcn', use_bn=True, ltr=True):
        super(GCN_TransE, self).__init__(num_ent, num_rel, num_base, init_dim, gcn_dim, embed_dim, n_layer,
                                         edge_type, edge_norm, bias, gcn_drop, opn, wni, wsi, encoder, use_bn, ltr)
        self.drop = nn.Dropout(hid_drop)
        self.gamma = gamma

    def forward(self, g, subj, rel):
        """
        :param g: dgl graph
        :param sub: subject in batch [batch_size]
        :param rel: relation in batch [batch_size]
        :return: score: [batch_size, ent_num], the prob in link-prediction
        """
        sub_emb, rel_emb, all_ent = self.forward_base(
            g, subj, rel, self.drop, self.drop)
        obj_emb = sub_emb + rel_emb

        x = self.gamma - torch.norm(obj_emb.unsqueeze(1) - all_ent, p=1, dim=2)

        score = torch.sigmoid(x)

        return score


class GCN_DistMult(GCNs):
    def __init__(self, num_ent, num_rel, num_base, init_dim, gcn_dim, embed_dim, n_layer, edge_type, edge_norm,
                 bias=True, gcn_drop=0., opn='mult', hid_drop=0., wni=False, wsi=False, encoder='compgcn', use_bn=True, ltr=True):
        super(GCN_DistMult, self).__init__(num_ent, num_rel, num_base, init_dim, gcn_dim, embed_dim, n_layer,
                                           edge_type, edge_norm, bias, gcn_drop, opn, wni, wsi, encoder, use_bn, ltr)
        self.drop = nn.Dropout(hid_drop)

    def forward(self, g, subj, rel):
        """
        :param g: dgl graph
        :param sub: subject in batch [batch_size]
        :param rel: relation in batch [batch_size]
        :return: score: [batch_size, ent_num], the prob in link-prediction
        """
        sub_emb, rel_emb, all_ent = self.forward_base(
            g, subj, rel, self.drop, self.drop)
        obj_emb = sub_emb * rel_emb  # [batch_size, emb_dim]
        x = torch.mm(obj_emb, all_ent.transpose(1, 0))  # [batch_size, ent_num]
        x += self.bias.expand_as(x)
        score = torch.sigmoid(x)
        return score


class GCN_ConvE(GCNs):
    def __init__(self, num_ent, num_rel, num_base, init_dim, gcn_dim, embed_dim, n_layer, edge_type, edge_norm,
                 bias=True, gcn_drop=0., opn='mult', hid_drop=0., input_drop=0., conve_hid_drop=0., feat_drop=0.,
                 num_filt=None, ker_sz=None, k_h=None, k_w=None, wni=False, wsi=False, encoder='compgcn', use_bn=True, ltr=True):
        """
        :param num_ent: number of entities
        :param num_rel: number of different relations
        :param num_base: number of bases to use
        :param init_dim: initial dimension
        :param gcn_dim: dimension after first layer
        :param embed_dim: dimension after second layer
        :param n_layer: number of layer
        :param edge_type: relation type of each edge, [E]
        :param bias: weather to add bias
        :param gcn_drop: dropout rate in compgcncov
        :param opn: combination operator
        :param hid_drop: gcn output (embedding of each entity) dropout
        :param input_drop: dropout in conve input
        :param conve_hid_drop: dropout in conve hidden layer
        :param feat_drop: feature dropout in conve
        :param num_filt: number of filters in conv2d
        :param ker_sz: kernel size in conv2d
        :param k_h: height of 2D reshape
        :param k_w: width of 2D reshape
        """
        super(GCN_ConvE, self).__init__(num_ent, num_rel, num_base, init_dim, gcn_dim, embed_dim, n_layer,
                                        edge_type, edge_norm, bias, gcn_drop, opn, wni, wsi, encoder, use_bn, ltr)
        self.hid_drop, self.input_drop, self.conve_hid_drop, self.feat_drop = hid_drop, input_drop, conve_hid_drop, feat_drop
        self.num_filt = num_filt
        self.ker_sz, self.k_w, self.k_h = ker_sz, k_w, k_h

        # one channel, do bn on initial embedding
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(
            self.num_filt)  # do bn on output of conv
        self.bn2 = torch.nn.BatchNorm1d(self.embed_dim)

        self.drop = torch.nn.Dropout(self.hid_drop)  # gcn output dropout
        self.input_drop = torch.nn.Dropout(
            self.input_drop)  # stacked input dropout
        self.feature_drop = torch.nn.Dropout(
            self.feat_drop)  # feature map dropout
        self.hidden_drop = torch.nn.Dropout(
            self.conve_hid_drop)  # hidden layer dropout

        self.conv2d = torch.nn.Conv2d(in_channels=1, out_channels=self.num_filt,
                                      kernel_size=(self.ker_sz, self.ker_sz), stride=1, padding=0, bias=bias)

        flat_sz_h = int(2 * self.k_h) - self.ker_sz + 1  # height after conv
        flat_sz_w = self.k_w - self.ker_sz + 1  # width after conv
        self.flat_sz = flat_sz_h * flat_sz_w * self.num_filt
        # fully connected projection
        self.fc = torch.nn.Linear(self.flat_sz, self.embed_dim)

    def concat(self, ent_embed, rel_embed):
        """
        :param ent_embed: [batch_size, embed_dim]
        :param rel_embed: [batch_size, embed_dim]
        :return: stack_input: [B, C, H, W]
        """
        ent_embed = ent_embed.view(-1, 1, self.embed_dim)
        rel_embed = rel_embed.view(-1, 1, self.embed_dim)
        # [batch_size, 2, embed_dim]
        stack_input = torch.cat([ent_embed, rel_embed], 1)

        assert self.embed_dim == self.k_h * self.k_w
        # reshape to 2D [batch, 1, 2*k_h, k_w]
        stack_input = stack_input.reshape(-1, 1, 2 * self.k_h, self.k_w)
        return stack_input

    def forward(self, g, subj, rel):
        """
        :param g: dgl graph
        :param sub: subject in batch [batch_size]
        :param rel: relation in batch [batch_size]
        :return: score: [batch_size, ent_num], the prob in link-prediction
        """
        sub_emb, rel_emb, all_ent = self.forward_base(
            g, subj, rel, self.drop, self.input_drop)
        # [batch_size, 1, 2*k_h, k_w]
        stack_input = self.concat(sub_emb, rel_emb)
        x = self.bn0(stack_input)
        x = self.conv2d(x)  # [batch_size, num_filt, flat_sz_h, flat_sz_w]
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz)  # [batch_size, flat_sz]
        x = self.fc(x)  # [batch_size, embed_dim]
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, all_ent.transpose(1, 0))  # [batch_size, ent_num]
        x += self.bias.expand_as(x)
        score = torch.sigmoid(x)
        return score
