"""
TS-HyperGCN  =  Hyper-GCN backbone
              + TPFM (Topology-Preserving Functional Modeling)
              + SPG  (Stage Prototype Guidance)

Returns: (logits, aux) where aux = (h_list, adj_list, stage_feats_norm,
                                    proto_norm, proto_logits, L_prior).

The training loop in main.py only needs to look at logits and aux[0]
(divergence loss); the extra terms are picked up if main.py is patched
to use ``model.compute_aux_loss(aux, label)``.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.hypergcn_base import (
    HYPERGC, MultiScale_TemporalConv, unit_tcn, conv_init, bn_init,
    import_class,
)
from model.tpfm import HeatKernelEncoder, topology_consistency_loss
from model.spg import StageDecomposer, StagePrototypeHead, stage_prototype_loss


class TCN_GCN_unit(nn.Module):
    """Same as in hypergcn_base, but exposes the adaptive adjacency G."""

    def __init__(self, in_channels, out_channels, num_point, hyper_joints, A,
                 stride=1, residual=True, kernel_size=5, dilations=(1, 2),
                 hyper=True):
        super().__init__()
        self.gcn1 = HYPERGC(in_channels, out_channels, num_point,
                            hyper_joints, A, hyper=hyper)
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            dilations=list(dilations),
                                            residual=False)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels,
                                     kernel_size=1, stride=stride)

    def forward(self, x):
        y, h_x = self.gcn1(x)
        y = self.relu(self.tcn1(y) + self.residual(x))
        return y, h_x


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2,
                 graph=None, graph_args=dict(), in_channels=3,
                 hyper_joints=3, drop_out=0,
                 num_stages=4, proto_dim=128, proto_temp=0.1,
                 lambda_topo=0.1, lambda_proto=0.5, lambda_ce_stage=0.3,
                 hk_taus=(0.2, 1.0, 5.0), use_hk=True,
                 use_topology_loss=True, use_spg=True,
                 use_proto_loss=True, use_stage_ce_loss=True,
                 use_diversity_loss=True):
        super().__init__()
        if graph is None:
            raise ValueError("graph must be provided")
        Graph = import_class(graph)
        self.graph = Graph(hyper_joints, **graph_args)
        A = self.graph.A

        self.num_class = num_class
        self.num_point = num_point
        self.num_stages = num_stages
        self.embedding_channels = 128

        self.lambda_topo = lambda_topo
        self.lambda_proto = lambda_proto
        self.lambda_ce_stage = lambda_ce_stage
        self.proto_temp = proto_temp
        self.use_hk = use_hk
        self.use_topology_loss = use_topology_loss
        self.use_spg = use_spg
        self.use_proto_loss = use_proto_loss
        self.use_stage_ce_loss = use_stage_ce_loss
        # Accepted here so v1/v3 share one explicit ablation interface. V1 has
        # no diversity term; v3 consumes this flag in its loss override.
        self.use_diversity_loss = use_diversity_loss

        edges = list(self.graph.neighbor)
        self.hk_encoder = HeatKernelEncoder(num_point, edges,
                                            self.embedding_channels,
                                            taus=hk_taus)

        self.data_bn = nn.BatchNorm1d(num_person * self.embedding_channels
                                      * num_point)
        self.to_joint_embedding = nn.Linear(in_channels,
                                            self.embedding_channels)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_point, self.embedding_channels))
        self.tanh = nn.Tanh()

        ec = self.embedding_channels
        self.l1 = TCN_GCN_unit(ec, ec, num_point, hyper_joints, A)
        self.l2 = TCN_GCN_unit(ec, ec, num_point, hyper_joints, A)
        self.l3 = TCN_GCN_unit(ec, ec, num_point, hyper_joints, A)
        self.l4 = TCN_GCN_unit(ec, ec * 2, num_point, hyper_joints, A,
                               stride=2)
        self.l5 = TCN_GCN_unit(ec * 2, ec * 2, num_point, hyper_joints, A)
        self.l6 = TCN_GCN_unit(ec * 2, ec * 2, num_point, hyper_joints, A)
        self.l7 = TCN_GCN_unit(ec * 2, ec * 2, num_point, hyper_joints, A,
                               stride=2)
        self.l8 = TCN_GCN_unit(ec * 2, ec * 2, num_point, hyper_joints, A)
        self.l9 = TCN_GCN_unit(ec * 2, ec * 2, num_point, hyper_joints, A)

        feat_dim = ec * 2
        self.fc = nn.Linear(feat_dim, num_class)

        self.stage_decomposer = StageDecomposer(feat_dim,
                                                num_stages=num_stages)
        self.stage_proj = nn.Sequential(
            nn.Linear(feat_dim, proto_dim),
            nn.GELU(),
            nn.Linear(proto_dim, proto_dim),
        )
        self.stage_proto_head = StagePrototypeHead(num_class, num_stages,
                                                   proto_dim,
                                                   temperature=proto_temp)

        nn.init.normal_(self.fc.weight, 0, math.sqrt(2.0 / num_class))
        bn_init(self.data_bn, 1)

        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    @staticmethod
    def _collect_adj(unit, x):
        return unit.gcn1

    def _collect_layer_adjacency(self):
        """Stack edge_importance * PA from every spatial GCN unit."""
        mats = []
        for layer in [self.l1, self.l2, self.l3, self.l4, self.l5,
                      self.l6, self.l7, self.l8, self.l9]:
            gcn = layer.gcn1
            A = gcn.edge_importance * gcn.PA
            mats.append(A)
        return mats

    def forward(self, x):
        N, C, T, V_in, M = x.size()
        x = x.permute(0, 4, 2, 3, 1).contiguous()
        x = self.to_joint_embedding(x)
        x = x + self.pos_embedding[:, :self.num_point]
        if self.use_hk:
            x = x + self.hk_encoder().unsqueeze(0).unsqueeze(0)
        x = self.tanh(x)
        x = x.permute(0, 1, 3, 4, 2).contiguous()

        x = x.view(N, M * V_in * self.embedding_channels, T)
        x = self.data_bn(x)
        x = x.view(N, M, V_in, self.embedding_channels, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, self.embedding_channels, T, V_in)

        x, h1 = self.l1(x); x1 = x
        x, h2 = self.l2(x)
        x, h3 = self.l3(x + x1)
        x, h4 = self.l4(x); x4 = x
        x, h5 = self.l5(x)
        x, h6 = self.l6(x + x4)
        x, h7 = self.l7(x); x7 = x
        x, h8 = self.l8(x)
        x, h9 = self.l9(x + x7)

        c_new = x.size(1)
        x_per_person = x.view(N, M, c_new, x.size(2), x.size(3))
        x_seq = x_per_person.mean(dim=1)

        x_pooled = x.view(N, M, c_new, -1).mean(3).mean(1)
        x_pooled = self.drop_out(x_pooled)
        ce_logits = self.fc(x_pooled)

        if self.use_spg:
            stage_feats, _stage_attn = self.stage_decomposer(x_seq)
            stage_proj = self.stage_proj(stage_feats)
            proto_logits, stage_feats_norm, proto_norm = \
                self.stage_proto_head(stage_proj)
            logits = 0.5 * ce_logits + 0.5 * proto_logits
        else:
            stage_feats_norm = None
            proto_norm = None
            proto_logits = None
            logits = ce_logits

        h_list = [h1, h2, h3, h4, h5, h6, h7, h8, h9]
        adj_list = self._collect_layer_adjacency()
        aux = {
            'h_list': h_list,
            'adj_list': adj_list,
            'stage_feats_norm': stage_feats_norm,
            'proto_norm': proto_norm,
            'proto_logits': proto_logits,
            'ce_logits': ce_logits,
            'L_prior': self.hk_encoder.L_prior,
            'num_real_nodes': self.num_point,
        }
        return logits, aux

    def compute_aux_loss(self, aux, label):
        zero = aux['ce_logits'].new_zeros(())
        if self.use_topology_loss:
            topo_loss = topology_consistency_loss(
                aux['adj_list'], aux['L_prior'], aux['num_real_nodes'])
        else:
            topo_loss = zero

        if self.use_spg and self.use_proto_loss:
            proto_loss = stage_prototype_loss(
                aux['stage_feats_norm'], aux['proto_norm'], label,
                temperature=self.proto_temp)
        else:
            proto_loss = zero

        if self.use_spg and self.use_stage_ce_loss:
            ce_stage = F.cross_entropy(aux['proto_logits'], label,
                                       label_smoothing=0.1)
        else:
            ce_stage = zero

        total = (self.lambda_topo * topo_loss
                 + self.lambda_proto * proto_loss
                 + self.lambda_ce_stage * ce_stage)

        parts = {
            'aux_topo': topo_loss.detach(),
            'aux_proto': proto_loss.detach(),
            'aux_ce_stage': ce_stage.detach(),
            'aux_total': total.detach(),
        }
        return total, parts


if __name__ == '__main__':
    m = Model(graph='graph.ntu_rgb_d.Graph', hyper_joints=3,
              graph_args={'labeling_mode': 'virtual_ensemble'}).cuda()
    x = torch.randn(2, 3, 64, 25, 2).cuda()
    logits, aux = m(x)
    aux_loss, parts = m.compute_aux_loss(aux,
                                          torch.zeros(2, dtype=torch.long).cuda())
    print(logits.shape, float(aux_loss), {k: float(v) for k, v in parts.items()})
