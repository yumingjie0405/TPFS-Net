"""
TPFS-Former (TS-HyperGCN v3).

Targets ≥93.5 on NTU60-CSub 4-stream by genuinely solving v1's two pathologies:

  1. **Topology drift**: v1's `edge_importance * PA` exploded (std ~9000,
     |max| ~3.5e5). Fixed by reading edge_importance through tanh so the
     multiplier on PA stays in [0, 2]. Topology-consistency loss is upgraded
     to a dual-term form (spectral low-frequency preservation + Frobenius
     anchor to the physical adjacency).

  2. **Stage collapse**: v1's StageDecomposer concentrated all K stages on
     the same temporal peak. v2 (frozen Gaussian buffer) helped only on a
     few classes. v3 replaces soft attention with a hard temporal partition
     parameterised by K-1 monotonic learnable boundaries plus a tiny
     content-aware modulator. Stages cannot cross or overlap by
     construction.

  3. **Stage diversity**: explicit penalty on inter-stage feature cosine
     similarity inside one sample. Even before the prototype loss kicks in,
     the K stages are pushed apart.

Curriculum-friendly: `compute_aux_loss(aux, label, epoch=...)` ramps the
auxiliary losses linearly from `ramp_start` to `ramp_end`. main.py only
needs to pass `epoch=epoch` -- if it doesn't, full strength is used.

Implementation note: this file extends the v1 Model class so all 9 GCN
units, embedding pipeline, residual structure, and the rest of the
backbone are inherited unchanged. Only the four critical bits (bounded
edge_importance, dual-term topology loss, hard-partition stage
decomposer, stage diversity loss) are replaced.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.ts_hgcn import Model as ModelV1
from model.tpfm import topology_consistency_loss_v3
from model.spg import (StageDecomposerV3, StagePrototypeHead,
                       stage_prototype_loss, stage_diversity_loss)


def _bounded_ei(layer):
    """Return (1 + tanh(raw)) where raw is the underlying nn.Parameter,
    bypassing the forward-time swap that may have replaced
    layer.gcn1.edge_importance with a bounded tensor.
    """
    gcn = layer.gcn1
    # _parameters always holds the raw Parameter; instance __dict__ may
    # transiently shadow it during forward via object.__setattr__.
    raw = gcn._parameters.get('edge_importance', gcn.edge_importance)
    return 1.0 + torch.tanh(raw)


class Model(ModelV1):
    """TPFS-Former: extends v1 with bounded edge_importance, dual-term
    topology loss, hard-partition stage decomposer and stage diversity."""

    def __init__(self, num_class=60, num_point=25, num_person=2,
                 graph=None, graph_args=dict(), in_channels=3,
                 hyper_joints=3, drop_out=0,
                 num_stages=4, proto_dim=128, proto_temp=0.1,
                 lambda_topo=0.5, lambda_proto=0.5, lambda_ce_stage=0.3,
                 lambda_div=0.1,
                 hk_taus=(0.2, 1.0, 5.0),
                 stage_sharpness=20.0,
                 ramp_start=30, ramp_end=100,
                 use_hk=True, use_topology_loss=True, use_spg=True,
                 use_proto_loss=True, use_stage_ce_loss=True,
                 use_diversity_loss=True):
        super().__init__(num_class=num_class, num_point=num_point,
                         num_person=num_person, graph=graph,
                         graph_args=graph_args, in_channels=in_channels,
                         hyper_joints=hyper_joints, drop_out=drop_out,
                         num_stages=num_stages, proto_dim=proto_dim,
                         proto_temp=proto_temp,
                         lambda_topo=lambda_topo, lambda_proto=lambda_proto,
                         lambda_ce_stage=lambda_ce_stage,
                         hk_taus=hk_taus, use_hk=use_hk,
                         use_topology_loss=use_topology_loss,
                         use_spg=use_spg, use_proto_loss=use_proto_loss,
                         use_stage_ce_loss=use_stage_ce_loss,
                         use_diversity_loss=use_diversity_loss)

        # === replace the soft-attention stage decomposer with a hard one ===
        feat_dim = self.stage_decomposer.attn[0].in_channels \
            if hasattr(self.stage_decomposer, 'attn') else 256
        self.stage_decomposer = StageDecomposerV3(feat_dim,
                                                  num_stages=num_stages,
                                                  sharpness=stage_sharpness)

        self.lambda_div = lambda_div
        self.ramp_start = ramp_start
        self.ramp_end = ramp_end

        # === register A_phys buffer for the Frobenius topology anchor ===
        A_phys = self.graph.A
        if isinstance(A_phys, (list, tuple)):
            A_phys = sum(A_phys) / len(A_phys)
        elif hasattr(A_phys, 'shape') and len(A_phys.shape) == 3:
            A_phys = A_phys.mean(0)
        self.register_buffer('A_phys', torch.from_numpy(A_phys).float())

        # === re-initialise edge_importance to small values (it was ones in
        #     v1; tanh(0)=0 means starting closer to the physical adjacency)
        for layer in [self.l1, self.l2, self.l3, self.l4, self.l5,
                      self.l6, self.l7, self.l8, self.l9]:
            with torch.no_grad():
                layer.gcn1.edge_importance.zero_()

    def _collect_layer_adjacency(self):
        """Override v1: stack BOUNDED edge_importance * PA per layer."""
        mats = []
        for layer in [self.l1, self.l2, self.l3, self.l4, self.l5,
                      self.l6, self.l7, self.l8, self.l9]:
            mats.append(_bounded_ei(layer) * layer.gcn1.PA)
        return mats

    def forward(self, x):
        """Same shape contract as v1 but applies bounded edge_importance
        each forward by temporarily swapping the read-out used inside
        HYPERGC.forward. We do this via a context manager that swaps the
        nn.Parameter for a buffer carrying the bounded value."""
        # cache raw params, install bounded values
        cache = []
        for layer in [self.l1, self.l2, self.l3, self.l4, self.l5,
                      self.l6, self.l7, self.l8, self.l9]:
            gcn = layer.gcn1
            raw = gcn.edge_importance
            bounded = 1.0 + torch.tanh(raw)
            # NOTE: setattr replaces the attribute on the instance (still
            # registered as Parameter via __setattr__), but we want it
            # autograd-connected to `raw`. nn.Module._parameters keeps the
            # original; bypass with object.__setattr__ to set as plain tensor.
            object.__setattr__(gcn, 'edge_importance', bounded)
            cache.append((gcn, raw))

        try:
            out = super().forward(x)
        finally:
            for gcn, raw in cache:
                # restore the actual Parameter so autograd can update raw
                object.__setattr__(gcn, 'edge_importance', raw)

        return out

    def compute_aux_loss(self, aux, label, epoch=None):
        """Override v1: dual-term topology loss + diversity + curriculum."""
        if epoch is None:
            ramp = 1.0
        else:
            r = (epoch - self.ramp_start) / max(1, self.ramp_end - self.ramp_start)
            ramp = float(max(0.0, min(1.0, r)))

        zero = aux['ce_logits'].new_zeros(())
        if self.use_topology_loss:
            topo = topology_consistency_loss_v3(
                aux['adj_list'], aux['L_prior'], self.A_phys,
                aux['num_real_nodes'])
        else:
            topo = zero

        if self.use_spg and self.use_proto_loss:
            proto = stage_prototype_loss(
                aux['stage_feats_norm'], aux['proto_norm'], label,
                temperature=self.proto_temp)
        else:
            proto = zero

        if self.use_spg and self.use_stage_ce_loss:
            ce_stage = F.cross_entropy(aux['proto_logits'], label,
                                       label_smoothing=0.1)
        else:
            ce_stage = zero

        if self.use_spg and self.use_diversity_loss:
            div = stage_diversity_loss(aux['stage_feats_norm'])
        else:
            div = zero

        total = ramp * (self.lambda_topo * topo
                        + self.lambda_proto * proto
                        + self.lambda_ce_stage * ce_stage
                        + self.lambda_div * div)
        parts = {
            'aux_topo': topo.detach(),
            'aux_proto': proto.detach(),
            'aux_ce_stage': ce_stage.detach(),
            'aux_div': div.detach(),
            'aux_ramp': zero.new_tensor(ramp),
            'aux_total': total.detach(),
        }
        return total, parts
