"""
SPG: Stage Prototype Guidance.

An action is decomposed into K temporal stages by soft attention. For each
stage we keep a learnable prototype bank P[c, k] indexed by class. During
training the prototypes act as auxiliary classifiers and are aligned to the
stage features via a contrastive InfoNCE term, which produces stronger
intra-class compactness without an external clustering pass.

At inference time only the stage features are used; the prototype bank is
just an extra logits head that gets averaged with the standard FC head.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class StageDecomposer(nn.Module):
    """
    Soft temporal segmentation into K stages.

    Input  : x (N, C, T, V)
    Output : stage_feats (N, K, C)
             stage_attn  (N, K, T)
    """

    def __init__(self, in_channels, num_stages=4, hidden=64,
                 order_strength=8.0):
        super().__init__()
        self.num_stages = num_stages
        self.attn = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden, num_stages, kernel_size=1),
        )
        # frozen so the four stage centers can't collapse during training
        self.register_buffer('order_bias',
                             torch.linspace(-1.0, 1.0, num_stages))
        self.order_strength = order_strength

    def forward(self, x):
        N, C, T, V = x.shape
        feat_t = x.mean(dim=-1)
        logits = self.attn(feat_t)

        pos = torch.linspace(-1.0, 1.0, T, device=x.device).view(1, 1, T)
        order = -((pos - self.order_bias.view(1, -1, 1)) ** 2) * self.order_strength
        logits = logits + order

        attn = F.softmax(logits, dim=-1)
        stage_feats = torch.einsum('nct,nkt->nkc', feat_t, attn)
        return stage_feats, attn


class StagePrototypeHead(nn.Module):
    """
    Class-conditional prototype bank P (num_class, num_stages, dim).

    Forward returns per-stage class logits (averaged over stages) AND the
    stage features so that a contrastive prototype loss can be applied
    outside this module.
    """

    def __init__(self, num_class, num_stages, dim, temperature=0.1):
        super().__init__()
        self.num_class = num_class
        self.num_stages = num_stages
        self.dim = dim
        self.temperature = temperature

        proto = torch.randn(num_class, num_stages, dim) * 0.02
        self.prototypes = nn.Parameter(proto)
        self.stage_weight = nn.Parameter(torch.ones(num_stages) / num_stages)

    def normalised_proto(self):
        return F.normalize(self.prototypes, dim=-1)

    def forward(self, stage_feats):
        f = F.normalize(stage_feats, dim=-1)
        p = self.normalised_proto()

        sim = torch.einsum('nkd,ckd->nkc', f, p) / self.temperature
        w = F.softmax(self.stage_weight, dim=0).view(1, -1, 1)
        logits = (sim * w).sum(dim=1)
        return logits, f, p


def stage_prototype_loss(stage_feats_norm, prototypes_norm, labels,
                         temperature=0.1):
    """
    InfoNCE style alignment: each stage feature should be closest to its
    own class' prototype at the same stage, and farther from other classes'.

    stage_feats_norm : (N, K, D)  L2-normalised
    prototypes_norm  : (C, K, D)  L2-normalised
    labels           : (N,)
    """
    N, K, D = stage_feats_norm.shape
    C = prototypes_norm.size(0)

    sim = torch.einsum('nkd,ckd->nkc', stage_feats_norm, prototypes_norm)
    sim = sim / temperature

    target = labels.view(N, 1).expand(N, K).reshape(N * K)
    sim = sim.reshape(N * K, C)
    return F.cross_entropy(sim, target)


class StageDecomposerV3(nn.Module):
    """
    Hard temporal partition with K-1 learnable boundaries.

    Each stage k corresponds to a soft window
        alpha_k(t) = sigmoid(s*(t - b_{k-1})) - sigmoid(s*(t - b_k))
    where 0 = b_0 < b_1 < ... < b_{K-1} < b_K = 1.

    The boundaries b_1..b_{K-1} are reparameterised through softmax-cumsum so
    they are guaranteed monotonic without manual constraints. A bigger
    sharpness s pushes the window towards a hard one-hot assignment.

    A small content-aware modulator (1D-Conv) lets the partition slightly
    shift per sample, so different action lengths can be respected without
    breaking the monotonic order.

    Input  : x (N, C, T, V)
    Output : stage_feats (N, K, C)
             stage_attn  (N, K, T)
    """

    def __init__(self, in_channels, num_stages=4, sharpness=40.0,
                 content_modulation=True, hidden=64, min_width=0.15):
        super().__init__()
        self.K = num_stages
        # K cumsum-softmax weights → K-1 internal boundaries in (0, 1)
        # min_width: floor so each segment stays ≥ min_width regardless of
        # how peaked the softmax becomes. Prevents v3-pre1.0 collapse to
        # [0.49, 0.01, 0.01, 0.49].
        self.raw_segments = nn.Parameter(torch.zeros(num_stages))
        self.sharpness = sharpness
        self.min_width = min_width
        self.content_modulation = content_modulation
        if content_modulation:
            # tiny per-sample shifter, output K-1 small offsets
            self.modulator = nn.Sequential(
                nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(hidden, num_stages - 1, kernel_size=1),
                nn.Tanh(),
            )

    def forward(self, x):
        N, C, T, V = x.shape
        feat_t = x.mean(dim=-1)  # (N, C, T)

        # structural floor: each segment is at least min_width;
        # the remaining (1 - K*min_width) is distributed via softmax
        free = 1.0 - self.K * self.min_width
        seg = self.min_width + free * F.softmax(self.raw_segments, dim=0)
        boundaries = torch.cumsum(seg, dim=0)[:-1]           # (K-1,) in (0,1)

        if self.content_modulation:
            offset = self.modulator(feat_t).squeeze(-1) * 0.05  # (N, K-1)
            b = boundaries.view(1, -1) + offset                # (N, K-1)
            b = b.clamp(0.05, 0.95)
        else:
            b = boundaries.view(1, -1).expand(N, -1)

        # build (N, K+1) endpoints. Use ±10 sentinels (with sharpness s=20,
        # sigmoid saturates well within ~0.3) so stage 0 and stage K-1 keep
        # full weight at t=0 and t=1 respectively.
        zeros = torch.full((N, 1), -10.0, device=x.device)
        ones = torch.full((N, 1), 10.0, device=x.device)
        b_full = torch.cat([zeros, b, ones], dim=1)            # (N, K+1)

        # normalised time positions (1, 1, T)
        t = torch.linspace(0.0, 1.0, T, device=x.device).view(1, 1, T)
        s = self.sharpness

        b_left = b_full[:, :-1].unsqueeze(-1)                  # (N, K, 1)
        b_right = b_full[:, 1:].unsqueeze(-1)                  # (N, K, 1)
        attn = (torch.sigmoid(s * (t - b_left))
                - torch.sigmoid(s * (t - b_right)))            # (N, K, T)
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-6)

        stage_feats = torch.einsum('nct,nkt->nkc', feat_t, attn)
        return stage_feats, attn


def stage_diversity_loss(stage_feats_norm):
    """
    Encourage K stage features within one sample to be MUTUALLY ORTHOGONAL.
    stage_feats_norm: (N, K, D), L2-normalised.
    Returns mean off-diagonal cosine similarity (lower = more orthogonal).
    """
    N, K, D = stage_feats_norm.shape
    sim = torch.einsum('nkd,nld->nkl', stage_feats_norm, stage_feats_norm)
    eye = torch.eye(K, device=sim.device, dtype=torch.bool).unsqueeze(0)
    off = sim.masked_select(~eye).abs()
    return off.mean()
