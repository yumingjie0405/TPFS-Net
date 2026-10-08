"""
TPFM: Topology-Preserving Functional Modeling.

Two pieces:
  * HeatKernelEncoder  -- spectral / functional positional features
                          based on the heat kernel of the physical skeleton.
  * topology_consistency_loss  -- Laplacian-smoothness regularizer
                                  pushing the adaptive hyper-graph to
                                  preserve the skeleton's low-frequency
                                  topology.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_skeleton_laplacian(num_node, edges):
    A = np.zeros((num_node, num_node), dtype=np.float32)
    for (i, j) in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    d = A.sum(axis=1)
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L = np.eye(num_node, dtype=np.float32) - D_inv_sqrt @ A @ D_inv_sqrt
    return L


def _heat_kernel_features(num_node, edges, taus):
    L = _build_skeleton_laplacian(num_node, edges)
    eigvals, eigvecs = np.linalg.eigh(L)
    feats = []
    for t in taus:
        Kt = (eigvecs * np.exp(-float(t) * eigvals)) @ eigvecs.T
        feats.append(Kt.astype(np.float32))
    return np.stack(feats, axis=0), L.astype(np.float32)


class HeatKernelEncoder(nn.Module):
    """
    Functional positional encoder. For each joint v we produce a vector
    summarising how heat diffuses from v over multiple timescales on
    the physical skeleton. Encoder is small and topology-aware.
    """

    def __init__(self, num_node, edges, out_channels, taus=(0.2, 1.0, 5.0)):
        super().__init__()
        self.num_node = num_node
        self.taus = tuple(taus)

        feats, L = _heat_kernel_features(num_node, edges, self.taus)
        self.register_buffer('hk_feats', torch.from_numpy(feats))
        self.register_buffer('L_prior', torch.from_numpy(L))

        in_dim = len(self.taus) * num_node
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels),
        )
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self):
        f = self.hk_feats.permute(1, 0, 2).reshape(self.num_node, -1)
        emb = self.proj(f)
        return torch.tanh(self.scale) * emb


def topology_consistency_loss(adjacency_list, L_prior, num_real_nodes,
                              num_modes=4):
    """
    Encourage adaptive adjacencies G to keep low-frequency components of the
    skeleton Laplacian. Concretely: project the leading eigen-modes of L_prior
    through G and penalise deviation from their projection through I.

    Each item in ``adjacency_list`` may be:
        - (V, V)
        - (K, V, V)              (multi-subset)
        - (N, V, V) / (N, K, V, V)
    Items with V > num_real_nodes are cropped to the real-node block.
    """
    if not adjacency_list:
        return torch.zeros((), device=L_prior.device)

    eigvals, eigvecs = torch.linalg.eigh(L_prior)
    basis = eigvecs[:, :num_modes].to(L_prior.device)
    L = L_prior.to(L_prior.device)

    loss = 0.0
    cnt = 0
    for G in adjacency_list:
        if G.dim() == 2:
            G = G.unsqueeze(0)
        if G.dim() == 4:
            G = G.mean(dim=1)
        if G.size(-1) > num_real_nodes:
            G = G[..., :num_real_nodes, :num_real_nodes]
        d = G.sum(dim=-1, keepdim=True).abs().clamp(min=1e-8)
        Gn = G / d

        b = basis.unsqueeze(0).expand(Gn.size(0), -1, -1)
        proj = Gn @ b
        smooth = torch.einsum('nvk,vu,nuk->nk', proj, L, proj)
        loss = loss + smooth.mean()
        cnt += 1
    return loss / max(cnt, 1)


def topology_consistency_loss_v3(adjacency_list, L_prior, A_phys,
                                 num_real_nodes, num_modes=4):
    """v3: spectral low-frequency preservation + direct Frobenius anchor.

    Combines:
      (a) the original spectral term:  basis^T (G_norm)^T L (G_norm) basis
      (b) Frobenius anchor: ||G_norm - A_phys_norm||_F^2 averaged over layers
    """
    if not adjacency_list:
        return torch.zeros((), device=L_prior.device)

    eigvals, eigvecs = torch.linalg.eigh(L_prior)
    basis = eigvecs[:, :num_modes].to(L_prior.device)
    L = L_prior.to(L_prior.device)

    A_phys = A_phys.to(L_prior.device)
    if A_phys.size(-1) > num_real_nodes:
        A_phys = A_phys[..., :num_real_nodes, :num_real_nodes]
    a_phys_norm = A_phys / (A_phys.sum(dim=-1, keepdim=True).abs().clamp(1e-8))

    spec, frob, cnt = 0.0, 0.0, 0
    for G in adjacency_list:
        if G.dim() == 2:
            G = G.unsqueeze(0)
        if G.dim() == 4:
            G = G.mean(dim=1)
        if G.size(-1) > num_real_nodes:
            G = G[..., :num_real_nodes, :num_real_nodes]
        d = G.sum(dim=-1, keepdim=True).abs().clamp(min=1e-8)
        Gn = G / d

        b = basis.unsqueeze(0).expand(Gn.size(0), -1, -1)
        proj = Gn @ b
        smooth = torch.einsum('nvk,vu,nuk->nk', proj, L, proj)
        spec = spec + smooth.mean()

        frob = frob + ((Gn - a_phys_norm.unsqueeze(0)) ** 2).mean()
        cnt += 1
    spec = spec / max(cnt, 1)
    frob = frob / max(cnt, 1)
    return spec + 0.5 * frob
