"""Render clearly labelled synthetic targets for layout testing only."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tools.vis_config import save_figure, setup_journal_style

DISCLAIMER = 'SYNTHETIC / NOT FOR SUBMISSION'
PROTOCOLS = ['N60-XSub', 'N60-XView', 'N120-XSub', 'N120-XSet', 'NW-UCLA']
MAIN_BASE = [92.8, 97.1, 89.5, 91.1, 96.7]
MAIN_OURS = [93.7, 97.8, 90.6, 92.1, 97.5]
TPFM = np.array([[91.19, 87.44], [91.47, 87.68],
                 [91.56, 87.77], [91.82, 87.96]])
SPG = np.array([[91.82, 87.96], [91.96, 88.08],
                [92.19, 88.29], [92.35, 88.43]])
K_VALUES = [2, 4, 6, 8]
K_N60 = [92.08, 92.35, 92.21, 91.97]
K_N120 = [88.19, 88.43, 88.31, 88.10]
ROBUST_LABELS = ['Clean', 'Drop25', 'Drop50', 'Mask10', 'Mask30']
ROBUST_BASE = [91.19, 90.22, 86.38, 87.82, 70.15]
ROBUST_OURS = [92.35, 91.61, 88.24, 89.63, 76.42]


def watermark(fig):
    fig.text(0.5, 0.5, DISCLAIMER, ha='center', va='center',
             fontsize=32, color='#b22222', alpha=0.13, rotation=28,
             weight='bold', zorder=1000)
    fig.text(0.5, 0.985, DISCLAIMER, ha='center', va='top',
             fontsize=10, color='#b22222', weight='bold')


def plot_grouped(ax, labels, baseline, ours, title, ylabel='Top-1 (%)'):
    x = np.arange(len(labels)); width = 0.36
    ax.bar(x - width / 2, baseline, width, label='Baseline', color='#4c78a8')
    ax.bar(x + width / 2, ours, width, label='TPFS-Net target', color='#e45756')
    ax.set_xticks(x, labels, rotation=18, ha='right')
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(axis='y', alpha=0.25)

# __APPEND__
