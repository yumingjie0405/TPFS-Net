# TPFS-Net Utilities

The scripts in this directory support experiment management, robustness evaluation, complexity profiling, and generation of analysis figures described in the TPFS-Net paper.

Most evaluation and visualization scripts require a trained checkpoint, the corresponding experiment configuration, and the dataset. Those artifacts are not included in this repository. Pass paths for your own run; generated output is written under `vis/` or the output path supplied to each script, and should not be committed.

Run a script's help to see its supported arguments, for example:

```bash
python -m tools.eval_robustness --help
python -m tools.profile_complexity --help
python -m tools.vis_topology_drift --help
```

For model training and evaluation, see the repository [README](../README.md).
要做严格 ours-vs-baseline 并排对比,需补齐**同协议**实验,详见 [EXPERIMENTS_TODO.md](../EXPERIMENTS_TODO.md)。
