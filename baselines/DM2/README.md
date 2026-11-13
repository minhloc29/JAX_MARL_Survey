# DM2 baselines

The original `dm2/` directory is a PyTorch/PyMARL implementation. The
registered `dm2_*` targets use the equivalent JAX-native JaxMARL learners and
therefore share the regular CLI, checkpoint, GPU, and W&B metric contracts:

```bash
python -m baselines.cli --algorithm dm2_ippo --suite smax --env 3s5z \
  --total-timesteps 1000000 --wandb-mode online
python -m baselines.cli --algorithm dm2_iql --suite smax --env 3s5z \
  --total-timesteps 1000000 --wandb-mode online
python -m baselines.cli --algorithm dm2_vdn --suite smax --env 3s5z \
  --total-timesteps 1000000 --wandb-mode online
python -m baselines.cli --algorithm dm2_qmix --suite smax --env 3s5z \
  --total-timesteps 1000000 --wandb-mode online
```

Each target retains algorithm-specific metrics and emits the shared
`core/*` metrics through `baselines.metrics.log_metrics`.

The PyMARL DM2 distribution-matching extensions (GAIL/SIL), COMA, and QTRAN
are not aliases: they require expert-data loading and counterfactual
centralised updates that are not present in the current JAX baseline set.
They remain in `dm2/src` as the reference implementation and are intentionally
not exposed as runnable JAX targets until those update rules are ported.
