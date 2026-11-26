# Differential-parity harness

Scripts used to verify that the array-API port of `loss.py` / `protein.py` /
`sequence_optimization.py` computes the same numbers as the pre-port code. They compare
*two checkouts of this repo* rather than two code paths, so they keep working as the port
advances.

Nothing here is imported by the package; it is all run by hand.

## Setting up the two trees

```bash
git worktree add /somewhere/ref <pre-port-commit>   # b6fb3ae was the baseline
export BC_REF=/somewhere/ref
export BC_NEW=/workspace/BindCraft2
```

Every script honours `BC_TREE`: it prepends that path to `sys.path` **and strips the
editable install's meta-path finder**, which otherwise wins and silently imports the
working tree no matter what `sys.path` says. `launch.py` asserts which tree it got.

## The scripts

| script | what it does |
|---|---|
| `parity.py` | Calls ~60 loss/protein helpers on seeded inputs, writes a JSON dump (dtype, shape, sum, absmax, head). Run once per tree. |
| `compare.py` | Diffs two dumps with a float32 tolerance (rtol 2e-5, atol 1e-6) and reports exact / within-tol / outside counts. Exit code 1 if anything is outside. |
| `inputs.py` | Dumps all 38 AlphaFold input features, to localise a difference to featurisation vs the model. |
| `save_pred.py` / `eval_pred.py` | Saves one real AF2 prediction, then scores every campaign loss against that fixed prediction — isolates loss maths from model nondeterminism. |
| `launch.py` | Runs the `bindcraft` CLI out of `BC_TREE`. |
| `launch_raw.py` | Same, with `trajectory_output.recorded_row` neutered so losses.csv keeps full precision instead of the 1–2 decimal rounding. |
| `launch_avals.py` | Wraps the jitted gradient function and logs each call's argument avals (incl. JAX device commitment, shown as `c0`/`c1`). This is what found the recompile bug. |
| `timegrad.py` | Steady-state timing of `sequence_gradients`, warm cache, to separate execution cost from compile cost. |
| `bench_eager.py` | Eager microbenchmark of the shim/namespace indirection alone. |
| `torch_device.py` | Calls the three loss helpers that build their own arrays with torch CUDA tensors and checks what comes back is on the card. Exit code 1 if anything is on the host. The only script here that can catch a missing `device=`. |
| `paired_campaigns.sh` | Runs one campaign of N trajectories out of each tree, with every stage gate at 0.0 so the whole step schedule always runs. `short` is a 10-round schedule, `long` a 103-round one. |
| `paired_stats.py` | Per metric: each tree's spread across trajectories, the paired difference, `\|Δ\|/sd`, and warm per-trajectory design seconds. |
| `tiny_smoke.json` | 9-gradient-step, single-trajectory, trajectory-only campaign on hPDL1. ~10 s on a 4090. Set `project_folder` before use. |

## Paired trajectories are cheap; paired campaigns are not

`jax.random.fold_in(campaign_key, trajectory_number)` is what each trajectory draws from,
so trajectory *n* of a campaign in one tree and trajectory *n* in the other are the same
draw. One campaign of 24 trajectories therefore buys 24 paired samples for one compile —
about 8 s a trajectory on a 10-round schedule against hPDL1, against ~60 s if each were
its own `campaign_seed` in its own process. `paired_campaigns.sh` does this; use it rather
than looping over seeds.

Set the stage gates (`min_plddt_screen` and friends) to 0.0 for a comparison. Otherwise a
trajectory is cut off wherever its pLDDT happens to fall, the two trees' trajectories run
for different numbers of rounds, and the metric being compared is partly "where did this
one stop".

## Two things that will waste an hour if forgotten

1. **XLA autotuning makes bit-comparison impossible.** Rewriting a loss expression changes
   the HLO of the one 85k-line `jit_sequence_design_loss` graph, the autotuner then picks
   different kernels throughout, and outputs differ by ~1e-3 — which amplifies to ~0.1 pLDDT
   over a few steps. For any numerical comparison, run both trees with:

   ```bash
   XLA_FLAGS=--xla_gpu_autotune_level=0
   ```

   Without it, a tree differs from *itself* between runs. `JAX_DEFAULT_MATMUL_PRECISION=highest`
   alone does **not** fix it.

2. **Give the two trees separate compile caches.** A shared `JAX_COMPILATION_CACHE_DIR`
   makes one tree serve the other's compiled graphs, which produced a bogus
   "gradients are bit-identical" result once.

## Known harness defect

`parity.py`'s `elastic_network_cov` case reports its `sum` metric, which is catastrophic
cancellation on a near-singular matrix inverse — it differs across runs while absmax and
head are identical. It should report a norm instead. It is the only false alarm in the set.
