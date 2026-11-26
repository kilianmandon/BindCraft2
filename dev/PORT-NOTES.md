# Array-API port: state, verification, and what is left

Context for `0f3eee1`, which moved `loss.py`, `protein.py` and the on-path half of
`sequence_optimization.py` from `jax.numpy` onto the Python array API
(`array_api_compat` + `array_api_extra`), so a PyTorch predictor can be dropped in
beside the JAX AlphaFold one.

## How the shims are arranged, and why

`array_api_compat.array_namespace()` is **asymmetric**, which is the single most
surprising thing about this port:

- given a torch tensor it returns `array_api_compat.torch`, a wrapper that fixes
  `axis`→`dim`, the `min`/`max` namedtuple returns, and `astype`-as-a-function;
- given a JAX array it returns **plain `jax.numpy`** — there is no compat wrapper
  (`_helpers.py` raises "JAX does not have an array-api-compat wrapper").

So any non-standard name works on torch only by accident of torch's flat namespace, and
fails on JAX. The named-function shims in `protein.py` (`softmax`, `log_softmax`, `relu`,
`sigmoid`, `elu`, `stop_gradient`, `block_diagonal`, `flat_nonzero`, `boxcar_smooth`)
exist because those have no array-API spelling at all. `array_api_extra` covers `at`,
`pad`, `one_hot` and `isclose` on both backends.

**Nothing in the on-path code may route through numpy or scipy.** The pre-port code called
`scipy.special` in five places; on eager JAX that silently returns a numpy `ndarray` and
**kills the gradient with no error at all**. Under jit it raises, under torch-with-grad it
raises, on torch CUDA it raises. That was the one real bug the port fixed.

## Verification as of this commit (JAX only; nothing has run on torch yet)

| check | result |
|---|---|
| 62-case differential parity, GPU | 58 exact, 4 within float32 tol, 0 outside |
| AlphaFold input features | 38/38 bit-identical |
| campaign losses on one saved real prediction | 8/8 bit-identical |
| 4-step trajectory, `--xla_gpu_autotune_level=0` | bit-identical every round |
| quality, 6 paired seeds | no significant difference (ΔpLDDT +0.042 ± 0.023, p≈0.13; Δi_pTM p≈0.6) |
| 9-step design wall time | 10.36 s vs 10.13 s pre-port = 1.02× |

The quality comparison is underpowered: detecting a 0.04 pLDDT shift needs ~16 paired
seeds, not 6. It establishes "no gross regression", not equivalence.

## Verification of the `device=` fix, and the distribution the metrics actually have

The `device=` round (`chain_group_rows`, `binder_domain_membership`,
`elastic_network_covariance`) was checked with `dev/parity/paired_campaigns.sh` plus
`paired_stats.py`, which run one campaign of N trajectories out of each tree with every
stage gate at 0.0, so the whole schedule always runs:

| check | result |
|---|---|
| 62-case differential parity, GPU | 62/62 exact |
| the three helpers under torch CUDA | all three return `cuda:0` tensors, and `elastic_network_covariance` backpropagates |
| the same three on the pre-fix tree | two raise "found at least two devices"; `chain_group_rows` silently returns a CPU index tensor |

| campaign (one per tree, `--core benchmark`, gates at 0.0) | paired trajectories | result |
|---|---|---|
| hPDL1, 10-round schedule | 24 | 24/24 bit-identical on all 10 metrics; warm design time 5.94 s vs 6.01 s |
| hPDL1, 103-round schedule | 4 | 4/4 bit-identical; 61.7 s vs 62.2 s |
| hIL7RA, 10-round schedule | 24 | 24/24 bit-identical (incl. `coldspot_repel`); 13.14 s vs 13.05 s |

Only trajectory 1 compiled in either tree of either campaign, so the new `device=` calls
add no recompile — they execute on tracers inside the graph, where commitment does not
exist (trap 2). The three design-time ratios are 0.989, 0.991 and 1.007: ±1%, no sign.

The distributions those campaigns measured, per trajectory, at the last recorded round:

| | hPDL1, 10 rounds | hPDL1, 103 rounds | hIL7RA, 10 rounds |
|---|---|---|---|
| pLDDT | 0.9396 ± 0.0062 | 0.9780 ± 0.0150 | 0.9368 ± 0.0061 |
| i_pTM | 0.198 ± 0.026 | 0.633 ± 0.225 | 0.141 ± 0.026 |
| pTM | 0.6571 ± 0.0087 | 0.797 ± 0.081 | 0.7108 ± 0.0089 |
| compactness | 0.92 ± 0.68 | 1.02 ± 0.39 | 1.96 ± 1.39 |

A 10-round schedule is not a design: i_pTM sits at 0.14–0.20 and its spread is narrow
because nothing has converged. The 103-round schedule is where the trajectories separate —
i_pTM 0.34 to 0.87 across four of them — so that is the schedule a quality claim has to be
made on, and 4 trajectories is nowhere near enough to make one.

Two things to take from the campaign rows. First, **a change that only adds `device=` is
bit-identical, not merely in-distribution** — with `--xla_gpu_autotune_level=0` the HLO is
unchanged, so every metric of every paired trajectory agrees to the last bit, on both
schedules and on both targets. That is a stronger result than the 6-seed comparison of
`0f3eee1`, and it is the result to expect from any edit that does not touch an expression.

Second, the per-metric spread above is what a *real* change has to be measured against,
and it is wide: on the short schedule i_pTM has sd ≈ 0.026 against a mean of 0.198, and
compactness sd ≈ 0.68 against a mean of 0.92. A 24-trajectory campaign resolves a mean
shift of about 0.01 in i_pTM; nothing smaller is visible, and single-trajectory
comparisons say nothing at all.

## Two traps that cost hours each

**1. XLA autotuning defeats bit-comparison.** `sequence_design_loss` is jitted as one
85,000-line graph holding featurisation, all of AlphaFold, every loss and the Kabsch
alignment. Rewriting any loss expression changes the HLO, the autotuner then benchmarks
and picks different kernels *throughout*, and outputs move by ~1e-3 — which amplifies to
~0.11 pLDDT over four steps, the same magnitude as changing the seed. It is deterministic
per tree and different between trees. Always compare with
`XLA_FLAGS=--xla_gpu_autotune_level=0`; `JAX_DEFAULT_MATMUL_PRECISION=highest` alone does
not help. Also give each tree its own `JAX_COMPILATION_CACHE_DIR`, or one tree serves the
other's compiled graphs.

**2. `device=` on an eagerly-executed JAX array creation is a latent recompile trigger.**
An array built with `device=` is *committed* to that device, commitment spreads through
arithmetic, and a jitted function handed one committed argument returns committed outputs.
`Protein.padded_to` passed `device=` to `xp.arange`; the committed gradients flowed back
into the next step's sequence, which then presented a different aval, and the whole
85k-line gradient graph recompiled once — a **3.31× slowdown** on a 9-step design for
identical work. Fixed by `protein.py:arange_beside`, which forwards `device=` only for
torch (where omitting it puts the tensor on the host). The ~34 `device=` sites inside
`loss.py` are safe because they execute on tracers inside the graph, where commitment does
not exist.

## Left to do

### Correctness, before any torch run

- **`loss.py:117` raises on an empty loss set** where the pre-port code returned
  `jnp.asarray(0.0)` (the `sum(..., start=)` form). The `required_states <= set(...)`
  filter can legitimately empty the list. Carries a TODO already; decide it deliberately,
  because it turns a quiet zero into a crash.
- ~~Four non-scalar array creations omit `device=`~~ — **done.** `chain_group_rows`,
  `binder_domain_membership` and `elastic_network_covariance` now read the device off an
  array they already hold. Two of the three *crashed* under torch CUDA
  ("found at least two devices, cuda:0 and cpu"); `chain_group_rows` did not, because torch
  accepts a CPU index tensor into a CUDA tensor and silently copies it every step — the
  worse failure of the two, since it is only slow. See the table above for the check.
  The scalar `xp.asarray(0.0)` sites (`:330`, `:441`, `:447`, `:778`) are fine — torch
  treats 0-dim CPU tensors as scalars in promotion and they cross devices happily.
- **`protein.py:156` drops `precision=jax.lax.Precision.HIGHEST`** from
  `alignment_matrix_product`. This leaves Kabsch superposition on TF32 (~10-bit mantissa)
  on an RTX 4090, which affects `target_rmsd`, `induced_fit_interface` and
  `fold_switching`. Invisible to CPU parity testing. In torch it is a global setting, so it
  probably belongs in predictor setup rather than in this function.

### Standard-conformance cleanups (work today, fragile tomorrow)

- ~55 `.sum()`, 9 `.any()`, 4 `.reshape()` and 10 `.T` remain as *methods* rather than
  `xp.` functions. None are in the array API standard; they survive on torch only by
  accident of its flat namespace — the same trap as `axis` vs `dim`. **Every spelling that
  actually occurs was run on both backends** (torch 2.15 CUDA and JAX, float and bool, 2-D
  and 4-D): `sum()`, `sum(0)`, `sum(-1)`, `sum(axis=0)`, `sum(-1, keepdims=True)`,
  `sum((0,1))`, `any()`, `any(0)`, `any(-1)`, `any(axis=-1)`, `any((1,3))`,
  `reshape(2,-1)`, `reshape((2,-1))`, `reshape(-1)` and `.T` all work on both, including
  `keepdims` (not torch's own `keepdim`) and varargs `reshape`. Two divergences to know:
  - `.T` on an array of rank other than 2 is **deprecated in torch and will become an
    error** (it reverses all axes and warns). Every current use is on a 2-D array, so
    nothing is broken today; a new `.T` on a rank-3 array would be.
  - `mask.sum()` on a boolean is `int64` on torch and `int32` on JAX. Harmless where it
    meets a float32, which is every current use.
- `.astype(...)` as a **method** does not exist on a torch tensor at all, unlike the four
  above. The only on-path occurrences are in the optimiser half of
  `sequence_optimization.py` (`:96`, `:100`, `:108`), which is still deliberately JAX — so
  the boundary conversion that half needs is not optional, it is what keeps those lines
  working. `loss.py` and `protein.py`'s array-API code use `xp.astype` throughout.
- `xp.diagonal` at `loss.py:920` is not in the standard; `xp.linalg.diagonal` (`:938`,
  `:1003`, `:1004`) is. Pick one.
- `parity.py`'s `elastic_network_cov` case reports a `sum` that is catastrophic
  cancellation on a near-singular matrix inverse, so it differs across runs while absmax
  and head agree. It should report a norm. It is the harness's only false alarm.

### Still entirely in JAX

- **`filters.py`** (38 refs) may legitimately stay: `af2.py` never imports it and it sees
  no grad or jit. The price is keeping JAX as an install dependency purely for filtering
  and scoring.
- **The optimiser half of `sequence_optimization.py`** (43 refs: optax,
  `GradientSequenceOptimizer` and subclasses, `combine_sequence_gradients`,
  `normalize_sequence_gradient`, `SemigreedySequenceSampler`) can stay in JAX with a
  conversion at the boundary — torch grads in, logits out. That is microseconds against
  ~2.6 s per iteration.
- **`prediction.py`** is the next file on the path: the `ProteinPredictor` /
  `DifferentiableProteinPredictor` protocols plus `residue_chain_ids`,
  `split_residue_arrays_by_chain` and `concatenate_chain_arrays`, all still `jnp` and
  annotated `jax.Array`.
- **`bindcraft/af2_pytorch.py`** is a WIP stub that **does not parse** —
  `sequence_gradients` has a signature and no body, and it references undefined
  `DesignLoss`/`Array`. Nothing imports it, so nothing breaks, but it is on `main`.

## Typing

`protein.py` carries a module-level `A = TypeVar('A')` deliberately untyped, so the package
imports on a machine with only one of the two frameworks installed. `class Protein[A]` and
`class StructurePrediction[A]` use PEP 695, while `ProteinStates` / `StructurePredictions`
are pre-695 implicit generic aliases and `DesignLoss` uses `Generic[A]`. Three styles
coexisting; it type-checks, but the class-scoped `A` shadows the module-level one.

Note that `A` is only a marker — it is never bound to a union of the two array types,
because that union cannot be spelled without importing both frameworks.
