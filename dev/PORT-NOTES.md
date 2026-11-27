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

| campaign (one per tree, `--core benchmark`, gates at 0.0) | paired trajectories | agreement | design s, new / head |
|---|---|---|---|
| hPDL1, 10 rounds | 24 | 24/24 bit-identical on all 10 metrics | 5.94 / 6.01 = 0.989 |
| hPDL1, 103 rounds | 4 | 4/4 | 61.7 / 62.2 = 0.991 |
| hIL7RA, 10 rounds | 24 | 24/24, `coldspot_repel` included | 13.14 / 13.05 = 1.007 |
| hIL7RA, 103 rounds | 4 | 4/4 | 137.7 / 138.1 = 0.997 |
| hIL2R, 10 rounds | 12 | 12/12, two-chain target | 49.95 / 49.94 = 1.000 |
| hIL2R, 103 rounds | 2 | 2/2 | 521.7 / 522.1 = 0.999 |
| mPDL1, 10 rounds | 12 | 12/12 | 6.34 / 6.34 = 1.000 |

**62 paired trajectories across four targets and two schedules, every one bit-identical.**
Only trajectory 1 compiled in either tree of any campaign, so the new `device=` calls add
no recompile — they execute on tracers inside the graph, where commitment does not exist
(trap 2). The seven design-time ratios run 0.989 to 1.007, with no sign to them.

The distributions those campaigns measured, per trajectory, at the last recorded round:

| | pLDDT | i_pTM | i_pTM range |
|---|---|---|---|
| hPDL1, 10 rounds (n=24) | 0.9396 ± 0.0062 | 0.198 ± 0.026 | 0.147–0.266 |
| hPDL1, 103 rounds (n=4) | 0.9780 ± 0.0150 | 0.633 ± 0.225 | 0.342–0.865 |
| hIL7RA, 10 rounds (n=24) | 0.9368 ± 0.0061 | 0.141 ± 0.026 | 0.074–0.179 |
| hIL7RA, 103 rounds (n=4) | 0.9668 ± 0.0174 | 0.481 ± 0.249 | 0.234–0.811 |
| hIL2R, 10 rounds (n=12) | 0.9297 ± 0.0055 | 0.189 ± 0.059 | 0.147–0.367 |
| mPDL1, 10 rounds (n=12) | 0.9413 ± 0.0039 | 0.204 ± 0.035 | 0.157–0.295 |

A 10-round schedule is not a design: i_pTM sits at 0.14–0.20 and its spread is narrow
because nothing has converged, so a change could pass there and still ruin real campaigns.
The 103-round schedule is where trajectories separate — i_pTM 0.23 to 0.87 — which makes it
the schedule a quality claim has to be made on, and also the expensive one: 4 trajectories
there cost about as much as 24 short ones, and 4 is nowhere near enough to make the claim.
That tension, not the arithmetic, is what makes a quality comparison hard here.

Two things to take from the campaign rows. First, **a change that only adds `device=` is
bit-identical, not merely in-distribution** — with `--xla_gpu_autotune_level=0` the HLO is
unchanged, so every metric of every paired trajectory agrees to the last bit, on both
schedules and on all four targets. That is a stronger result than the 6-seed comparison of
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

# OpenFold design model: parity findings on `of2-design-model`

Context for the uncommitted diff on this branch (`af2_pytorch.py` +106/−2,
`protein.py` +20/−2, `dev/parity/torch_device.py` +10/−1). Read the "what is not true"
section before reusing any number from an earlier write-up of this work.

## The confounder that invalidates campaign-level backend comparison

`campaign_design_model` samples **a different AlphaFold model per gradient step** from
`MULTIMER_POOL` (5 models), and each backend samples from its own RNG: `af2.py` uses
`jax.random.split` + `jax.random.randint` (threefry), `af2_pytorch.py` uses
`torch.randint` on a `torch.Generator` (Philox). These are different algorithms, so the
sequences cannot coincide.

Measured by wrapping `_resolve_model_name` / `_sample_design_model` and logging every
choice over one campaign per backend from the same seed:

| | |
|---|---|
| step-by-step agreement | **9/33 = 27%** (chance = 20%) |
| of2 model mix | `{1: 12, 2: 3, 3: 7, 4: 8, 5: 3}` |
| af2 model mix | `{1: 11, 2: 10, 3: 3, 4: 7, 5: 5}` |

So ~73% of "paired" steps compared **different model weights**, and the mix itself is
skewed (model_2: 3 vs 10; model_3: 7 vs 3) — which is a mechanism for a systematic
metric offset with a consistent sign, and is the most likely explanation for the
consistent-sign offsets seen across all four shapes (ptm +, iptm +, plddt −, pae −).

It also resolves the discrepancy that forced this investigation: the forward test agreed
to 1e-3 relative while campaign step 1 disagreed 2–11%. The forward test forced one model
on both sides; campaign step 1 compared model_4 against model_1.

**Consequence for anyone comparing backends:** paired comparison is impossible unless
`model=` is forced. Byte-identical inputs are necessary but not sufficient — verify the
weights match too, which is one line of instrumentation.

**Consequence for users:** `campaign_seed` does not reproduce a trajectory across
backends even though its name implies it does. A fix would derive of2's model index from
the same JAX key `af2.py` uses, keeping the torch `Generator` only for dropout masks,
which genuinely cannot be matched.

## What this branch fixes (all verified)

1. **`align_prediction_to_target_template`** in `af2_pytorch.py`, porting `af2.py:88`,
   called on the whole complex before the per-chain split as `af2.py:480` does. The
   `.bool()` casts in it are load-bearing: `complex_residue_arrays` returns `atom_mask`
   as float, and openfold's `final_atom_mask` is float. Verified against af2: of2 binder
   centroid moved to `[21.44, −54.00, −32.89]` against af2's `[21.53, −54.01, −32.77]`,
   max coordinate deviation 99.985 → 1.853 Å, Rg and RMSD bit-identical.
2. **`experimentally_resolved_ca`** on metrics (`loss.py:228` reads it; it was commented
   out).
3. **Weighted loss terms written back to metrics** after `total_loss.backward()`,
   mirroring `af2.py:502-505` including the overwrite guard.
4. **`kabsch` CUDA device bug in shared code** (`protein.py`) — a pre-existing bug on
   `main`, not a port artefact. `xp.eye(3, ...)` at `:164` and `xp.ones(2, ...)` at `:177`
   were built on the host. Since `xp.where` evaluates both branches, **every `kabsch`
   call on a CUDA tensor raised a device mismatch**, not only the zero-weight case. Fixed
   with `eye_beside` / `ones_beside` next to the existing `arange_beside` (which exists
   because passing `device=` to an eager JAX call commits the array and can trigger
   recompilation — see the comment at `protein.py:86-88`). This is logically independent
   of this branch and could land on `main` on its own.
5. **`dev/parity/torch_device.py`** gained `kabsch_rotation` and `kabsch_zero_weights`.
   The script written specifically to catch a missing `device=` had missed this entire
   bug class, because nothing in the torch path called `kabsch` until of2 gained the
   template alignment. Now 6 checks, all on the device.

Verified by: `torch_device.py` all-on-device; `parity.py` 62/62, zero errors; campaign
smoke test exit 0 with all 8 loss terms plus `experimentally_resolved_ca` present on
metrics; step-1 loss moved 9.5e-07.

## Backend agreement that does hold

Only measurements that forced a single model, or that involve no model pool, survive:

| check | result |
|---|---|
| forward pass, one-hot sequence, 3 seeds | pTM 7.2e-4, ipTM 8.7e-4, pLDDT 1.4e-3 relative |
| — for scale | smaller than af2's own fp32-vs-bf16 gap |
| predicted structure | 0.59 Å CA RMSD |
| distogram-loss gradient w.r.t. sequence logits | cosine 0.99846, magnitude ratio 1.0123 |
| JAX-side differential parity after the `protein.py` fix | 62/62 |

On the two standing questions this answers: the weights are the identical
`params_model_1_multimer_v3.npz` on both sides, so "fit for JAX" is not a property the
weights can have, and the 1e-4 forward agreement is direct evidence openfold's import is
faithful. PyTorch and JAX reverse-mode AD are equivalent by construction; measured at
0.998 cosine, with of2 at least as reproducible as af2.

## What is **not** true — retracted claims

Do not reuse these; they are in the session history and they are wrong.

- **The blocked significance test** (ptm +0.0102 p=0.014, ipTM +0.0213 p=0.020, pae_mean
  −0.59 p=0.012) over 12 trajectory pairs. Withdrawn: unsynchronised model draws, so it
  measures model-pool variance, not backend parity.
- **The compactness / radius-of-gyration trajectory analysis**, and "the entire loss gap
  is compactness". True at step 1 only, the most degenerate point; mean ΔRg across the
  trajectory is +0.86 ± 0.84 (t = 1.03, n.s.) with oscillating sign.
- **"The heads agree while the coordinates diverge"**: this mixed measurement
  conditions — the 1e-3 figure is from the forced-single-model forward test, the 5.4 Å
  figure from campaign step 1, where the metrics in fact disagree 2.7–11.4%.
- **"A 17× gradient defect in of2"**: artefact of an ill-conditioned pLDDT probe measured
  with XLA autotuning on. With a distogram loss and `--xla_gpu_autotune_level=0`:
  cosine 0.99846, ratio 1.0123.
- **"af2's bf16 pLDDT gradient is 2238× its fp32 one"**: did not reproduce at another
  seed (ratio 1.3, cosine −0.322), and the per-element spread (p10 452, p90 2903, with
  83.4% of the norm in 10 of 1100 elements) rules out the uniform 2^11 scale factor I
  speculated. The honest statement is that af2's bf16 pLDDT gradient bears **no reliable
  relation** to its fp32 one, which is why the prior session's bf16 recommendation is
  withdrawn on numerical grounds. The speed case stands on its own (1.199 s vs 1.774 s).
- **"Initial sequences were unmatched"**: refuted, md5 `89b49b0e403c` on both. The
  inference from differing step-0 losses was unsound — loss depends on the prediction and
  on `bucket_length`, not on the sequence alone.
- **"l64 is unpadded"**: wrong, 64 + 115 = 179 → 192, pad 13. Every shape tested had af2
  padding of 5–22 residues.

Two methodological traps worth remembering:

- **Finite differences cannot validate these gradients.** `aatype = sequence.argmax(-1)`
  feeds the atom14/atom37 masks, `template_aatype` and `msa`, so the loss is piecewise
  constant in the sequence; FD ratios swung 300× between eps 1e-2 and 1e-3.
- **`plddt_loss` (`loss.py:206-212`) is a masked mean over designed binder residues
  only.** A pLDDT measured over the whole complex is a different quantity and must not be
  compared against it.

Also: af2 is **not** padding-invariant — 1.27 Å CA RMSD between `bucket=1` and
`bucket=32` on the same input — but the effect on the metrics is small (3e-4 on pTM,
−0.04 Å on Rg), so padding is not an explanation for anything here.

Stage schedule, measured rather than derived: `softmax_weight` at the first six steps is
**0.225, 0.45, 0.675, 0.90, 0.95, 1.0** — the first value is 0.225, not the 0.25 that
reading `_linear_sequence_schedule(0, 1, 0, 4)` suggests.

## Remaining gaps in `af2_pytorch.py`, not fixed

- **Model sampling cannot be synchronised with af2** (above). Highest value of the four.
- `sym_id = ones_like(asym_id)` and `entity_id = asym_id.clone()` diverge from
  `residue_entity_ids(..., multi_chain_binders)` when `copies > 1` and
  `oligomer_tie='symmetric'`.
- **No cyclic-offset support**, and `__init__` accepts no `cyclic_offset_mode`,
  `multi_chain_binders`, `target_flexibility` or `target_pad_length`.
- `settings.get('design_recycles')` is read without the `DEFAULT_SETTINGS` fallback
  `af2.py` uses. The two coincide at 1 today; it breaks silently if the default changes.
