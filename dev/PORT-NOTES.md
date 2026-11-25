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
- **Four non-scalar array creations omit `device=`** and will land on the host under torch
  CUDA: `loss.py:152` (`chain_group_rows`' `arange`, used as an index),
  `loss.py:164` (`binder_domain_membership`'s `asarray(domain_ids)`), and
  `loss.py:908`/`910`/`911`/`918`/`921` (`elastic_network_covariance`'s `eye`/`arange`).
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
  accident of its flat namespace — the same trap as `axis` vs `dim`.
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
