# Rebuilding the GPU box

Everything outside this git repo lives on the instance's ephemeral disk, so a scrapped
instance loses the venv, the AlphaFold parameters and any scratch output. Rebuilding:

```bash
git clone git@github.com:kilianmandon/BindCraft2.git && cd BindCraft2
bash install.sh cuda13        # venv + deps + the one-time 5.3 GB parameter download
.venv/bin/python -m bindcraft.selfcheck cuda13   # should exit 0

# the PyTorch AlphaFold reference implementation, a plain clone, no local edits
git clone git@github.com:aqlaboratory/openfold.git bindcraft/openfold
git -C bindcraft/openfold checkout be2ec18

# openfold imports itself absolutely (`from openfold...`), in 48 of its files, so the clone's
# root has to be a path entry: the package it offers is bindcraft/openfold/openfold.
echo "$PWD/bindcraft/openfold" > .venv/lib/python3.12/site-packages/openfold-clone.pth
.venv/bin/python -c "import openfold.config; print(openfold.__file__)"

# openfold's own dependencies that bindcraft does not carry
.venv/bin/python -m pip install dm-tree biopython modelcif

# the extension openfold imports unconditionally, built as the stub upstream uses off-GPU
.venv/bin/python dev/build_openfold_stub.py
.venv/bin/python -c "from openfold.model.model import AlphaFold; print(AlphaFold.__module__)"
```

A `.pth` file is one line holding a directory, read by `site` at startup, so this puts
`openfold` on the path for the venv's interpreter and for every subprocess it spawns —
which matters, because the design workers are subprocesses. Use an absolute path: a
relative one is resolved against site-packages, not against the repo. `pip install -e
bindcraft/openfold` would be the tidier route and does not work here, because openfold's
`setup.py` builds CUDA extensions (see below).

`[tool.pyright]` in `pyproject.toml` carries the same path for the type checker, which
cannot see a `.pth`.

The path alone gets `openfold.config` importing but not `openfold.model.model`, which holds
the `AlphaFold` class: `openfold/utils/kernel/attention_core.py` does a bare
`importlib.import_module("attn_core_inplace_cuda")` at module scope, and `primitives.py`
pulls it in. (`openfold.config` escapes it only because `openfold/model/__init__.py` is
empty, so importing that subpackage never reaches `model.py`.)

**The CUDA kernel cannot be built here and does not need to be.** torch refuses an extension
whose toolkit differs in CUDA major version from the one it was built with, and this box has
nvcc 12.8 against torch's cu132. openfold's own `setup.py` already has the answer: off-GPU it
builds the same module from `softmax_cuda_stub.cpp`, whose two functions raise "not
implemented on CPU". `dev/build_openfold_stub.py` builds exactly that branch, from openfold's
sources, and drops the .so in the clone root. The stub is reached only through
`AttentionCoreFunction`, which `primitives.py` uses only when
`use_memory_efficient_kernel=True` — off by default — so a design runs on the ordinary
attention implementation and anything that does ask for the kernel gets a clear error instead
of a wrong number.

That script exists rather than `CUDA_HOME= python setup.py build_ext` because openfold picks
its branch on `torch.utils.cpp_extension.CUDA_HOME`, which falls back to `/usr/local/cuda`
whenever that directory exists, whatever the environment says. Nor is `pip install
bindcraft/openfold` wanted: it would copy the package into site-packages, where it would
shadow the clone the `.pth` points at.

Verified after both steps: `AlphaFold(model_config('model_1_multimer_v3'))` builds in ~11 s
and `import_jax_weights_` loads `params_model_1_multimer_v3.npz` into it in 0.3 s, giving
93.2 M parameters — the same count the npz holds.

`install.sh` reads the accelerator off the driver, so the explicit `cuda13` is only needed
if `nvidia-smi` is unavailable or reports something else. `--no-weights` skips the download.

Verified working on this configuration:

| | |
|---|---|
| card | RTX 4090, 24 GB (driver CUDA UMD 13.3) |
| python | 3.12.3 |
| jax | 0.11.2 + `jax[cuda13]` |
| torch | 2.15.0+cu132 |
| array-api-compat / -extra | 1.16.0 / 0.11.5 |

jax and torch were confirmed to coexist on the card in one process.

`dev/PORT-NOTES.md` records the state of the array-API port: what has been verified, the
two traps worth knowing about before measuring anything, and what is left to do.

`dev/parity/` holds the differential harness used to check the array-API port; read its
README before trusting any numerical comparison, because XLA autotuning has to be turned
off for one.
