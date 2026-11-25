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
```

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

`dev/parity/` holds the differential harness used to check the array-API port; read its
README before trusting any numerical comparison, because XLA autotuning has to be turned
off for one.
