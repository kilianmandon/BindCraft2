"""Eager-mode cost of the namespace indirection, on CPU. No jit: this is the per-call Python
overhead that a torch predictor would pay on every step, since torch does not trace."""
import json, sys, time
import numpy as np, jax
jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
from bindcraft.protein import Protein, ResidueFlags, real_residue_weights, has_residue_flag, kabsch
import bindcraft.loss as L

rng = np.random.default_rng(0)
def mk(n, designed):
    f = np.full(n, int(ResidueFlags.DESIGN if designed else ResidueFlags.SEQUENCE), dtype=np.uint8)
    return Protein(sequence=jnp.asarray(rng.normal(size=(n,20)), dtype=jnp.float32),
                   atoms=jnp.asarray(rng.normal(size=(n,37,3))*9, dtype=jnp.float32),
                   atom_mask=jnp.asarray(rng.random((n,37)) > .1),
                   flags=jnp.asarray(f), residue_index=jnp.arange(1,n+1,dtype=jnp.int32))
binder, target = mk(55, True), mk(115, False)
cplx = {'binder': binder, 'target': target}
ca = binder.atoms[:,1]; w = real_residue_weights(binder.flags)
CASES = {
 'real_residue_weights':  lambda: real_residue_weights(binder.flags),
 'has_residue_flag':      lambda: has_residue_flag(binder.flags, ResidueFlags.DESIGN),
 'pairwise_atom_distances': lambda: L.pairwise_atom_distances(ca, ca),
 'chain_atom_coordinates':lambda: L.chain_atom_coordinates(binder),
 'binder_sequence_offsets': lambda: L.binder_sequence_offsets(binder),
 'kabsch':                lambda: kabsch(ca, ca[::-1], w),
 'complex_residue_weights': lambda: L.complex_residue_weights(cplx),
 'amino_acid_probabilities': lambda: L.amino_acid_probabilities(binder.sequence),
 '_masked_mean':          lambda: L._masked_mean(ca[:,0], w),
}
out = {}
for name, fn in CASES.items():
    for _ in range(5): jax.block_until_ready(fn())          # warm
    n, t0 = 200, time.perf_counter()
    for _ in range(n): jax.block_until_ready(fn())
    out[name] = (time.perf_counter() - t0) / n * 1e6         # microseconds per call
json.dump(out, open(sys.argv[1],'w'), indent=1, sort_keys=True)
for k,v in sorted(out.items()): print(f'  {k:28s} {v:9.1f} us/call')
