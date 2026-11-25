"""Compare the exact feature dict handed to AlphaFold, computed eagerly on concrete arrays.
No jit, no autotuning, no GPU nondeterminism: any difference here is a real code difference."""
import json, sys
import numpy as np, jax, jax.numpy as jnp
import bindcraft.af2 as af2
from bindcraft.prediction import residue_chain_ids, concatenate_chain_arrays
from bindcraft.protein import Protein, ResidueFlags, real_residue_weights

jax.config.update('jax_platform_name', 'cpu')   # eager, on CPU: removes every source of GPU noise

target = Protein.from_structure('settings/target/structures/hPDL1.pdb')['A']
rng = np.random.default_rng(0)
atoms0, mask0, index0 = Protein.unresolved_atom_arrays(55)
binder = Protein(sequence=jnp.asarray(0.01*rng.normal(size=(55,20)), dtype=jnp.float16),
                 atoms=atoms0, atom_mask=mask0, residue_index=index0,
                 flags=jnp.full((55,), int(ResidueFlags.DESIGN), dtype=jnp.uint8))
complexes = {'binder': binder, 'target': target}
names = tuple(sorted(complexes))
lengths = tuple(len(complexes[n]) for n in names)
arrays = concatenate_chain_arrays(names, complexes, 'sequence', 'atoms', 'atom_mask', 'flags', 'residue_index')
sequence, atoms, atom_mask, flags, residue_index = (arrays[k] for k in ('sequence','atoms','atom_mask','flags','residue_index'))
asym_id = residue_chain_ids(lengths)
entity_id = af2.residue_entity_ids(names, lengths, ())
seq_mask = real_residue_weights(flags)

out = {}
def record(label, value):
    if isinstance(value, dict):
        for sub, inner in sorted(value.items()):
            record(f'{label}.{sub}', inner)
        return
    a = np.asarray(value)
    out[label] = {'shape': list(a.shape), 'dtype': str(a.dtype),
                  'sum': repr(float(a.astype(np.float64).sum())),
                  'absmax': repr(float(np.abs(a.astype(np.float64)).max())) if a.size else '0.0'}
record('_seq_mask', seq_mask)
record('_flags', flags)
record('_residue_index', residue_index)
record('_asym_id', asym_id)
record('_entity_id', entity_id)

features, profile = af2.prepare_design_sequence_features(
    sequence, flags, jnp.asarray(1.0), jnp.asarray(1.0), jnp.asarray(0.01), jnp.asarray(2.0), None)
record('_design_features', features)
record('_design_profile', profile)

model_inputs = af2.alphafold_input_features(features, profile, atoms, atom_mask, residue_index,
                                            asym_id, seq_mask, flags, jnp.asarray(False),
                                            'direction', entity_id, 0.0, False)
for key, value in model_inputs.items():
    record(key, value)
json.dump(out, open(sys.argv[1],'w'), indent=1, sort_keys=True)
print(f'recorded {len(out)} arrays')
