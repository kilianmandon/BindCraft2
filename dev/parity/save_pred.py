import numpy as np, jax, jax.numpy as jnp, sys
from bindcraft.protein import Protein, ResidueFlags
from bindcraft.af2 import AlphaFoldDesignModel
from bindcraft.model_weights import model_weights
params, _ = model_weights(download=False)
target = Protein.from_structure('settings/target/structures/hPDL1.pdb')['A']
rng = np.random.default_rng(0)
a0, m0, i0 = Protein.unresolved_atom_arrays(55)
binder = Protein(sequence=jnp.asarray(0.01*rng.normal(size=(55,20)), dtype=jnp.float16),
                 atoms=a0, atom_mask=m0, residue_index=i0,
                 flags=jnp.full((55,), int(ResidueFlags.DESIGN), dtype=jnp.uint8))
model = AlphaFoldDesignModel(presets='model_1_multimer_v3', data_dir=params, num_recycle=1,
                             dropout=False, key=jax.random.PRNGKey(0))
preds = model.predict({'complex': {'binder': binder, 'target': target}}, model='model_1_multimer_v3')
save = {}
for chain, p in preds['complex'].protein_complex.items():
    for field in ('sequence','atoms','atom_mask','flags','residue_index'):
        save[f'pred.{chain}.{field}'] = np.asarray(getattr(p, field))
for chain, p in {'binder': binder, 'target': target}.items():
    for field in ('sequence','atoms','atom_mask','flags','residue_index'):
        save[f'in.{chain}.{field}'] = np.asarray(getattr(p, field))
for k, v in preds['complex'].metrics.items():
    save[f'met.{k}'] = np.asarray(v)
np.savez(sys.argv[1], **save)
print('saved', len(save), 'arrays')
