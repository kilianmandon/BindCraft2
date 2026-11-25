"""Call a fixed set of loss/protein helpers with seeded inputs; dump results as JSON.
Run once per tree (reference vs working) and diff the two dumps."""
import sys, os
_ROOT = os.environ.get('BC_TREE')
if _ROOT:
    sys.meta_path = [f for f in sys.meta_path
                     if 'editable' not in getattr(f, '__module__', '').lower()
                     and 'Editable' not in type(f).__name__]
    sys.path.insert(0, _ROOT)
import json, sys
import numpy as np
import jax.numpy as jnp
from bindcraft.protein import (Protein, ResidueFlags, has_residue_flag, real_residue_mask,
    real_residue_count, real_residue_weights, redesignable_residue_mask,
    alignment_matrix_product, kabsch)
import bindcraft.loss as L
import bindcraft.sequence_optimization as SO

rng = np.random.default_rng(0)
def mk(length, designed):
    flags = np.full(length, int(ResidueFlags.SEQUENCE | ResidueFlags.TEMPLATE), dtype=np.uint8)
    if designed: flags[:] = int(ResidueFlags.DESIGN)
    flags[-2:] = int(ResidueFlags.PADDING)
    return Protein(sequence=jnp.asarray(rng.normal(size=(length, 20)), dtype=jnp.float16),
                   atoms=jnp.asarray(rng.normal(size=(length, 37, 3)) * 8, dtype=jnp.float16),
                   atom_mask=jnp.asarray(rng.random((length, 37)) > 0.2),
                   flags=jnp.asarray(flags), residue_index=jnp.arange(1, length + 1, dtype=jnp.int32))
binder, target = mk(14, True), mk(11, False)
cplx = {'binder': binder, 'target': target}
ca_b = binder.atoms[:, 1].astype(jnp.float32)
ca_t = target.atoms[:, 1].astype(jnp.float32)
w = real_residue_weights(binder.flags)
vals = jnp.asarray(rng.normal(size=14), dtype=jnp.float32)

dgram = jnp.asarray(rng.normal(size=(14, 14, 16)), dtype=jnp.float32)
dgram_pair = jnp.asarray(rng.normal(size=(14, 14)), dtype=jnp.float32)
bool_mask = jnp.asarray(rng.random((14, 14)) > 0.3)
dgram_bins_sel = jnp.asarray(rng.random((16,)) > 0.5, dtype=jnp.float32)
CASES = {
 'has_residue_flag':        lambda: has_residue_flag(binder.flags, ResidueFlags.DESIGN),
 'padded_to_sequence':      lambda: binder.padded_to(20).sequence,
 'padded_to_flags':         lambda: binder.padded_to(20).flags,
 'padded_to_atoms':         lambda: binder.padded_to(20).atoms,
 'padded_to_atom_mask':     lambda: binder.padded_to(20).atom_mask,
 'padded_to_residue_index': lambda: binder.padded_to(20).residue_index,
 'padded_to_noop':          lambda: binder.padded_to(10).residue_index,
 'trimmed_to':              lambda: binder.trimmed_to(8).residue_index,
 'omitted_aa_mask':         lambda: SO.omitted_amino_acid_mask(binder.sequence),
 'straight_through_onehot': lambda: SO.straight_through_one_hot(L.amino_acid_probabilities(binder.sequence)),
 'seq_features_logits':     lambda: SO.sequence_features_from_logits(jnp.asarray(binder.sequence, dtype=jnp.float32), jnp.asarray(0.5), jnp.asarray(0.0), jnp.asarray(1.0), jnp.asarray(2.0)),
 'seq_features_onehot':     lambda: SO.sequence_features_from_logits(jnp.asarray(binder.sequence, dtype=jnp.float32), jnp.asarray(1.0), jnp.asarray(1.0), jnp.asarray(0.01), jnp.asarray(2.0)),
 'seq_features_bias':       lambda: SO.sequence_features_from_logits(jnp.asarray(binder.sequence, dtype=jnp.float32), jnp.asarray(1.0), jnp.asarray(0.0), jnp.asarray(1.0), jnp.asarray(2.0), jnp.asarray(rng.normal(size=20), dtype=jnp.float32)),
 'real_residue_mask':       lambda: real_residue_mask(binder.flags),
 'real_residue_count':      lambda: real_residue_count(binder.flags),
 'real_residue_weights':    lambda: real_residue_weights(binder.flags),
 'redesignable_mask':       lambda: redesignable_residue_mask(binder.flags),
 'alignment_matrix_product':lambda: alignment_matrix_product(ca_b.T, ca_b),
 'kabsch':                  lambda: kabsch(ca_b, ca_b[::-1], w),
 '_masked_mean':            lambda: L._masked_mean(vals, w),
 'amino_acid_probabilities':lambda: L.amino_acid_probabilities(binder.sequence),
 'chain_group_rows':        lambda: L.chain_group_rows(cplx, ('binder',)),
 'chain_group_res_weights': lambda: L.chain_group_residue_weights(cplx, ('binder',)),
 'complex_residue_weights': lambda: L.complex_residue_weights(cplx),
 'binder_domain_membership':lambda: L.binder_domain_membership(binder, (0,0,1,1,1), 2),
 'helical_pair_mask':       lambda: L.helical_pair_mask(binder),
 'chain_residue_mask':      lambda: _try(L.chain_residue_mask, (25, slice(0,14), cplx), (25, slice(0,14))),
 'expand_chain_res_mask':   lambda: L.expand_chain_residue_mask(25, slice(0,14), w),
 'pairwise_atom_distances': lambda: L.pairwise_atom_distances(ca_b, ca_t),
 'chain_atom_coordinates':  lambda: L.chain_atom_coordinates(binder),
 'masked_coord_centroid':   lambda: L.masked_coordinate_centroid(ca_b, w),
 'binder_sequence_offsets': lambda: L.binder_sequence_offsets(binder),
 'soft_maximum':            lambda: L.soft_maximum(vals, 0.5),
 'soft_maximum_2d':         lambda: L.soft_maximum(dgram_pair, 0.5),
 'mhc_epitope_I':           lambda: L.mhc_epitope_score(L.amino_acid_probabilities(binder.sequence), L.mhc_panels('human')[0], 0.5, 0.3, 0.1),
 'mhc_epitope_I_masked':    lambda: L.mhc_epitope_score(L.amino_acid_probabilities(binder.sequence), L.mhc_panels('human')[0], 0.5, 0.3, 0.1, L.redesignable_residue_mask(binder.flags)),
 'mhc_epitope_II':          lambda: L.mhc_epitope_score(L.amino_acid_probabilities(binder.sequence), L.mhc_panels('human')[1], 0.5, 0.0, 0.1),
 'mhc_epitope_short':       lambda: L.mhc_epitope_score(L.amino_acid_probabilities(mk(6, True).sequence), L.mhc_panels('human')[0], 0.5, 0.0, 0.1),
 'binder_ca_pseudo_dihedral':lambda: L.binder_ca_pseudo_dihedral(ca_b),
 'chain_terminal_rows':     lambda: L.chain_terminal_rows(binder.flags, 3),
 'receptor_chain_index':    lambda: L.receptor_chain_index(binder),
 'distogram_bin_distances': lambda: _try(L.distogram_bin_distances, (jnp.zeros((4,4,16), dtype=jnp.float32),), (16,)),
 'pseudo_beta_coordinates': lambda: L.pseudo_beta_coordinates(binder),
 'binder_chain_mask':       lambda: L.binder_chain_mask(cplx, 'binder', 25),
 'binder_distant_pair_mask':lambda: L.binder_distant_pair_mask(cplx, 'binder', 25, 4),
 'aligned_binder_tm_score': lambda: L.aligned_binder_tm_score(ca_b, ca_b[::-1], w),
 'core_aligned_iface_rmsd': lambda: L.core_aligned_interface_rmsd(ca_b, ca_b[::-1], w, w),
 'template_distance_dev':   lambda: L.template_distance_deviation(binder, binder, jnp.asarray(rng.random((14,14)) > .5, dtype=jnp.float32)),
 'best_contact_mean':       lambda: L.best_contact_mean(dgram_pair, 3, bool_mask),
 'distogram_pair_loss':     lambda: L.distogram_pair_loss(dgram, dgram_bins_sel, False),
 'distogram_pair_loss_bin': lambda: L.distogram_pair_loss(dgram, dgram_bins_sel, True),
 'mutual_pair_choice':      lambda: L.mutual_pair_choice(dgram_pair, 0.3),
 'binder_loop_sheets':      lambda: L.binder_loop_geometry_propensity(ca_b, w, distinguish_sheets=True),
 'boxcar_even_window':      lambda: L.binder_strand_propensity(ca_b, w, smoothing=4),
 'boxcar_odd_window':       lambda: L.binder_strand_propensity(ca_b, w, smoothing=5),
 'protease_site_score':     lambda: L.protease_site_score(L.amino_acid_probabilities(binder.sequence), w),
 'binder_loop_geometry':    lambda: L.binder_loop_geometry_propensity(ca_b, w),
 'binder_strand_propensity':lambda: L.binder_strand_propensity(ca_b, w),
 'elastic_network_cov':     lambda: L.elastic_network_covariance(ca_b, w, 2.0, 0.1),
 'domain_linker_membership':lambda: L.domain_linker_membership(jnp.asarray(rng.random((14,14)), dtype=jnp.float32), L.binder_domain_membership(binder, (0,)*7+(1,)*7, 2), 0.1, 0.03),
 'induced_fit_masks_resid': lambda: L.induced_fit_interface_masks({'complex': cplx}, {}, ca_b, w, 'complex', 'target', 8.0, interface_residues=(1,3,5)),
 'binder_domain_mem_3dom':  lambda: L.binder_domain_membership(binder, (0,0,0,1,1,1,2,2), 3),
}
def _try(fn, *argsets):
    last = None
    for args in argsets:
        try: return fn(*args)
        except TypeError as e: last = e
    raise last

def flat(v):
    if isinstance(v, tuple): return [flat(i) for i in v]
    a = np.asarray(v, dtype=np.float64)
    return {'dtype': str(getattr(v, 'dtype', '')), 'shape': list(a.shape), 'sum': round(float(np.nansum(a)), 6),
            'absmax': round(float(np.nanmax(np.abs(a))) if a.size else 0.0, 6),
            'head': [round(float(x), 6) for x in a.ravel()[:5]]}
out = {}
for name, fn in CASES.items():
    try: out[name] = flat(fn())
    except Exception as e: out[name] = f'ERROR {type(e).__name__}: {e}'
json.dump(out, open(sys.argv[1], 'w'), indent=1, sort_keys=True)
print(f'wrote {sys.argv[1]}: {sum(1 for v in out.values() if isinstance(v,str))} errors of {len(out)}')
