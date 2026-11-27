import functools
import inspect
import random
from typing import Callable, Generic, NamedTuple
from bindcraft.developability import EPITOPE_CORE_LENGTH, HYDROPHOBICITY, MHCPanel, mhc_panels, protease_panel
from bindcraft.protein import A, AMINO_ACIDS, ATOM_INDEX, BINDER_ALONE, Protein, ProteinStates, ResidueFlags, StructurePrediction, StructurePredictions, alignment_matrix_product, block_diagonal, boxcar_smooth, elu, flat_nonzero, has_residue_flag, kabsch, log_softmax, real_residue_count, real_residue_mask, real_residue_weights, redesignable_residue_mask, relu, sigmoid, softmax, stop_gradient
import array_api_compat
import array_api_extra as xpx

class DesignLoss(NamedTuple, Generic[A]):
    function: Callable[[ProteinStates[A], StructurePredictions], A]
    weight: float
    required_states: frozenset[str]
    interface_mask: A | None = None
REGISTERED_LOSSES: dict[str, Callable] = {}
LOSS_TARGET_WEIGHTING: dict[str, str] = {}
BINDER_ROLE_PARAMETERS = frozenset({'binder', 'chain'})
PADDING_ANCHOR_STIFFNESS = 1000.0

def loss(name: str, target_weighting: str='binder_only') -> Callable:
    def register(loss_function: Callable) -> Callable:
        REGISTERED_LOSSES[name] = loss_function
        LOSS_TARGET_WEIGHTING[name] = target_weighting
        return loss_function
    return register

@functools.lru_cache(maxsize=None)
def _bind_metric_parameters(loss_function: Callable, loss_parameters: tuple[tuple[str, object], ...]) -> Callable:
    return functools.partial(loss_function, **dict(loss_parameters))

def _bound_metric(metric_function: Callable, metric_parameters: dict) -> Callable:
    return _bind_metric_parameters(metric_function, tuple(sorted((name, _immutable_metric_parameter(value)) for name, value in metric_parameters.items())))

def _immutable_metric_parameter(value):
    return tuple(_immutable_metric_parameter(item) for item in value) if isinstance(value, (list, tuple)) else value

def resolve_prediction_state(predictions: StructurePredictions, prediction_state: str) -> str:
    return prediction_state if prediction_state in predictions else next((name for name in predictions if name != BINDER_ALONE), prediction_state)

def resolve_target_chain(protein_complex: dict, name: str, state: str) -> str:
    return name if name in protein_complex else f'{name}_{state}' if f'{name}_{state}' in protein_complex else name

def resolve_binder_role(metric_function: Callable, metric_parameters: dict, binder_chain: str) -> dict:
    parameters = inspect.signature(metric_function).parameters
    return {**metric_parameters, **{name: binder_chain for name in parameters.keys() & BINDER_ROLE_PARAMETERS if metric_parameters.get(name, parameters[name].default) == 'binder' != binder_chain}}

def bind_state_metric(metric_function: Callable, metric_settings: dict) -> tuple[Callable, frozenset[str]]:
    metric_parameters = dict(metric_settings.get('params', {}))
    if 'prediction_state' in metric_settings:
        metric_parameters['prediction_state'] = metric_settings['prediction_state']
    prediction_state_parameter = inspect.signature(metric_function).parameters.get('prediction_state')
    if prediction_state_parameter is None:
        metric_parameters.pop('prediction_state', None)
    prediction_state = metric_parameters.get('prediction_state', prediction_state_parameter.default if prediction_state_parameter is not None else ())
    required_states = () if 'prediction_state' not in metric_parameters and prediction_state == 'complex' else (prediction_state,) if isinstance(prediction_state, str) else prediction_state
    return _bound_metric(metric_function, metric_parameters), frozenset(required_states)

STATE_LOSS_PARAMETERS = ('prediction_state', 'reference_state', 'binder_shapes')

def renamed_loss_name(name: str, state_names: dict[str, str]) -> str:
    base, separator, state = name.partition('.')
    return f'{base}.{state_names[state]}' if separator and state in state_names else name

def _renamed_states(value, state_names: dict[str, str]):
    if isinstance(value, str):
        return state_names.get(value, value)
    return tuple(_renamed_states(item, state_names) for item in value) if isinstance(value, tuple) else value

def renamed_state_losses(losses: dict[str, DesignLoss], state_names: dict[str, str]) -> dict[str, DesignLoss]:
    renamed = {}
    for name, entry in losses.items():
        keywords = getattr(entry.function, 'keywords', None)
        function = entry.function if keywords is None else _bind_metric_parameters(entry.function.func, tuple(sorted((key, _renamed_states(value, state_names) if key in STATE_LOSS_PARAMETERS else value) for key, value in keywords.items())))
        renamed[renamed_loss_name(name, state_names)] = DesignLoss(function, entry.weight, frozenset(state_names.get(state, state) for state in entry.required_states), entry.interface_mask)
    return renamed

def resolve_loss_weight(weight, seed: int=0) -> float:
    if not isinstance(weight, (list, tuple)) or not weight:
        return float(weight or 0)
    values = tuple(float(value) for value in weight)
    if len(values) > 2:
        return random.Random(seed).choice(tuple(value for value in values if value) or (0.0,))
    drawn = random.Random(seed).uniform(min(values), max(values))
    return drawn or max(values, key=abs)

def sampled_loss_weights(settings: dict, seed: int=0) -> dict[str, float]:
    return {name: resolve_loss_weight(value, seed) for name, value in settings.items() if name.startswith('weights_') and isinstance(value, (list, tuple))}

PARATOPE_CONFORMATIONS = 'extended', 'folded_back'

def sampled_paratope_conformation(settings: dict, seed: int=0) -> str:
    conformations = settings.get('paratope_conformations') or ()
    named = (conformations,) if isinstance(conformations, str) else tuple(conformations)
    unknown = [name for name in named if name not in PARATOPE_CONFORMATIONS]
    if unknown:
        raise ValueError(f'unknown paratope_conformations {", ".join(unknown)}; a paratope loop is {" or ".join(PARATOPE_CONFORMATIONS)}')
    return random.Random(seed).choice(named) if named else ''

def build_losses(settings: dict, seed: int=0) -> dict[str, DesignLoss]:
    losses = {}
    folded_back = sampled_paratope_conformation(settings, seed) == 'folded_back'
    for name, loss_function in REGISTERED_LOSSES.items():
        weight = resolve_loss_weight(settings.get(f'weights_{name}'), seed)
        if name == 'binder_intra_coldspot' and folded_back:
            weight = -weight
        if weight:
            bound_loss, required_states = bind_state_metric(loss_function, settings.get('losses', {}).get(name, {}))
            losses[name] = DesignLoss(bound_loss, weight, required_states)
    return losses

def weighted_design_loss(losses: dict[str, DesignLoss[A]], protein_states: ProteinStates, predictions: StructurePredictions) -> A:
    frozen_interfaces = frozen_interface_arguments(losses)

    terms = [entry.weight * entry.function(protein_states, predictions, **frozen_interfaces.get(name, {})) for name, entry in losses.items() if entry.required_states <= set(protein_states)]
    if len(terms)==0:
        # TODO: Check if we really want to raise on empty valid losses
        raise RuntimeError('No losses provided.')
    return functools.reduce(lambda a,b: a+b, terms)

def globular_radius(residues: int) -> float:
    return 2.38 * residues ** 0.365

def _masked_mean(values: A, mask: A, eps: float=1e-08) -> A:
    xp = array_api_compat.array_namespace(values)
    values, mask = xp.asarray(values, dtype=xp.float32), xp.asarray(mask, dtype=xp.float32)
    return xp.sum(xp.where(mask>0, values*mask, 0.0)) / (xp.sum(mask) + eps)


def amino_acid_probabilities(sequence: A) -> A:
    xp = array_api_compat.array_namespace(sequence)
    probabilities = xp.astype(sequence, xp.float32)
    return xp.where((xp.min(probabilities, axis=-1, keepdims=True) >= 0) & xpx.isclose(xp.sum(probabilities, axis=-1, keepdims=True), 1.0), probabilities, softmax(probabilities))

def chain_residue_slices(protein_complex: dict[str, Protein]) -> dict[str, slice]:
    chain_slices, residue_start = {}, 0
    for chain_name in sorted(protein_complex):
        chain_slices[chain_name] = slice(residue_start, residue_start + len(protein_complex[chain_name]))
        residue_start += len(protein_complex[chain_name])
    return chain_slices

def binder_copy_chains(protein_complex: dict[str, Protein], binder: str) -> tuple[str, ...]:
    prefix = binder.rsplit('_', 1)[0] if binder.rsplit('_', 1)[-1].isdigit() else binder
    return tuple(name for name in sorted(protein_complex) if name.startswith(f'{prefix}_') and name.rsplit('_', 1)[-1].isdigit()) or (binder,)

def binder_protomer_groups(protein_complex: dict[str, Protein], binder: str, per_protomer: bool) -> tuple[tuple[str, ...], ...]:
    binder_chains = binder_copy_chains(protein_complex, binder)
    return tuple((name,) for name in binder_chains) if per_protomer else (binder_chains,)

def chain_group_rows(protein_complex: dict[str, Protein], chain_names: tuple[str, ...]) -> A:
    xp = array_api_from_protein_complex(protein_complex)
    chain_slices = chain_residue_slices(protein_complex)
    device = array_device_from_protein_complex(protein_complex)
    return xp.concat([xp.arange(chain_slices[name].start, chain_slices[name].stop, device=device) for name in chain_names])

def chain_group_residue_weights(protein_complex: dict[str, Protein], chain_names: tuple[str, ...]) -> A:
    xp = array_api_from_protein_complex(protein_complex)
    return xp.concat([real_residue_weights(protein_complex[name].flags) for name in chain_names])

def complex_residue_weights(protein_complex: dict[str, Protein]) -> A:
    xp = array_api_from_protein_complex(protein_complex)
    return xp.concat([real_residue_weights(protein_complex[name].flags) for name in sorted(protein_complex)])

def binder_domain_membership(protein: Protein, domain_ids: tuple[int, ...], domain_count: int) -> A:
    xp = array_api_compat.array_namespace(protein.sequence)
    domains = xpx.pad(xp.asarray(domain_ids, dtype=xp.int32, device=array_api_compat.device(protein.sequence)), (0, max(len(protein) - len(domain_ids), 0)))[:len(protein)]
    return xpx.one_hot(domains, domain_count) * real_residue_weights(protein.flags)[:, None]

def helical_pair_mask(protein: Protein) -> A:
    residue_mask = real_residue_mask(protein.flags)
    return (binder_sequence_offsets(protein) == 3) & residue_mask[:, None] & residue_mask[None, :]

#The complex is carried in only to say which namespace and device the mask belongs to: residue_count
#and chain_slice are plain Python, so there is nothing else here to read them off.
def chain_residue_mask(residue_count: int, chain_slice: slice, protein_complex: dict[str, Protein]) -> A:
    xp = array_api_from_protein_complex(protein_complex)
    residue_indices = xp.arange(residue_count, device=array_device_from_protein_complex(protein_complex))
    return (residue_indices >= chain_slice.start) & (residue_indices < chain_slice.stop)

def expand_chain_residue_mask(residue_count: int, chain_slice: slice, chain_values: A, dtype=bool) -> A:
    xp = array_api_compat.array_namespace(chain_values)
    empty = xp.zeros(residue_count, dtype=dtype, device=array_api_compat.device(chain_values))
    return xpx.at(empty)[chain_slice].set(xp.astype(chain_values, empty.dtype))

def pairwise_atom_distances(first: A, second: A, eps: float=0.0001) -> A:
    xp = array_api_compat.array_namespace(first)
    first, second = xp.astype(first, xp.float32), xp.astype(second, xp.float32)
    return xp.sqrt(xp.sum((first[:, None] - second[None, :]) ** 2, axis=-1) + eps)

def chain_atom_coordinates(protein: Protein, atom: str='CA') -> tuple[A, A]:
    xp = array_api_compat.array_namespace(protein.atoms)
    return xp.astype(protein.atoms[:, ATOM_INDEX[atom]], xp.float32), xp.astype(protein.atom_mask[:, ATOM_INDEX[atom]], xp.float32)

def masked_coordinate_centroid(coordinates: A, mask: A) -> A:
    xp = array_api_compat.array_namespace(coordinates)
    return xp.sum(coordinates * mask[:, None], axis=0) / (xp.sum(mask) + 1e-08)

def binder_sequence_offsets(protein: Protein) -> A:
    xp = array_api_compat.array_namespace(protein.residue_index)
    offsets = protein.residue_index[:, None] - protein.residue_index[None, :]
    cyclic_distance = xp.minimum(xp.abs(offsets), real_residue_count(protein.flags) - xp.abs(offsets))
    cyclic_offsets = xp.where(cyclic_distance < xp.abs(offsets), -cyclic_distance, cyclic_distance) * xp.sign(offsets)
    cyclic_residues = has_residue_flag(protein.flags, ResidueFlags.CYCLIC) & real_residue_mask(protein.flags)
    return xp.where(cyclic_residues[:, None] & cyclic_residues[None, :], cyclic_offsets, offsets)

@loss('plddt_loss')
def plddt_loss(protein_states: ProteinStates[A], predictions: StructurePredictions[A], prediction_state: str='binder_alone', chain: str='binder') -> A:
    xp = array_api_from_protein_states(protein_states)
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    protein_complex = protein_states[prediction_state]
    binder_chains = binder_copy_chains(protein_complex, chain)
    confidence = predictions[prediction_state].metrics['plddt'][chain_group_rows(protein_complex, binder_chains)]
    return _masked_mean(1 - confidence, xp.concat([has_residue_flag(protein_complex[name].flags, ResidueFlags.DESIGN) for name in binder_chains]))

@loss('target_plddt', target_weighting='every_target')
def target_plddt_loss(protein_states: ProteinStates[A], predictions: StructurePredictions[A], prediction_state: str='complex', target: str='target') -> A:
    xp = array_api_from_protein_states(protein_states)
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    target = resolve_target_chain(protein_states[prediction_state], target, prediction_state)
    hotspot_mask = has_residue_flag(protein_states[prediction_state][target].flags, ResidueFlags.HOTSPOT)
    return _masked_mean(1 - predictions[prediction_state].metrics['plddt'][chain_residue_slices(protein_states[prediction_state])[target]], xp.where(hotspot_mask.any(), hotspot_mask, real_residue_mask(protein_states[prediction_state][target].flags)))

@loss('experimentally_resolved')
def experimentally_resolved_loss(protein_states: ProteinStates[A], predictions: StructurePredictions[A], prediction_state: str='complex', chain: str='binder') -> A:
    xp = array_api_from_protein_states(protein_states)
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    protein_complex = protein_states[prediction_state]
    binder_chains = binder_copy_chains(protein_complex, chain)
    resolved = predictions[prediction_state].metrics['experimentally_resolved_ca'][chain_group_rows(protein_complex, binder_chains)]
    return _masked_mean(1 - resolved, xp.concat([has_residue_flag(protein_complex[name].flags, ResidueFlags.DESIGN) for name in binder_chains]))

def array_api_from_protein_states(protein_states: ProteinStates):
    proteins = sum([list(protein_state.values()) for protein_state in protein_states.values()], [])
    arrays = []
    for p in proteins:
        arrays += [p.sequence, p.atoms, p.atom_mask, p.flags, p.residue_index]

    return array_api_compat.array_namespace(*arrays)

#The device the complex lives on, read off a deterministically chosen chain: an array built with
#xp.arange and no device argument lands on the CPU under torch and then will not meet a CUDA tensor.
def array_device_from_protein_complex(protein_complex: dict[str, Protein]):
    return array_api_compat.device(protein_complex[sorted(protein_complex)[0]].residue_index)

def array_api_from_protein_complex(protein_complex: dict[str, Protein]):
    proteins = list(protein_complex.values())
    arrays = []
    for p in proteins:
        arrays += [p.sequence, p.atoms, p.atom_mask, p.flags, p.residue_index]

    return array_api_compat.array_namespace(*arrays)


@loss('sequence_entropy')
def sequence_entropy_loss(protein_states: ProteinStates[A], predictions: StructurePredictions[A], prediction_state: str='complex', chain: str='binder') -> A:
    xp = array_api_from_protein_states(protein_states)
    protein_complex = protein_states[resolve_prediction_state(predictions, prediction_state)]
    binder_chains = binder_copy_chains(protein_complex, chain)
    logits = xp.astype(xp.concat([protein_complex[name].sequence for name in binder_chains]), xp.float32)
    return _masked_mean(-xp.sum(softmax(logits) * log_softmax(logits), axis=-1), xp.concat([has_residue_flag(protein_complex[name].flags, ResidueFlags.DESIGN) for name in binder_chains]))

DISTOGRAM_DEPENDENT_LOSSES = ('distogram_cce', 'binder_coldspot', 'binder_intra_hotspot', 'binder_intra_coldspot', 'coldspot_repel', 'interface_contacts', 'non_contact', 'binder_contacts', 'binder_helicity', 'non_helical', 'multidomain')

def chain_pair_pae_loss(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str, aligned_chains: tuple[str, ...] | None, reference_chains: tuple[str, ...] | None) -> A:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    pae = predictions[prediction_state].metrics['pae'] / 31.0
    protein_complex = protein_states[prediction_state]
    if aligned_chains is None and reference_chains is None:
        weights = complex_residue_weights(protein_complex)
        return _masked_mean(pae, weights[:, None] * weights[None, :])
    aligned_rows, reference_rows = chain_group_rows(protein_complex, aligned_chains), chain_group_rows(protein_complex, reference_chains)
    aligned_weights, reference_weights = chain_group_residue_weights(protein_complex, aligned_chains), chain_group_residue_weights(protein_complex, reference_chains)
    return _masked_mean(pae[aligned_rows[:, None], reference_rows[None, :]], aligned_weights[:, None] * reference_weights[None, :])

#axis=None keeps the whole-array reduction every existing caller wants; naming an axis is for the
#scores of a stack of epitope cores, where each core's alleles have to be reduced on their own.
def soft_maximum(values: A, temperature: float, axis: int | None=None) -> A:
    xp = array_api_compat.array_namespace(values)
    if axis is None:
        #Left exactly as it was: on anything but the 1-D arrays every caller passes, this weights by a
        #softmax over the last axis and then sums the lot, which is odd but is not ours to change here.
        return xp.sum(values * softmax(values / temperature))
    return xp.sum(values * softmax(values / temperature, axis=axis), axis=axis)

def mhc_epitope_score(sequence: A, panel: MHCPanel, coupling_weight: float, hydrophobicity_weight: float, temperature: float, residue_mask: A | None=None) -> A:
    xp = array_api_compat.array_namespace(sequence)
    device = array_api_compat.device(sequence)
    if len(panel.anchor_matrices) == 0 or len(sequence) < EPITOPE_CORE_LENGTH:
        return xp.asarray(0.0, device=device)
    anchor_matrices, coupling_matrices = xp.asarray(panel.anchor_matrices, device=device), xp.asarray(panel.coupling_matrices, device=device)
    hydrophobicity, groove_burial = xp.asarray(HYDROPHOBICITY, device=device), xp.asarray(panel.groove_burial, device=device)
    #Every core is scored at once rather than one at a time under vmap, so each term below carries a
    #leading core axis n that the per-allele axis a sits beside: (n, a) in, one score per core out.
    def epitope_core_scores(core_starts: A) -> A:
        cores = sequence[core_starts[:, None] + xp.arange(EPITOPE_CORE_LENGTH, device=device)]
        anchor_match = xp.einsum('apq,npq->na', anchor_matrices, cores)
        anchor_coupling = sum((xp.einsum('ni,aij,nj->na', cores[:, first], coupling_matrices[:, pair], cores[:, second]) for pair, (first, second, _) in enumerate(panel.coupled_anchors)))
        groove_burial_match = xp.sum((cores @ hydrophobicity) * groove_burial, axis=-1)
        allele_scores = anchor_match + coupling_weight * anchor_coupling + hydrophobicity_weight * groove_burial_match[:, None]
        return soft_maximum(allele_scores, temperature, axis=-1)
    core_starts = xp.arange(len(sequence) - EPITOPE_CORE_LENGTH + 1, device=device)
    core_scores = epitope_core_scores(core_starts)
    if residue_mask is not None:
        core_residues_mask = xp.astype(residue_mask, xp.float32)[core_starts[:, None] + xp.arange(EPITOPE_CORE_LENGTH, device=device)]
        core_is_real = xp.min(core_residues_mask, axis=-1) > 0
        core_scores = xp.where(core_is_real, core_scores, -1000000000.0)
    return soft_maximum(core_scores, temperature) / EPITOPE_CORE_LENGTH

@loss('humanization')
def humanization_loss(protein_states: ProteinStates[A], predictions: StructurePredictions[A], prediction_state: str='complex', chain: str='binder', species: str='human', coupling_weight: float=0.5, hydrophobicity_weight: float=0.0, mhc_class_ii_weight: float=1.0, temperature: float=0.1) -> A:
    xp = array_api_from_protein_states(protein_states)
    protein_complex = protein_states[resolve_prediction_state(predictions, prediction_state)]
    binder_chains = binder_copy_chains(protein_complex, chain)
    worst_epitope = lambda panel: soft_maximum(xp.stack([mhc_epitope_score(amino_acid_probabilities(protein_complex[name].sequence), panel, coupling_weight, hydrophobicity_weight, temperature, redesignable_residue_mask(protein_complex[name].flags)) for name in binder_chains]), temperature)
    mhc_class_i_panel, mhc_class_ii_panel = mhc_panels(species)
    return worst_epitope(mhc_class_i_panel) + mhc_class_ii_weight * worst_epitope(mhc_class_ii_panel)

def protease_site_score(sequence: A, residue_mask: A) -> A:
    xp = array_api_compat.array_namespace(sequence)
    device = array_api_compat.device(sequence)
    panel = protease_panel()
    p1, p1_block, weights = xp.asarray(panel.p1, device=device), xp.asarray(panel.p1_block, device=device), xp.asarray(panel.weights, device=device)
    p1_probability, block_probability = sequence @ p1.T, sequence @ p1_block.T
    p1_prime_is_real = xp.concat([xp.astype(residue_mask[1:], xp.float32), xp.zeros((1,), dtype=xp.float32, device=device)])[:, None]
    block_at_p1_prime = xp.where(p1_prime_is_real > 0, xp.concat([block_probability[1:], xp.zeros((1, block_probability.shape[1]), dtype=block_probability.dtype, device=device)]), 1.0)
    return _masked_mean((p1_probability * (1.0 - block_at_p1_prime)) @ weights, residue_mask)

def pooled_protease_site_score(chains) -> A:
    scored = [(protease_site_score(sequence, residue_mask), residue_mask.sum()) for sequence, residue_mask in chains]
    xp = array_api_compat.array_namespace(scored[0][0])
    rows = sum(count for _score, count in scored)
    return xp.where(rows > 0, sum(score * count for score, count in scored) / xp.maximum(rows, xp.asarray(1, dtype=rows.dtype)), 0.0)

@loss('protease_sites')
def protease_site_loss(protein_states: ProteinStates[A], predictions: StructurePredictions[A], prediction_state: str='complex', chain: str='binder') -> A:
    xp = array_api_from_protein_states(protein_states)
    protein_complex = protein_states[resolve_prediction_state(predictions, prediction_state)]
    return pooled_protease_site_score([(xp.astype(softmax(protein_complex[name].sequence), xp.float32), redesignable_residue_mask(protein_complex[name].flags)) for name in binder_copy_chains(protein_complex, chain)])

def binder_residue_exposure(protein_states: ProteinStates[A], predictions: StructurePredictions[A], prediction_state: str, chain: str, radius: float=10.0, contact_temperature: float=2.0, neighbor_threshold: float=16.0, temperature: float=4.0) -> A:
    xp = array_api_from_protein_states(protein_states)
    predicted_complex = predictions[prediction_state].protein_complex
    complex_coordinates = xp.concat([chain_atom_coordinates(predicted_complex[name])[0] for name in sorted(predicted_complex)])
    complex_mask = complex_residue_weights(protein_states[prediction_state])
    binder_coordinates, _ = chain_atom_coordinates(predicted_complex[chain])
    neighbor_count = xp.sum(sigmoid((radius - pairwise_atom_distances(binder_coordinates, complex_coordinates)) / contact_temperature) * complex_mask[None, :], axis=-1) - 1.0
    return sigmoid((neighbor_threshold - neighbor_count) / temperature)

def binder_ca_pseudo_dihedral(coordinates: A, eps: float=1e-08) -> A:
    xp = array_api_compat.array_namespace(coordinates)
    b1, b2, b3 = coordinates[1:-2] - coordinates[:-3], coordinates[2:-1] - coordinates[1:-2], coordinates[3:] - coordinates[2:-1]
    b2_hat = b2 / (xp.linalg.vector_norm(b2, axis=-1, keepdims=True) + eps)
    v = b1 - xp.sum(b1 * b2_hat, axis=-1, keepdims=True) * b2_hat
    w = b3 - xp.sum(b3 * b2_hat, axis=-1, keepdims=True) * b2_hat
    return xp.atan2(xp.sum(xp.linalg.cross(b2_hat, v) * w, axis=-1), xp.sum(v * w, axis=-1))

def binder_strand_propensity(coordinates: A, residue_weights: A, extension_cutoff: float=0.3, temperature: float=0.2, smoothing: int=4) -> A:
    xp = array_api_compat.array_namespace(coordinates)
    length = coordinates.shape[0]
    if length < 4:
        return xp.zeros(length, dtype=coordinates.dtype, device=array_api_compat.device(coordinates))
    extension = -xp.cos(binder_ca_pseudo_dihedral(coordinates))
    valid_window = residue_weights[:-3] * residue_weights[3:]
    strand_probability = sigmoid((extension - extension_cutoff) / temperature) * valid_window
    smoothed = boxcar_smooth(xp.concat([strand_probability, xp.zeros(3, dtype=strand_probability.dtype, device=array_api_compat.device(strand_probability))]), smoothing)
    return xp.clip(smoothed, 0.0, 1.0)

def binder_loop_geometry_propensity(coordinates: A, residue_weights: A, cutoff: float=7.0, temperature: float=0.8, smoothing: int=4, distinguish_sheets: bool=False, strand_extension_cutoff: float=0.3, strand_temperature: float=0.2) -> A:
    xp = array_api_compat.array_namespace(coordinates)
    length = coordinates.shape[0]
    if length < 4:
        return xp.ones(length, dtype=coordinates.dtype, device=array_api_compat.device(coordinates))
    valid_window = residue_weights[:-3] * residue_weights[3:]
    close_probability = sigmoid((cutoff - xp.sqrt(xp.sum(xp.square(coordinates[:-3] - coordinates[3:]), axis=-1) + 1e-08)) / temperature) * valid_window
    structured = xp.clip(boxcar_smooth(xp.concat([close_probability, xp.zeros(3, dtype=close_probability.dtype, device=array_api_compat.device(close_probability))]), smoothing), 0.0, 1.0)
    if distinguish_sheets:
        structured = xp.maximum(structured, binder_strand_propensity(coordinates, residue_weights, strand_extension_cutoff, strand_temperature, smoothing))
    return 1.0 - structured

def chain_loop_susceptibility(protein_states: ProteinStates[A], predictions: StructurePredictions[A], prediction_state: str, chain: str, measure: str, plddt_threshold: float, plddt_temperature: float, window: int, distinguish_sheets: bool) -> A:
    xp = array_api_from_protein_states(protein_states)
    protein = protein_states[prediction_state][chain]
    coordinates, _ = chain_atom_coordinates(predictions[prediction_state].protein_complex[chain])
    residue_weights = real_residue_weights(protein.flags)
    exposure = binder_residue_exposure(protein_states, predictions, prediction_state, chain)
    if measure == 'geometry':
        loopiness = binder_loop_geometry_propensity(coordinates, residue_weights, distinguish_sheets=distinguish_sheets)
    else:
        plddt_loopiness = sigmoid((plddt_threshold - predictions[prediction_state].metrics['plddt'][chain_residue_slices(protein_states[prediction_state])[chain]]) / plddt_temperature)
        loopiness = xp.maximum(plddt_loopiness, binder_loop_geometry_propensity(coordinates, residue_weights, distinguish_sheets=distinguish_sheets)) if measure == 'both' else plddt_loopiness
    susceptibility = exposure * loopiness * residue_weights
    return susceptibility * boxcar_smooth(susceptibility, window)

@loss('exposed_loops')
def exposed_loop_loss(protein_states: ProteinStates[A], predictions: StructurePredictions[A], prediction_state: str='complex', chain: str='binder', measure: str='plddt', plddt_threshold: float=0.85, plddt_temperature: float=0.05, window: int=3, distinguish_sheets: bool=True) -> A:
    xp = array_api_from_protein_states(protein_states)
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    binder_chains = binder_copy_chains(protein_states[prediction_state], chain)
    susceptibility = [chain_loop_susceptibility(protein_states, predictions, prediction_state, name, measure, plddt_threshold, plddt_temperature, window, distinguish_sheets) for name in binder_chains]
    return _masked_mean(xp.concat(susceptibility), xp.concat([real_residue_mask(protein_states[prediction_state][name].flags) for name in binder_chains]))

def chain_terminal_rows(flags: A, terminus_length: int) -> A:
    xp = array_api_compat.array_namespace(flags)
    c_terminus = real_residue_count(flags) - 1
    terminus_rows = xp.arange(terminus_length, device=array_api_compat.device(flags))
    return xp.concat([xp.clip(terminus_rows, 0, c_terminus), xp.clip(c_terminus - terminus_rows, 0, None)])

@loss('exposed_termini')
def exposed_termini_loss(protein_states: ProteinStates[A], predictions: StructurePredictions[A], prediction_state: str='complex', chain: str='binder', terminus_length: int=3) -> A:
    xp = array_api_from_protein_states(protein_states)
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    exposure, weights = [], []
    for name in binder_copy_chains(protein_states[prediction_state], chain):
        flags = protein_states[prediction_state][name].flags
        terminal_rows = chain_terminal_rows(flags, terminus_length)
        exposure.append(binder_residue_exposure(protein_states, predictions, prediction_state, name)[terminal_rows])
        weights.append(real_residue_weights(flags)[terminal_rows])
    return _masked_mean(xp.concat(exposure), xp.concat(weights))

@loss('binder_pae')
def binder_pae_loss(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='binder_alone', chain: str | None=None, domain_ids: tuple[int, ...]=(), per_protomer: bool=False) -> A:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    if domain_ids:
        chain_slice = chain_residue_slices(protein_states[prediction_state])[chain or 'binder']
        binder_pae = predictions[prediction_state].metrics['pae'][chain_slice, chain_slice] / 31.0
        domain_membership = binder_domain_membership(protein_states[prediction_state][chain or 'binder'], domain_ids, max(domain_ids) + 1)
        return _masked_mean(binder_pae, domain_membership @ domain_membership.T)
    if not chain:
        return chain_pair_pae_loss(protein_states, predictions, prediction_state, None, None)
    protomer_groups = binder_protomer_groups(protein_states[prediction_state], chain, per_protomer)
    return sum((chain_pair_pae_loss(protein_states, predictions, prediction_state, group, group) for group in protomer_groups)) / len(protomer_groups)

@loss('interface_pae', target_weighting='binds_target')
def interface_pae_loss(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', binder: str='binder', target: str='target') -> A:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    return chain_pair_pae_loss(protein_states, predictions, prediction_state, (binder,), (resolve_target_chain(protein_states[prediction_state], target, prediction_state),))

@loss('compactness')
def radius_of_gyration_loss(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='binder_alone', chain: str | None=None, eps: float=1e-08, per_protomer: bool=False) -> A:
    xp = array_api_from_protein_states(protein_states)
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    designed_chain_groups = binder_protomer_groups(protein_states[prediction_state], chain, per_protomer) if chain else [(name,) for name in sorted(protein_states[prediction_state])]
    radius_loss_sum, designed_chain_count = xp.asarray(0.0), xp.asarray(eps)
    for group in designed_chain_groups:
        proteins = [predictions[prediction_state].protein_complex[name] for name in group]
        coordinates = xp.concat([chain_atom_coordinates(protein)[0] for protein in proteins])
        resolved_ca_mask = xp.concat([chain_atom_coordinates(protein)[1] for protein in proteins])
        radius = xp.sqrt(_masked_mean(xp.square(coordinates - masked_coordinate_centroid(coordinates, resolved_ca_mask)).sum(-1), resolved_ca_mask) + eps)
        designed_chain_weight = xp.asarray(1.0) if chain else has_residue_flag(protein_states[prediction_state][group[0]].flags, ResidueFlags.DESIGN).any()
        designed_chain_weight = xp.astype(designed_chain_weight, xp.float32)
        radius_loss_sum, designed_chain_count = radius_loss_sum + elu(radius - globular_radius(sum((real_residue_count(protein.flags) for protein in proteins)))) * designed_chain_weight, designed_chain_count + designed_chain_weight
    return radius_loss_sum / designed_chain_count

@loss('iptm_loss', target_weighting='binds_target')
def iptm_loss(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex') -> A:
    return 1 - predictions[resolve_prediction_state(predictions, prediction_state)].metrics['iptm']

@loss('ptm_loss', target_weighting='every_target')
def ptm_loss(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex') -> A:
    return 1 - predictions[resolve_prediction_state(predictions, prediction_state)].metrics['ptm']

def receptor_chain_index(protein: Protein) -> A:
    from bindcraft.protein_preparation import RECEPTOR_CHAIN_BREAK_GAP
    xp = array_api_compat.array_namespace(protein.residue_index)
    device = array_api_compat.device(protein.residue_index)
    chain_breaks = xp.astype(xp.diff(protein.residue_index) >= RECEPTOR_CHAIN_BREAK_GAP, xp.int32)
    return xp.concat([xp.zeros(1, dtype=xp.int32, device=device), xp.cumulative_sum(chain_breaks)])

def template_distance_deviation(template: Protein, predicted: Protein, selected_pairs: A) -> A:
    xp = array_api_compat.array_namespace(predicted.atoms)
    template_coordinates, template_mask = chain_atom_coordinates(template)
    predicted_coordinates, predicted_mask = chain_atom_coordinates(predicted)
    valid_mask = template_mask * predicted_mask
    pair_mask = selected_pairs * valid_mask[:, None] * valid_mask[None, :]
    distance_error = pairwise_atom_distances(predicted_coordinates, predicted_coordinates, 1e-08) - pairwise_atom_distances(template_coordinates, template_coordinates, 1e-08)
    return xp.sqrt(_masked_mean(xp.square(distance_error), pair_mask) + 1e-08) * xp.astype(xp.sum(pair_mask) > 0, xp.float32)

@loss('target_rmsd', target_weighting='every_target')
def target_rmsd_loss(protein_states: ProteinStates[A], predictions: StructurePredictions[A], prediction_state: str='complex', target: str='target') -> A:
    xp = array_api_from_protein_states(protein_states)
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    target = resolve_target_chain(protein_states[prediction_state], target, prediction_state)
    return template_distance_deviation(protein_states[prediction_state][target], predictions[prediction_state].protein_complex[target], xp.asarray(1.0, device=array_device_from_protein_complex(protein_states[prediction_state])))

@loss('target_rigidity', target_weighting='every_target')
def target_rigidity_loss(protein_states: ProteinStates[A], predictions: StructurePredictions[A], prediction_state: str='complex', target: str='target') -> A:
    xp = array_api_from_protein_states(protein_states)
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    target = resolve_target_chain(protein_states[prediction_state], target, prediction_state)
    template = protein_states[prediction_state][target]
    chain_index = receptor_chain_index(template)
    return template_distance_deviation(template, predictions[prediction_state].protein_complex[target], xp.astype(chain_index[:, None] != chain_index[None, :], xp.float32))

def best_contact_mean(values: A, contact_count: int | float, mask: A, eps: float=0.0001) -> A:
    xp = array_api_compat.array_namespace(values)
    device = array_api_compat.device(values)
    contact_ranking = stop_gradient(xp.argsort(xp.where(mask, values, xp.asarray(xp.inf, dtype=values.dtype, device=device))))
    ranked_values = xp.take_along_axis(values, contact_ranking, axis=-1)
    selected = (xp.arange(values.shape[-1], device=device) < contact_count) & xp.take_along_axis(mask, contact_ranking, axis=-1)
    return xp.sum(xp.where(selected, ranked_values, xp.asarray(0.0, dtype=values.dtype, device=device)), axis=-1) / (xp.sum(selected, axis=-1) + eps)

# Takes the distogram itself as input and not num_bins, so that we are able to grab the array api from it
def distogram_bin_distances(distogram: A) -> A:
    xp = array_api_compat.array_namespace(distogram)
    device = array_api_compat.device(distogram)
    first_bin = xp.zeros(1, dtype=xp.float32, device=device)
    return xp.concat((first_bin, xp.linspace(2.3125, 21.6875, num=distogram.shape[-1] - 1, dtype=xp.float32, device=device)))

def distogram_pair_loss(distogram: A, selected_bins: A, binary: bool) -> A:
    xp = array_api_compat.array_namespace(distogram)
    distogram = xp.astype(distogram, xp.float32)
    if binary:
        return -xp.log(xp.sum(selected_bins * softmax(distogram), axis=-1) + 1e-08)
    selected_probabilities = softmax(distogram - 10000000.0 * (1 - xp.astype(selected_bins, xp.float32)))
    return -xp.sum(selected_probabilities * log_softmax(distogram), axis=-1)

def mean_selected_contact_loss(pair_loss: A, contacts_per_residue: int | float, contact_residue_count: int | float, residue_mask: A, contact_pair_mask: A) -> A:
    return best_contact_mean(best_contact_mean(pair_loss, contacts_per_residue, contact_pair_mask), contact_residue_count, residue_mask)

def distogram_contact_loss(distogram: A, cutoff: float, contacts_per_residue: int | float, contact_residue_count: int | float, residue_mask: A, contact_pair_mask: A, binary: bool=False) -> A:
    return mean_selected_contact_loss(distogram_pair_loss(distogram, distogram_bin_distances(distogram) < cutoff, binary), contacts_per_residue, contact_residue_count, residue_mask, contact_pair_mask)

def pseudo_beta_coordinates(protein: Protein) -> tuple[A, A]:
    xp = array_api_compat.array_namespace(protein.atoms)
    beta_coordinates, beta_mask = chain_atom_coordinates(protein, 'CB')
    alpha_coordinates, alpha_mask = chain_atom_coordinates(protein, 'CA')
    return xp.where(beta_mask[:, None] > 0, beta_coordinates, alpha_coordinates), xp.where(beta_mask > 0, beta_mask, alpha_mask)

def nonbinding_residue_repulsion_loss(distogram: A, residue_mask: A, partner_residue_mask: A, cutoff: float, contacts_per_residue: int, eligible_pair_mask: A | None=None) -> A:
    xp = array_api_compat.array_namespace(distogram)
    pair_loss = distogram_pair_loss(distogram, distogram_bin_distances(distogram) < cutoff, False)
    contact_pair_mask = residue_mask[:, None] & partner_residue_mask[None, :]
    contact_pair_mask = contact_pair_mask if eligible_pair_mask is None else contact_pair_mask & eligible_pair_mask
    contact_loss = mean_selected_contact_loss(pair_loss, contacts_per_residue, float('inf'), residue_mask, contact_pair_mask)
    return xp.where(contact_pair_mask.any(), -contact_loss, 0.0)

@loss('distogram_cce', target_weighting='every_target')
def distogram_cce_loss(protein_states: ProteinStates[A], predictions: StructurePredictions[A], prediction_state: str='complex', chain: str='target') -> A:
    xp = array_api_from_protein_states(protein_states)
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    chain = resolve_target_chain(protein_states[prediction_state], chain, prediction_state)
    chain_slice = chain_residue_slices(protein_states[prediction_state])[chain]
    reference_coordinates, reference_mask = pseudo_beta_coordinates(protein_states[prediction_state][chain])
    distogram = xp.astype(predictions[prediction_state].metrics['distogram'][chain_slice, chain_slice], xp.float32)
    reference_bins = xp.sum(xp.astype(pairwise_atom_distances(reference_coordinates, reference_coordinates)[..., None] > distogram_bin_distances(distogram)[1:], xp.int64), axis=-1)
    cross_entropy = -xp.sum(xp.astype(xpx.one_hot(reference_bins, distogram.shape[-1]), distogram.dtype) * log_softmax(distogram), axis=-1)
    return _masked_mean(cross_entropy, reference_mask[:, None] * reference_mask[None, :])

@loss('com_distance', target_weighting='every_target')
def com_distance_loss(protein_states: ProteinStates[A], predictions: StructurePredictions[A], prediction_state: str='complex', binder: str='binder', target: str='target', radius_ratio: float=0.9, eps: float=1e-08) -> A:
    xp = array_api_from_protein_states(protein_states)
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    target = resolve_target_chain(protein_states[prediction_state], target, prediction_state)
    predicted_complex = predictions[prediction_state].protein_complex
    binder_coordinates, binder_mask = chain_atom_coordinates(predicted_complex[binder])
    target_coordinates, target_mask = chain_atom_coordinates(predicted_complex[target])
    binder_centroid = masked_coordinate_centroid(binder_coordinates, binder_mask)
    binder_radius = xp.sqrt(_masked_mean(xp.sum(xp.square(binder_coordinates - binder_centroid), axis=-1), binder_mask) + eps)
    centroid_distance = xp.linalg.vector_norm(binder_centroid - masked_coordinate_centroid(target_coordinates, target_mask))
    return relu(centroid_distance - radius_ratio * binder_radius)

@loss('binder_coldspot', target_weighting='every_target')
def binder_coldspot_loss(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', binder: str='binder', target: str='target', cutoff: float=8.0, contacts_per_residue: int=1, designed_only: bool=True) -> A:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    target = resolve_target_chain(protein_states[prediction_state], target, prediction_state)
    protein_complex = protein_states[prediction_state]
    distogram = predictions[prediction_state].metrics['distogram']
    chain_slices, residue_count = chain_residue_slices(protein_complex), distogram.shape[0]
    framework_mask = binder_repelled_mask(protein_complex, binder, residue_count, designed_only)
    target_mask = chain_residue_mask(residue_count, chain_slices[target], protein_complex) & (complex_residue_weights(protein_complex) > 0)
    return nonbinding_residue_repulsion_loss(distogram, target_mask, framework_mask, cutoff, contacts_per_residue)

def binder_distant_pair_mask(protein_complex: dict[str, Protein[A]], binder: str, residue_count: int, sequence_separation: int) -> A:
    xp = array_api_from_protein_complex(protein_complex)
    device = array_device_from_protein_complex(protein_complex)
    chain_slices = chain_residue_slices(protein_complex)
    distant_pairs = xp.ones((residue_count, residue_count), dtype=xp.bool, device=device)
    for name in binder_copy_chains(protein_complex, binder):
        chain_slice = chain_slices[name]
        distant_pairs = xpx.at(distant_pairs)[chain_slice, chain_slice].set(xp.abs(binder_sequence_offsets(protein_complex[name])) >= sequence_separation)
    return distant_pairs

def binder_intra_contact(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str, binder: str, cutoff: float, contacts_per_residue: int, sequence_separation: int) -> A:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    protein_complex = protein_states[prediction_state]
    distogram = predictions[prediction_state].metrics['distogram']
    residue_count = distogram.shape[0]
    framework_mask = binder_framework_mask(protein_complex, binder, residue_count)
    paratope_mask = binder_binding_mask(protein_complex, binder, residue_count)
    distant_pairs = binder_distant_pair_mask(protein_complex, binder, residue_count, sequence_separation)
    return nonbinding_residue_repulsion_loss(distogram, framework_mask, paratope_mask, cutoff, contacts_per_residue, distant_pairs)

@loss('binder_intra_coldspot')
def binder_intra_coldspot_loss(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', binder: str='binder', cutoff: float=8.0, contacts_per_residue: int=1, sequence_separation: int=20) -> A:
    return binder_intra_contact(protein_states, predictions, prediction_state, binder, cutoff, contacts_per_residue, sequence_separation)

@loss('binder_intra_hotspot')
def binder_intra_hotspot_loss(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', binder: str='binder', cutoff: float=8.0, contacts_per_residue: int=1, sequence_separation: int=20) -> A:
    return -binder_intra_contact(protein_states, predictions, prediction_state, binder, cutoff, contacts_per_residue, sequence_separation)

@loss('coldspot_repel', target_weighting='every_target')
def coldspot_repel_loss(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', binder: str='binder', target: str='target', cutoff: float=8.0, contacts_per_residue: int=1) -> A:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    target = resolve_target_chain(protein_states[prediction_state], target, prediction_state)
    protein_complex = protein_states[prediction_state]
    distogram = predictions[prediction_state].metrics['distogram']
    chain_slices, residue_count = chain_residue_slices(protein_complex), distogram.shape[0]
    coldspot_mask = expand_chain_residue_mask(residue_count, chain_slices[target], has_residue_flag(protein_complex[target].flags, ResidueFlags.COLDSPOT))
    binder_mask = binder_binding_mask(protein_complex, binder, residue_count)
    return nonbinding_residue_repulsion_loss(distogram, coldspot_mask, binder_mask, cutoff, contacts_per_residue)

def binder_binding_mask(protein_complex: dict[str, Protein[A]], binder: str, residue_count: int) -> A:
    xp = array_api_from_protein_complex(protein_complex)
    declared = binder_chain_flag_mask(protein_complex, binder, ResidueFlags.CONTACT, residue_count)
    return xp.where(declared.any(), declared, binder_chain_mask(protein_complex, binder, residue_count)) & (complex_residue_weights(protein_complex) > 0)

def binder_chain_mask(protein_complex: dict[str, Protein[A]], binder: str, residue_count: int) -> A:
    xp = array_api_from_protein_complex(protein_complex)
    chain_slices = chain_residue_slices(protein_complex)
    mask = xp.zeros(residue_count, dtype=xp.bool, device=array_device_from_protein_complex(protein_complex))
    for name in binder_copy_chains(protein_complex, binder):
        mask = mask | chain_residue_mask(residue_count, chain_slices[name], protein_complex)
    return mask

def binder_chain_flag_mask(protein_complex: dict[str, Protein[A]], binder: str, flag: ResidueFlags, residue_count: int) -> A:
    xp = array_api_from_protein_complex(protein_complex)
    chain_slices = chain_residue_slices(protein_complex)
    mask = xp.zeros(residue_count, dtype=xp.bool, device=array_device_from_protein_complex(protein_complex))
    for name in binder_copy_chains(protein_complex, binder):
        mask = mask | expand_chain_residue_mask(residue_count, chain_slices[name], has_residue_flag(protein_complex[name].flags, flag))
    return mask

def binder_framework_mask(protein_complex: dict[str, Protein], binder: str, residue_count: int, designed_only: bool=True) -> A:
    declared = binder_chain_flag_mask(protein_complex, binder, ResidueFlags.CONTACT, residue_count)
    occupied = binder_chain_flag_mask(protein_complex, binder, ResidueFlags.DESIGN, residue_count) if designed_only else binder_chain_mask(protein_complex, binder, residue_count)
    return occupied & ~declared & declared.any() & (complex_residue_weights(protein_complex) > 0)

def binder_repelled_mask(protein_complex: dict[str, Protein[A]], binder: str, residue_count: int, designed_only: bool=True) -> A:
    xp = array_api_from_protein_complex(protein_complex)
    coldspot = binder_chain_flag_mask(protein_complex, binder, ResidueFlags.COLDSPOT, residue_count)
    return xp.where(coldspot.any(), coldspot & (complex_residue_weights(protein_complex) > 0), binder_framework_mask(protein_complex, binder, residue_count, designed_only))

@loss('interface_contacts', target_weighting='binds_target')
def interface_contacts_loss(protein_states: ProteinStates[A], predictions: StructurePredictions[A], prediction_state: str='complex', binder: str='binder', target: str='target', cutoff: float=21.6875, contacts_per_residue: int=1, contact_residue_count: int | float=float('inf')) -> A:
    xp = array_api_from_protein_states(protein_states)
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    target = resolve_target_chain(protein_states[prediction_state], target, prediction_state)
    distogram = predictions[prediction_state].metrics['distogram']
    chain_slices, residue_count = chain_residue_slices(protein_states[prediction_state]), distogram.shape[0]
    real_mask = complex_residue_weights(protein_states[prediction_state]) > 0
    binder_mask, target_mask = binder_binding_mask(protein_states[prediction_state], binder, residue_count), chain_residue_mask(residue_count, chain_slices[target], protein_states[prediction_state]) & real_mask
    hotspot_mask = expand_chain_residue_mask(residue_count, chain_slices[target], has_residue_flag(protein_states[prediction_state][target].flags, ResidueFlags.HOTSPOT))
    pair_loss = distogram_pair_loss(distogram, distogram_bin_distances(distogram) < cutoff, False)
    hotspot_contacts = mean_selected_contact_loss(pair_loss, contacts_per_residue, contact_residue_count, hotspot_mask, hotspot_mask[:, None] & binder_mask[None, :])
    surface_contacts = mean_selected_contact_loss(pair_loss, contacts_per_residue, contact_residue_count, binder_mask, binder_mask[:, None] & target_mask[None, :])
    return xp.where(hotspot_mask.any(), hotspot_contacts, surface_contacts)

@loss('non_contact', target_weighting='avoids_target')
def non_contact_loss(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', binder: str='binder', target: str='target', cutoff: float=14.0, contacts_per_residue: int=2) -> A:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    target = resolve_target_chain(protein_states[prediction_state], target, prediction_state)
    chain_slices = chain_residue_slices(protein_states[prediction_state])
    distogram = predictions[prediction_state].metrics['distogram'][chain_slices[target], chain_slices[binder]]
    pair_loss = distogram_pair_loss(distogram, distogram_bin_distances(distogram) >= cutoff, False)
    target_mask, binder_mask = real_residue_mask(protein_states[prediction_state][target].flags), real_residue_mask(protein_states[prediction_state][binder].flags)
    return -_masked_mean(best_contact_mean(-pair_loss, contacts_per_residue, target_mask[:, None] & binder_mask[None, :]), target_mask)

@loss('binder_contacts')
def binder_contacts_loss(protein_states: ProteinStates[A], predictions: StructurePredictions[A], prediction_state: str='binder_alone', chain: str='binder', cutoff: float=14.0, contacts_per_residue: int=2, contact_residue_count: int | float=float('inf'), sequence_separation: int | None=9, per_protomer: bool=False) -> A:
    xp = array_api_from_protein_states(protein_states)
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    protein_complex = protein_states[prediction_state]
    binder_chains = binder_copy_chains(protein_complex, chain)
    binder_rows = chain_group_rows(protein_complex, binder_chains)
    distogram = predictions[prediction_state].metrics['distogram'][binder_rows[:, None], binder_rows[None, :]]
    residue_mask = xp.concat([real_residue_mask(protein_complex[name].flags) for name in binder_chains])
    pair_mask = residue_mask[:, None] & residue_mask[None, :]
    if sequence_separation is not None:
        pair_mask = pair_mask & ~block_diagonal(*[xp.abs(binder_sequence_offsets(protein_complex[name])) < sequence_separation for name in binder_chains])
    if per_protomer:
        pair_mask = pair_mask & (block_diagonal(*[xp.ones((len(protein_complex[name]),) * 2, device=array_device_from_protein_complex(protein_complex)) for name in binder_chains]) > 0)
    return distogram_contact_loss(distogram, cutoff, contacts_per_residue, contact_residue_count, residue_mask, pair_mask)

@loss('binder_helicity')
def binder_helicity_loss(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='binder_alone', chain: str='binder', cutoff: float=6.0) -> A:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    protein_complex = protein_states[prediction_state]
    binder_chains = binder_copy_chains(protein_complex, chain)
    binder_rows = chain_group_rows(protein_complex, binder_chains)
    distogram = predictions[prediction_state].metrics['distogram'][binder_rows[:, None], binder_rows[None, :]]
    return _masked_mean(distogram_pair_loss(distogram, distogram_bin_distances(distogram) < cutoff, True), block_diagonal(*[helical_pair_mask(protein_complex[name]) for name in binder_chains]))

@loss('target_helicity', target_weighting='every_target')
def target_helicity_loss(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', target: str='target', cutoff: float=6.0) -> A:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    return binder_helicity_loss(protein_states, predictions, prediction_state, resolve_target_chain(protein_states[prediction_state], target, prediction_state), cutoff)

@loss('non_helical')
def non_helical_loss(protein_states: ProteinStates[A], predictions: StructurePredictions[A], prediction_state: str='complex', chain: str='binder', cutoff: float=6.0) -> A:
    xp = array_api_from_protein_states(protein_states)
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    protein_complex = protein_states[prediction_state]
    binder_chains = binder_copy_chains(protein_complex, chain)
    binder_rows = chain_group_rows(protein_complex, binder_chains)
    distogram = xp.astype(predictions[prediction_state].metrics['distogram'][binder_rows[:, None], binder_rows[None, :]], xp.float32)
    close_probability = xp.sum(softmax(distogram) * xp.astype(distogram_bin_distances(distogram) < cutoff, distogram.dtype), axis=-1)
    return _masked_mean(close_probability, block_diagonal(*[helical_pair_mask(protein_complex[name]) for name in binder_chains]))

@loss('termini_distance')
def termini_distance_loss(protein_states: ProteinStates[A], predictions: StructurePredictions[A], prediction_state: str='complex', chain: str='binder', threshold_distance: float=7.0) -> A:
    xp = array_api_from_protein_states(protein_states)
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    coordinates, mask = chain_atom_coordinates(predictions[prediction_state].protein_complex[chain])
    c_terminus = real_residue_count(protein_states[prediction_state][chain].flags) - 1
    distance = xp.sqrt(xp.sum(xp.square(coordinates[0] - coordinates[c_terminus])) + 1e-08)
    return relu(distance - threshold_distance) * mask[0] * mask[c_terminus]

def terminus_target_direction_cosine(protein_states: ProteinStates[A], predictions: StructurePredictions[A], prediction_state: str, binder: str, target: str, terminus: str) -> tuple[A, A]:
    xp = array_api_from_protein_states(protein_states)
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    protein_complex = predictions[prediction_state].protein_complex
    binder_coordinates, binder_mask = chain_atom_coordinates(protein_complex[binder])
    target_coordinates, target_mask = chain_atom_coordinates(protein_complex[resolve_target_chain(protein_complex, target, prediction_state)])
    binder_center = masked_coordinate_centroid(binder_coordinates, binder_mask)
    c_terminus = real_residue_count(protein_complex[binder].flags) - 1
    terminus_position = (binder_coordinates[0] + binder_coordinates[c_terminus]) / 2 if terminus == 'both' else binder_coordinates[0 if terminus == 'n' else c_terminus]
    away_vector, terminus_vector = binder_center - masked_coordinate_centroid(target_coordinates, target_mask), terminus_position - binder_center
    denominator = xp.sqrt(xp.sum(xp.square(away_vector)) + 1e-08) * xp.sqrt(xp.sum(xp.square(terminus_vector)) + 1e-08)
    valid = (binder_mask.sum() > 0) * (target_mask.sum() > 0) * (binder_mask[0] * binder_mask[c_terminus] if terminus == 'both' else binder_mask[0 if terminus == 'n' else c_terminus])
    return xp.vecdot(away_vector, terminus_vector) / (denominator + 1e-08), valid

@loss('termini_angle', target_weighting='every_target')
def termini_angle_loss(protein_states: ProteinStates[A], predictions: StructurePredictions[A], prediction_state: str='complex', binder: str='binder', target: str='target') -> A:
    xp = array_api_from_protein_states(protein_states)
    cosine, valid = terminus_target_direction_cosine(protein_states, predictions, prediction_state, binder, target, 'both')
    return xp.acos(xp.clip(cosine, -1 + 1e-07, 1 - 1e-07)) * valid

@loss('n_terminus_away', target_weighting='every_target')
def n_terminus_away_loss(protein_states: ProteinStates[A], predictions: StructurePredictions[A], prediction_state: str='complex', binder: str='binder', target: str='target') -> A:
    cosine, valid = terminus_target_direction_cosine(protein_states, predictions, prediction_state, binder, target, 'n')
    return relu(-cosine) * valid

@loss('c_terminus_away', target_weighting='every_target')
def c_terminus_away_loss(protein_states: ProteinStates[A], predictions: StructurePredictions[A], prediction_state: str='complex', binder: str='binder', target: str='target') -> A:
    cosine, valid = terminus_target_direction_cosine(protein_states, predictions, prediction_state, binder, target, 'c')
    return relu(-cosine) * valid

def mutual_pair_choice(affinity: A, temperature: float) -> A:
    chosen = softmax(affinity / temperature, axis=-1)
    return chosen * chosen.T

@loss('disulfide')
def disulfide_loss(protein_states: ProteinStates[A], predictions: StructurePredictions[A], prediction_state: str='complex', chain: str='binder', distance: float=3.8, sigma: float=1.5, sequence_separation: int=3, temperature: float=0.1) -> A:
    xp = array_api_from_protein_states(protein_states)
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    protein_complex = protein_states[prediction_state]
    device = array_device_from_protein_complex(protein_complex)
    binder_chains = binder_copy_chains(protein_complex, chain)
    sequence = xp.concat([amino_acid_probabilities(protein_complex[name].sequence) for name in binder_chains])
    cysteine_probability = sequence[:, AMINO_ACIDS.index('C')]
    residue_mask = xp.concat([real_residue_mask(protein_complex[name].flags) for name in binder_chains])
    hardness = _masked_mean(xp.max(sequence, axis=-1), residue_mask)
    coordinates = xp.concat([chain_atom_coordinates(predictions[prediction_state].protein_complex[name], 'CB')[0] for name in binder_chains])
    bond_quality = xp.exp(-xp.square((pairwise_atom_distances(coordinates, coordinates, 1e-08) - distance) / sigma))
    chain_rows = lambda name: xp.arange(len(protein_complex[name]), device=device)
    sequence_local = block_diagonal(*[xp.abs(chain_rows(name)[:, None] - chain_rows(name)[None, :]) < sequence_separation for name in binder_chains])
    pairable = xp.astype(~sequence_local, xp.float32) * xp.astype(residue_mask[None, :], xp.float32) * xp.astype(residue_mask[:, None], xp.float32)
    partner_bond = bond_quality * stop_gradient(cysteine_probability)[None, :] * pairable
    satisfaction = xp.sum(partner_bond * mutual_pair_choice(partner_bond, temperature), axis=-1)
    return hardness * xp.sum(cysteine_probability * relu(1 - satisfaction) * xp.astype(residue_mask, cysteine_probability.dtype))

def align_binder_coordinates(coordinates: A, reference_coordinates: A, valid_mask: A) -> A:
    rotation, center, reference_center = kabsch(coordinates, reference_coordinates, valid_mask)
    return alignment_matrix_product(coordinates - center, stop_gradient(rotation).T) + reference_center

def aligned_binder_tm_score(coordinates: A, reference_coordinates: A, valid_mask: A, scored_length: A | float | None=None, eps: float=1e-08) -> A:
    xp = array_api_compat.array_namespace(coordinates)
    squared_distances = xp.sum(xp.square(align_binder_coordinates(coordinates, reference_coordinates, valid_mask) - reference_coordinates), axis=-1)
    length = xp.sum(valid_mask) if scored_length is None else scored_length
    distance_scale = xp.maximum(1.24 * xp.maximum(length - 15, xp.asarray(eps)) ** (1 / 3) - 1.8, xp.asarray(0.5))
    return _masked_mean(1 / (1 + (squared_distances + eps) / distance_scale ** 2), valid_mask)

def binder_state_confidence(prediction: StructurePrediction[A], chain: str) -> A:
    return prediction.metrics['plddt'][chain_residue_slices(prediction.protein_complex)[chain]]

def bound_and_unbound_binder_coordinates(predictions: StructurePredictions, prediction_state: str, reference_state: str, chain: str) -> tuple[A, A, A]:
    coordinates, mask = chain_atom_coordinates(predictions[prediction_state].protein_complex[chain])
    reference_coordinates, reference_mask = chain_atom_coordinates(predictions[reference_state].protein_complex[chain])
    return coordinates, reference_coordinates, mask * reference_mask

def confident_binder_comparison(predictions: StructurePredictions[A], prediction_state: str, reference_state: str, chain: str, confidence_floor: float=0.7) -> tuple[A, A, A]:
    coordinates, reference_coordinates, valid_mask = bound_and_unbound_binder_coordinates(predictions, prediction_state, reference_state, chain)
    xp = array_api_compat.array_namespace(coordinates)
    confidence = xp.minimum(binder_state_confidence(predictions[prediction_state], chain), binder_state_confidence(predictions[reference_state], chain))
    return coordinates, reference_coordinates, valid_mask * relu(confidence - confidence_floor) / (1.0 - confidence_floor + 1e-08)

def binder_interface_mask(protein_states: ProteinStates[A], predictions: StructurePredictions[A], chain: str='binder', target: str='target', cutoff: float=8.0) -> A:
    xp = array_api_from_protein_states(protein_states)
    prediction_state = resolve_prediction_state(predictions, 'complex')
    protein_complex = predictions[prediction_state].protein_complex
    coordinates, _ = chain_atom_coordinates(protein_complex[chain])
    target_coordinates, target_mask = chain_atom_coordinates(protein_complex[resolve_target_chain(protein_states[prediction_state], target, prediction_state)])
    facing_target = ((pairwise_atom_distances(coordinates, target_coordinates) < cutoff) * target_mask[None, :]).any(-1) & real_residue_mask(protein_states[prediction_state][chain].flags)
    return xp.astype(facing_target, xp.float32)

def interface_mask_residues(interface_mask: A | None) -> tuple[int, ...]:
    return () if interface_mask is None else tuple(int(residue) for residue in flat_nonzero(interface_mask))

def freeze_induced_fit_interface(losses: dict[str, DesignLoss], protein_states: ProteinStates, predictions: StructurePredictions) -> dict[str, DesignLoss]:
    frozen_losses = dict(losses)
    for name, entry in losses.items():
        if name.split('.')[0] != 'induced_fit_interface':
            continue
        chain = entry.function.keywords.get('chain', 'binder')
        interface_mask = binder_interface_mask(protein_states, predictions, chain, entry.function.keywords.get('target', 'target'), entry.function.keywords.get('cutoff', 8.0))
        interface_residue_count = int(interface_mask.sum())
        core_residues = int(real_residue_count(protein_states[resolve_prediction_state(predictions, 'complex')][chain].flags)) - interface_residue_count
        if interface_residue_count >= 3 and core_residues >= 3:
            frozen_losses[name] = DesignLoss(entry.function, entry.weight, entry.required_states, interface_mask)
    return frozen_losses

def frozen_interface_arguments(losses: dict[str, DesignLoss[A]], padded_chains: dict[str, Protein[A]] | None=None) -> dict[str, dict[str, A]]:
    frozen = {name: entry.interface_mask for name, entry in losses.items() if entry.interface_mask is not None}
    if padded_chains is None:
        return {name: {'interface_mask': interface_mask} for name, interface_mask in frozen.items()}
    return {name: {'interface_mask': xpx.pad(interface_mask, (0, len(padded_chains[losses[name].function.keywords.get('chain', 'binder')]) - len(interface_mask)))} for name, interface_mask in frozen.items()}

def induced_fit_hinge_names(losses: dict[str, DesignLoss]) -> tuple[str, ...]:
    return tuple(name for name in losses if name.split('.')[0] == 'induced_fit_interface')

BINDER_ALONE_LOSS_NAMES = ('plddt_loss', 'binder_pae', 'compactness', 'binder_contacts', 'binder_helicity')
PROTOMER_SCOPED_LOSSES = ('binder_pae', 'compactness', 'binder_contacts')

def induced_fit_binder_alone_losses(losses: dict[str, DesignLoss]) -> dict[str, DesignLoss]:
    binder_alone_losses = {}
    for name, entry in losses.items():
        if name.split('.')[0] in BINDER_ALONE_LOSS_NAMES:
            bound_loss, required_states = bind_state_metric(entry.function.func, {'params': {**entry.function.keywords, 'prediction_state': BINDER_ALONE}})
            binder_alone_losses[f'{name}.{BINDER_ALONE}'] = DesignLoss(bound_loss, entry.weight, required_states)
    return binder_alone_losses

def induced_fit_reference_losses(losses: dict[str, DesignLoss], reference_state: str, prediction_state: str | None=None) -> dict[str, DesignLoss]:
    reference_losses = dict(losses)
    for name in induced_fit_hinge_names(losses):
        entry = losses[name]
        state = prediction_state or entry.function.keywords.get('prediction_state', 'complex')
        parameters = {**entry.function.keywords, 'prediction_state': state, 'reference_state': reference_state}
        function = _bound_metric(entry.function.func, parameters)
        reference_losses[name] = DesignLoss(function, entry.weight, frozenset({state}), entry.interface_mask)
    return reference_losses

def binder_alone_design_losses(losses: dict[str, DesignLoss], reference_state: str) -> dict[str, DesignLoss]:
    losses = induced_fit_reference_losses(losses, reference_state, BINDER_ALONE)
    selected, hinge_names = {}, induced_fit_hinge_names(losses)
    for name, entry in losses.items():
        base_name = name.split('.')[0]
        if name in hinge_names:
            selected[name] = entry
        elif LOSS_TARGET_WEIGHTING.get(base_name) == 'binder_only':
            bound_loss, required_states = bind_state_metric(entry.function.func, {'params': entry.function.keywords, 'prediction_state': BINDER_ALONE})
            selected[base_name] = DesignLoss(bound_loss, entry.weight, required_states)
    return selected

def core_aligned_interface_rmsd(coordinates: A, reference_coordinates: A, interface_mask: A, alignment_mask: A, eps: float=1e-08) -> A:
    xp = array_api_compat.array_namespace(coordinates)
    aligned_coordinates = align_binder_coordinates(coordinates, reference_coordinates, alignment_mask)
    return xp.sqrt(_masked_mean(xp.sum(xp.square(aligned_coordinates - reference_coordinates), axis=-1), interface_mask) + eps)

def induced_fit_interface_masks(protein_states: ProteinStates[A], predictions: StructurePredictions[A], coordinates: A, valid_mask: A, prediction_state: str, target: str, cutoff: float, interface_residues: tuple[int, ...]=(), interface_mask: A | None=None) -> tuple[A, A]:
    xp = array_api_compat.array_namespace(coordinates)
    device = array_api_compat.device(coordinates)
    if interface_mask is None and interface_residues:
        empty = xp.zeros(len(coordinates), dtype=xp.float32, device=device)
        interface_mask = xpx.at(empty)[xp.asarray(interface_residues, device=device)].set(1.0)
    if interface_mask is None:
        target_coordinates, target_mask = chain_atom_coordinates(predictions[prediction_state].protein_complex[resolve_target_chain(protein_states[prediction_state], target, prediction_state)])
        interface_mask = stop_gradient(xp.astype(xp.any((pairwise_atom_distances(coordinates, target_coordinates) < cutoff) * (target_mask[None, :] > 0), axis=-1), xp.float32))
    interface_mask = interface_mask * valid_mask
    return interface_mask, (1 - interface_mask) * valid_mask

@loss('induced_fit_interface', target_weighting='every_target')
def induced_fit_interface_loss(protein_states: ProteinStates[A], predictions: StructurePredictions[A], prediction_state: str='complex', reference_state: str=BINDER_ALONE, chain: str='binder', target: str='target', interface_rmsd_target: float=3.0, cutoff: float=8.0, interface_mask: A | None=None, confidence_floor: float=0.7) -> A:
    xp = array_api_from_protein_states(protein_states)
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    if prediction_state not in predictions or reference_state not in predictions:
        return xp.asarray(0.0, device=array_device_from_protein_complex(protein_states[prediction_state]))
    coordinates, reference_coordinates, valid_mask = confident_binder_comparison(predictions, prediction_state, reference_state, chain, confidence_floor)
    interface_mask, alignment_mask = induced_fit_interface_masks(protein_states, predictions, coordinates, valid_mask, prediction_state, target, cutoff, interface_mask=interface_mask)
    interface_rmsd = core_aligned_interface_rmsd(coordinates, reference_coordinates, interface_mask, alignment_mask)
    return xp.square(relu(interface_rmsd_target - interface_rmsd)) * xp.astype(interface_mask.sum() >= 3, xp.float32) * xp.astype(alignment_mask.sum() >= 3, xp.float32)

@loss('fold_switching', target_weighting='binds_target')
def fold_switching_loss(protein_states: ProteinStates[A], predictions: StructurePredictions[A], prediction_state: str='complex', reference_state: str=BINDER_ALONE, chain: str='binder', tm_target: float=0.6, rmsd_target: float=1.0, confidence_floor: float=0.7) -> A:
    xp = array_api_from_protein_states(protein_states)
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    if prediction_state not in predictions or reference_state not in predictions:
        return xp.asarray(0.0, device=array_device_from_protein_complex(protein_states[prediction_state]))
    coordinates, reference_coordinates, valid_mask = confident_binder_comparison(predictions, prediction_state, reference_state, chain, confidence_floor)
    _, _, structural_mask = bound_and_unbound_binder_coordinates(predictions, prediction_state, reference_state, chain)
    separated = xp.square(relu(aligned_binder_tm_score(coordinates, reference_coordinates, valid_mask, structural_mask.sum()) - tm_target))
    moved = xp.square(relu(rmsd_target - core_aligned_interface_rmsd(coordinates, reference_coordinates, valid_mask, valid_mask)))
    return (separated + moved) * xp.astype(valid_mask.sum() >= 3, xp.float32)

def elastic_network_covariance(coordinates: A, residue_mask: A, contact_decay: float, damping: float, eps: float=1e-08) -> A:
    xp = array_api_compat.array_namespace(coordinates)
    device = array_api_compat.device(coordinates)
    residue_count = coordinates.shape[0]
    offsets = coordinates[None, :, :] - coordinates[:, None, :]
    squared_distances = xp.sum(xp.square(offsets), axis=-1)
    bond_directions = offsets / xp.sqrt(squared_distances + eps)[..., None]
    spring_constants = xp.where(xp.eye(residue_count, dtype=xp.bool, device=device), 0.0, xp.exp(-squared_distances / contact_decay ** 2) * residue_mask[:, None] * residue_mask[None, :])
    coupling_blocks = -spring_constants[..., None, None] * bond_directions[..., :, None] * bond_directions[..., None, :]
    anchored_blocks = xp.einsum('ijab->iab', -coupling_blocks) + (1 - residue_mask)[:, None, None] * PADDING_ANCHOR_STIFFNESS * xp.eye(3, device=device)
    hessian = xp.permute_dims(xpx.at(coupling_blocks)[xp.arange(residue_count, device=device), xp.arange(residue_count, device=device)].set(anchored_blocks), (0, 2, 1, 3)).reshape(3 * residue_count, 3*residue_count)


    centred_coordinates = (coordinates - masked_coordinate_centroid(coordinates, residue_mask)) * residue_mask[:, None]
    still = xp.zeros_like(residue_mask)
    rigid_body_motions = ((residue_mask, still, still), (still, residue_mask, still), (still, still, residue_mask), (still, -centred_coordinates[:, 2], centred_coordinates[:, 1]), (centred_coordinates[:, 2], still, -centred_coordinates[:, 0]), (-centred_coordinates[:, 1], centred_coordinates[:, 0], still))
    rigid_body_basis, _ = xp.linalg.qr(xp.stack([xp.reshape(xp.stack(motion, axis=-1), (-1,)) for motion in rigid_body_motions], axis=-1))
    internal_projector = xp.eye(3 * residue_count, device=device) - rigid_body_basis @ rigid_body_basis.T
    internal_hessian = internal_projector @ hessian @ internal_projector
    mean_curvature = xp.clip((xp.diagonal(internal_hessian) * xp.repeat(residue_mask, 3)).sum() / xp.clip(3 * xp.sum(residue_mask) - 6, 1.0, None), eps, None)
    return internal_projector @ xp.linalg.inv(internal_hessian + damping * mean_curvature * xp.eye(3 * residue_count, device=device)) @ internal_projector

@loss('collective_softness')
def collective_softness_loss(protein_states: ProteinStates[A], predictions: StructurePredictions[A], prediction_state: str='binder_alone', chain: str='binder', contact_decay: float=8.0, damping: float=0.01, power_iterations: int=12, eps: float=1e-08) -> A:
    xp = array_api_from_protein_states(protein_states)
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    binder_chains = binder_copy_chains(protein_states[prediction_state], chain)
    chain_coordinates = [chain_atom_coordinates(predictions[prediction_state].protein_complex[name]) for name in binder_chains]
    coordinates, residue_mask = xp.concat([coordinate for coordinate, _ in chain_coordinates]), xp.concat([mask for _, mask in chain_coordinates])
    covariance = elastic_network_covariance(coordinates, residue_mask, contact_decay, damping, eps)
    fluctuations = xp.clip(xp.einsum('iaia->i', xp.reshape(covariance, (len(coordinates), 3, len(coordinates), 3))), 0.0, None) * residue_mask
    fluctuation_share = fluctuations / (xp.sum(fluctuations) + eps)
    collectivity = xp.exp(-xp.sum(xp.where(fluctuation_share > 0, fluctuation_share * xp.log(fluctuation_share + eps), 0.0))) / (xp.sum(residue_mask) + eps)
    softest_mode = xp.reshape((coordinates - masked_coordinate_centroid(coordinates, residue_mask)) * residue_mask[:, None], (-1,))
    for _ in range(power_iterations):
        softest_mode = covariance @ softest_mode
        softest_mode = softest_mode / (xp.sqrt(xp.sum(xp.square(softest_mode))) + eps)
    concentration = xp.clip(softest_mode @ (covariance @ softest_mode) / (xp.sum(xp.linalg.diagonal(covariance) * xp.repeat(residue_mask, 3)) + eps), 0.0, 1.0)
    return 1 - xp.clip(collectivity, 0.0, 1.0) * concentration

def binder_domain_ids(binder_length: int, n_domains: int=2, min_domain_size: int=50, max_domain_size: int=180, seed: int=0) -> tuple[int, ...]:
    minimum_size = max(1, min_domain_size)
    domain_count = min(n_domains, binder_length // minimum_size)
    if domain_count < 2:
        return (0,) * binder_length
    random_generator, domain_sizes = random.Random(seed), [minimum_size] * domain_count
    for _ in range(binder_length - minimum_size * domain_count):
        extendable_domains = [index for index, domain_size in enumerate(domain_sizes) if domain_size < max(max_domain_size, minimum_size)]
        domain_sizes[random_generator.choice(extendable_domains) if extendable_domains else random_generator.randrange(domain_count)] += 1
    return tuple(domain_index for domain_index, domain_size in enumerate(domain_sizes) for _ in range(domain_size))

def domain_linker_membership(pae: A, domain_membership: A, gap: float, sharpness: float) -> A:
    xp = array_api_compat.array_namespace(pae)
    residue_indices = xp.arange(len(pae), device=array_api_compat.device(pae))
    separated_pairs = xp.astype(xp.abs(residue_indices[:, None] - residue_indices[None, :]) > 2, pae.dtype)
    counts = separated_pairs @ domain_membership
    pae_to_domain = pae * separated_pairs @ domain_membership / (counts + 1e-08)
    own_domain_pae = xp.sum(pae_to_domain * domain_membership, axis=-1)
    empty_domains = xp.astype(xp.sum(domain_membership, axis=0) < 0.5, pae.dtype)
    nearest_other_pae = xp.min(pae_to_domain + domain_membership * 1000000000.0 + empty_domains[None, :] * 1000000000.0, axis=-1)
    return sigmoid((gap - (nearest_other_pae - own_domain_pae)) / sharpness)

def multidomain_linker_residues(settings: dict, binder: Protein[A], binder_pae: A, seed: int=0) -> A:
    xp = array_api_compat.array_namespace(binder_pae)
    domain_ids = binder_domain_ids(len(binder), settings.get('n_domains', 2), settings.get('min_domain_size', 50), settings.get('max_domain_size', 180), seed)
    if max(domain_ids) < 1:
        return xp.zeros(len(binder), dtype=xp.bool, device=array_api_compat.device(binder_pae))
    domain_membership = binder_domain_membership(binder, domain_ids, max(settings.get('max_domains', 2), settings.get('n_domains', 2), max(domain_ids) + 1))
    linker_membership = domain_linker_membership(binder_pae / 31.0, domain_membership, float(settings.get('domain_linker_gap', 0.1)), float(settings.get('domain_linker_sharpness', 0.03)))
    return (linker_membership >= float(settings.get('domain_linker_fix_cut', 0.5))) & real_residue_mask(binder.flags)

@loss('multidomain')
def multidomain_loss(protein_states: ProteinStates[A], predictions: StructurePredictions[A], prediction_state: str='complex', chain: str='binder', domain_ids: tuple[int, ...]=(), n_domains: int=2, min_domain_size: int=50, max_domain_size: int=180, max_domains: int=2, seed: int=0, domain_rg_weight: float=0.3, domain_sep_weight: float=0.0, domain_contact_cutoff: float=8.0, domain_pae_margin: float=0.15, domain_linker_gap: float=0.1, domain_linker_sharpness: float=0.03, domain_linker_helix_weight: float=0.0) -> A:
    xp = array_api_from_protein_states(protein_states)
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    protein = protein_states[prediction_state][chain]
    residue_weights = real_residue_weights(protein.flags)
    domain_ids = domain_ids or binder_domain_ids(len(protein), n_domains, min_domain_size, max_domain_size, seed)
    domain_membership = binder_domain_membership(protein, domain_ids, max(max_domains, n_domains, max(domain_ids) + 1))
    same_domain = domain_membership @ domain_membership.T
    different_domain = (1 - same_domain) * residue_weights[:, None] * residue_weights[None, :]
    chain_slice = chain_residue_slices(protein_states[prediction_state])[chain]
    binder_pae = predictions[prediction_state].metrics['pae'][chain_slice, chain_slice] / 31.0
    sequence_hardness = _masked_mean(xp.max(amino_acid_probabilities(protein.sequence), axis=-1), residue_weights)
    linker_membership = domain_linker_membership(binder_pae, domain_membership, domain_linker_gap, domain_linker_sharpness) * residue_weights
    domain_residue_weights = (1 - sequence_hardness * linker_membership) * residue_weights
    domain_pair_weights = domain_residue_weights[:, None] * domain_residue_weights[None, :]
    domain_residue_weight = domain_membership.T @ domain_residue_weights
    coordinates, _ = chain_atom_coordinates(predictions[prediction_state].protein_complex[chain])
    domain_centroids = domain_membership.T @ (coordinates * domain_residue_weights[:, None]) / (domain_residue_weight[:, None] + 1e-08)
    squared_distances = xp.sum(xp.square(coordinates - domain_membership @ domain_centroids), axis=-1)
    domain_sizes = xp.sum(domain_membership, axis=0)
    domain_radii = xp.sqrt(domain_membership.T @ (squared_distances * domain_residue_weights) / (domain_residue_weight + 1e-08) + 1e-08)
    radius_loss = _masked_mean(elu(domain_radii - globular_radius(domain_sizes + 1e-08)), domain_sizes > 0.5)
    within_domain_pair_weights, between_domain_pair_weights = same_domain * domain_pair_weights, different_domain * domain_pair_weights
    within_domain_pae, between_domain_pae = _masked_mean(binder_pae, within_domain_pair_weights), _masked_mean(binder_pae, between_domain_pair_weights)
    domain_loss = within_domain_pae + relu(domain_pae_margin - (between_domain_pae - within_domain_pae)) + domain_rg_weight * radius_loss
    if domain_sep_weight:
        domain_loss += domain_sep_weight * _masked_mean(sigmoid((domain_contact_cutoff - pairwise_atom_distances(coordinates, coordinates, 1e-08)) / 2), between_domain_pair_weights)
    if domain_linker_helix_weight:
        distogram = predictions[prediction_state].metrics['distogram'][chain_slice, chain_slice]
        helix_loss = distogram_pair_loss(distogram, distogram_bin_distances(distogram) < 6, True)
        linker_pair_weights = xp.linalg.diagonal(linker_membership[:, None] * linker_membership[None, :], offset=3)
        domain_loss -= domain_linker_helix_weight * sequence_hardness * _masked_mean(xp.linalg.diagonal(helix_loss, offset=3), linker_pair_weights)
    return domain_loss * (different_domain.sum() > 0)

def build_design_losses(settings: dict, targets: dict[str, float], binder_length: int, seed: int=0) -> dict[str, DesignLoss]:
    from bindcraft.settings import load_settings, resolve_designed_binder_chain
    settings = load_settings(settings)
    binder_chain = resolve_designed_binder_chain(settings)
    domain_ids = binder_domain_ids(binder_length, settings.get('n_domains', 2), settings.get('min_domain_size', 50), settings.get('max_domain_size', 180), seed) if settings.get('weights_multidomain') else ()
    design_losses = {}
    for name, entry in build_losses(settings, seed).items():
        loss_function = REGISTERED_LOSSES[name]
        parameters = inspect.signature(loss_function).parameters
        loss_parameters = resolve_binder_role(loss_function, dict(entry.function.keywords), binder_chain)
        #multi-domain case
        if domain_ids and 'domain_ids' in parameters:
            loss_parameters['domain_ids'] = domain_ids
        if 'prediction_state' not in parameters:
            #fold switching case
            design_losses[name] = DesignLoss(_bound_metric(loss_function, loss_parameters), entry.weight, entry.required_states, entry.interface_mask)
            continue
        configured_state = loss_parameters.get('prediction_state', 'complex')
        reads_binder_only = configured_state == 'complex' and LOSS_TARGET_WEIGHTING[name] == 'binder_only'
        target_states = {'complex': 1.0} if reads_binder_only else targets if configured_state == 'complex' else {configured_state: targets.get(configured_state, 1.0)}
        for target_name, target_weight in target_states.items():
            target_loss_weight = {'binds_target': max(target_weight, 0), 'avoids_target': max(-target_weight, 0), 'every_target': 1.0, 'binder_only': 1.0}[LOSS_TARGET_WEIGHTING[name]]
            if not target_loss_weight:
                continue
            target_params = {**loss_parameters, 'prediction_state': target_name}
            if 'target' in parameters:
                target_params.setdefault('target', settings.get('target_chain', 'target'))
            required_states = set() if reads_binder_only else {target_name}
            if 'reference_state' in parameters:
                required_states.add(target_params.get('reference_state', parameters['reference_state'].default))
            loss_name = name if len(target_states) == 1 else f'{name}.{target_name}'
            bound_loss = _bound_metric(loss_function, target_params)
            design_losses[loss_name] = DesignLoss(bound_loss, entry.weight * target_loss_weight, frozenset(required_states))
    return design_losses
