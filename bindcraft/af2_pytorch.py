import copy
import os

import torch

from bindcraft.af2 import alphafold_model_family
from bindcraft.loss import DesignLoss
from bindcraft.prediction import DifferentiableProteinPredictor
from bindcraft.protein import (A, AMINO_ACIDS, ATOM_NAMES, Protein, ProteinStates, ResidueFlags,
                               StructurePredictions, has_residue_flag, real_residue_mask)
from bindcraft.sequence_optimization import sequence_features_from_logits
from openfold.data import data_transforms
from openfold.model.model import AlphaFold
from openfold.config import model_config
from openfold.utils.import_weights import import_jax_weights_

def alphafold_npz_path(data_dir: str, model_name: str) -> str:
    for candidate in (f'params_{model_name}.npz', os.path.join('params', f'params_{model_name}.npz')):
        path = os.path.join(data_dir, candidate)
        if os.path.isfile(path):
            return path
    raise ValueError(f'no AlphaFold parameters for {model_name!r} in {data_dir!r}; '
                     'run "bindcraft fetch-weights" to download them')

import contextlib
import openfold.model.primitives as primitives

OPENFOLD_INITS = ('trunc_normal_init_', 'lecun_normal_init_', 'he_normal_init_',
                  'glorot_uniform_init_', 'final_init_', 'gating_init_', 'normal_init_',
                  'ipa_point_weights_init_')

@contextlib.contextmanager
def skip_openfold_init():
    """Build the module without initialising it: import_jax_weights_ writes every tensor."""
    saved = {name: getattr(primitives, name) for name in OPENFOLD_INITS if hasattr(primitives, name)}
    for name in saved:
        setattr(primitives, name, lambda *args, **kwargs: None)
    try:
        yield
    finally:
        for name, function in saved.items():
            setattr(primitives, name, function)

#one row of the alignment, one template, and openfold's 49 msa_feat channels: msa_1hot(23),
#has_deletion(1), deletion_value(1), cluster_profile(23), deletion_mean(1), which is the layout
#data_transforms_multimer.create_msa_feat builds and the one af2.py:191 writes into.
MSA_DEPTH, TEMPLATE_COUNT, MSA_FEATURE_CHANNELS = 1, 1, 49

def complex_residue_arrays(protein_complex: dict[str, Protein[A]]) -> dict[str, A]:
    chains = [protein_complex[name] for name in sorted(protein_complex)]
    device = chains[0].sequence.device
    return {'sequence': torch.cat([chain.sequence.float() for chain in chains]),
            'atoms': torch.cat([chain.atoms.float() for chain in chains]),
            'atom_mask': torch.cat([chain.atom_mask.float() for chain in chains]),
            'flags': torch.cat([chain.flags for chain in chains]),
            'residue_index': torch.cat([chain.residue_index.long() for chain in chains]),
            'asym_id': torch.cat([torch.full((len(chain),), index + 1, dtype=torch.long, device=device)
                                  for index, chain in enumerate(chains)])}

def openfold_input_features(protein_complex: dict[str, Protein[A]], num_recycle: int=1, softmax_weight: float=1.0,
                            one_hot_weight: float=0.0, temperature: float=1.0, logit_scale: float=2.0,
                            amino_acid_bias: A | None=None) -> dict[str, A]:
    """The dict openfold's multimer AlphaFold reads, from the arrays a design step already holds.

    The counterpart of af2.py's alphafold_input_features, and built the same way: no MSA search and no
    template search, because there is nothing to search for. The alignment is the designed sequence
    itself as a probability per position, which is the only reason a gradient reaches the sequence at
    all - target_feat and msa_feat are the two features it flows through, and a featuriser that emits
    an integer aatype cannot carry it. The one template is the target's own coordinates, which arrive
    in atom37 order on Protein.atoms, masked by the TEMPLATE residue flag.

    AlphaFold.forward reads num_iters off the last axis and then indexes t[..., cycle], so every
    feature carries one slot per cycle. They are expanded views of one slot: BindCraft does not
    resample features between cycles the way openfold's training pipeline does.
    """
    arrays = complex_residue_arrays(protein_complex)
    sequence, flags, asym_id = arrays['sequence'], arrays['flags'], arrays['asym_id']
    residues, device = sequence.shape[0], sequence.device
    scalar = lambda value: torch.as_tensor(value, dtype=sequence.dtype, device=device)
    designed = has_residue_flag(flags, ResidueFlags.DESIGN)[:, None]
    aatype = sequence.argmax(-1).long()
    residue_mask = real_residue_mask(flags).float()

    #the same two feature sets af2.py:102 prepares, from the same function: below softmax_weight the
    #linear path hands its logits over verbatim, so target_feat and the profile are not the same array
    sequence_features = torch.where(designed, sequence_features_from_logits(
        sequence, scalar(softmax_weight), scalar(one_hot_weight), scalar(temperature), scalar(logit_scale),
        amino_acid_bias), sequence)
    sequence_profile = torch.where(designed, sequence_features_from_logits(
        sequence, scalar(softmax_weight), scalar(1.0), scalar(temperature), scalar(logit_scale),
        amino_acid_bias), sequence)
    msa_feat = torch.zeros((MSA_DEPTH, residues, MSA_FEATURE_CHANNELS), dtype=sequence.dtype, device=device)
    msa_feat[0, :, 0:len(AMINO_ACIDS)] = sequence_features
    msa_feat[0, :, 25:25 + len(AMINO_ACIDS)] = sequence_profile
    #tf_dim is 21 for multimer and the designed sequence is 20 wide; af2's patched modules_multimer pads
    #the 21st column itself, openfold does not
    target_feat = torch.cat([sequence_features, torch.zeros((residues, 1), dtype=sequence.dtype, device=device)], dim=-1)

    #atom14/atom37 bookkeeping is a lookup on the argmax, so it moves whenever the design does
    atom_maps = data_transforms.make_atom14_masks({'aatype': aatype})
    template_residues = has_residue_flag(flags, ResidueFlags.TEMPLATE)
    template_atom_mask = arrays['atom_mask'] * template_residues[:, None].float()
    template_positions = torch.where(template_residues[:, None, None], arrays['atoms'], 0.0)
    expand = lambda array, *shape: array[None].expand(*shape)

    features = {'aatype': aatype, 'residue_index': arrays['residue_index'], 'asym_id': asym_id,
                'entity_id': asym_id.clone(), 'sym_id': torch.ones_like(asym_id),
                'seq_mask': residue_mask, 'seq_length': torch.full((), residues, dtype=torch.long, device=device),
                'target_feat': target_feat, 'msa_feat': msa_feat,
                'msa': expand(aatype, MSA_DEPTH, residues).contiguous(),
                'true_msa': expand(aatype, MSA_DEPTH, residues).contiguous(),
                'msa_mask': expand(residue_mask, MSA_DEPTH, residues).contiguous(),
                'msa_profile': torch.nn.functional.one_hot(aatype, 22).float(),
                'cluster_profile': expand(torch.nn.functional.one_hot(aatype, 23).float(), MSA_DEPTH, residues, 23),
                'num_alignments': torch.full((), MSA_DEPTH, dtype=torch.long, device=device),
                #no alignment means no deletions, nothing clustered and nothing masked. the extra-MSA
                #stack is given one empty row rather than none, so its shapes exist
                'deletion_matrix': torch.zeros((MSA_DEPTH, residues), device=device),
                'cluster_deletion_mean': torch.zeros((MSA_DEPTH, residues), device=device),
                'bert_mask': torch.zeros((MSA_DEPTH, residues), device=device),
                'extra_msa': torch.zeros((1, residues), dtype=torch.long, device=device),
                'extra_msa_mask': torch.zeros((1, residues), device=device),
                'extra_deletion_matrix': torch.zeros((1, residues), device=device),
                'use_clamped_fape': torch.zeros((), device=device),
                **{name: atom_maps[name] for name in ('atom14_atom_exists', 'atom37_atom_exists',
                                                      'residx_atom14_to_atom37', 'residx_atom37_to_atom14')},
                'template_aatype': expand(aatype, TEMPLATE_COUNT, residues).contiguous(),
                'template_all_atom_mask': expand(template_atom_mask, TEMPLATE_COUNT, residues, len(ATOM_NAMES)),
                'template_all_atom_positions': expand(template_positions, TEMPLATE_COUNT, residues, len(ATOM_NAMES), 3)}
    return {name: value.unsqueeze(-1).expand(*value.shape, num_recycle + 1) for name, value in features.items()}

class OpenFoldDesignModel(DifferentiableProteinPredictor):
    def __init__(self, models, data_dir, num_recycle, amino_acid_bias):
        self.data_dir = data_dir
        self.num_recycle = 1 if num_recycle is None else int(num_recycle)
        self.amino_acid_bias = amino_acid_bias
        # TODO: Respect amino_acid_bias

        self.model_families = dict()
        self.model_family_models = dict()
        self.runners = dict()
        self.model_parameters = dict()

        self.attention_backend = None
        self.last_attention_backend = None

        for model_name in models:
            model_family = alphafold_model_family(model_name)
            self.model_families[model_name] = model_family
            self.model_family_models.setdefault(model_family, model_name)
            model = self._openfold_runner(model_family)
            import_jax_weights_(model, alphafold_npz_path(data_dir, model_name), version=model_name)
            self.model_parameters[model_name] = {k: tensor.detach().clone() for k, tensor in model.state_dict().items()}

    def _openfold_runner(self, model_family):
        if model_family not in self.runners:
            config = model_config(self.model_family_models[model_family])
            with skip_openfold_init():
                self.runners[model_family] = AlphaFold(config)

        return self.runners[model_family]



    def predict(self, protein_states: ProteinStates, model: str | None=None) -> StructurePredictions:
        ...

    def sequence_gradients(self, protein_states: ProteinStates, losses: dict[str, DesignLoss], model: str | None=None, softmax_weight: float=1.0, one_hot_weight: float=0.0, temperature: float=1.0, logit_scale: float=2.0, reference_predictions: StructurePredictions | None=None, compile_only=False) -> tuple[StructurePredictions, dict[str, A], A]:
        for experiment, protein in protein_states.items():
            features = openfold_input_features(protein, self.num_recycle, softmax_weight, one_hot_weight, temperature, logit_scale, self.amino_acid_bias)

            model_family = self.model_families[model]
            runner = self._openfold_runner(model_family)

            out = runner(features)




