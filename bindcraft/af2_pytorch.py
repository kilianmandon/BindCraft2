import os

import jax
import numpy as np
import torch
import jax.numpy as jnp

import contextlib
import openfold.model.primitives as primitives

from bindcraft.af2 import alphafold_model_family
from bindcraft.loss import DesignLoss
from bindcraft.prediction import DifferentiableProteinPredictor
from bindcraft.protein import (
    A,
    AMINO_ACIDS,
    ATOM_INDEX,
    ATOM_NAMES,
    Protein,
    ProteinStates,
    ResidueFlags,
    StructurePrediction,
    StructurePredictions,
    has_residue_flag,
    kabsch,
    real_residue_mask,
)
from bindcraft.sequence_optimization import sequence_features_from_logits
from openfold.data import data_transforms
from openfold.model.model import AlphaFold
from openfold.config import model_config
from openfold.utils.import_weights import import_jax_weights_
from openfold.model.dropout import Dropout as OpenFoldDropout
from openfold.utils.loss import compute_tm


def alphafold_npz_path(data_dir: str, model_name: str) -> str:
    for candidate in (
        f"params_{model_name}.npz",
        os.path.join("params", f"params_{model_name}.npz"),
    ):
        path = os.path.join(data_dir, candidate)
        if os.path.isfile(path):
            return path
    raise ValueError(
        f"no AlphaFold parameters for {model_name!r} in {data_dir!r}; "
        'run "bindcraft fetch-weights" to download them'
    )



OPENFOLD_INITS = (
    "trunc_normal_init_",
    "lecun_normal_init_",
    "he_normal_init_",
    "glorot_uniform_init_",
    "final_init_",
    "gating_init_",
    "normal_init_",
    "ipa_point_weights_init_",
)


@contextlib.contextmanager
def skip_openfold_init():
    """Build the module without initialising it: import_jax_weights_ writes every tensor."""
    saved = {
        name: getattr(primitives, name)
        for name in OPENFOLD_INITS
        if hasattr(primitives, name)
    }
    for name in saved:
        setattr(primitives, name, lambda *args, **kwargs: None)
    try:
        yield
    finally:
        for name, function in saved.items():
            setattr(primitives, name, function)


# one row of the alignment, one template, and openfold's 49 msa_feat channels: msa_1hot(23),
# has_deletion(1), deletion_value(1), cluster_profile(23), deletion_mean(1), which is the layout
# data_transforms_multimer.create_msa_feat builds and the one af2.py:191 writes into.
MSA_DEPTH, TEMPLATE_COUNT, MSA_FEATURE_CHANNELS = 1, 1, 49


def complex_residue_arrays(protein_complex: dict[str, Protein[A]]) -> dict[str, A]:
    chains = [protein_complex[name] for name in sorted(protein_complex)]
    device = chains[0].sequence.device
    return {
        "sequence": torch.cat([chain.sequence.float() for chain in chains]),
        "atoms": torch.cat([chain.atoms.float() for chain in chains]),
        "atom_mask": torch.cat([chain.atom_mask.float() for chain in chains]),
        "flags": torch.cat([chain.flags for chain in chains]),
        "residue_index": torch.cat([chain.residue_index.long() for chain in chains]),
        "asym_id": torch.cat(
            [
                torch.full((len(chain),), index + 1, dtype=torch.long, device=device)
                for index, chain in enumerate(chains)
            ]
        ),
    }


def align_prediction_to_target_template(
    predicted_atom_positions: A,
    predicted_atom_mask: A,
    template_atom_positions: A,
    template_atom_mask: A,
    flags: A,
) -> A:
    """The torch counterpart of af2.py:88, which af2 applies on both of its paths (af2.py:435
    in predict, af2.py:480 in the gradient path) and which this port was missing.

    AlphaFold emits the complex in an arbitrary frame. af2 superposes it onto the target's own
    template coordinates using the target residues (TEMPLATE and not DESIGN) as the alignment
    set, so the prediction comes back in the frame the caller supplied the target in. Measured
    without it, of2's centroid sat at [0.75, 1.73, 2.00] against af2's [21.53, -54.01, -32.77]
    for the same input, while the superposed CA RMSD was 0.019A: a pure rigid offset.

    Rigid motion leaves distances alone, so this does not move contact or radius-of-gyration
    losses. It matters for everything that reads the coordinates in the target's frame: written
    structures, the MPNN stage, validation, and any template-relative loss. The final `where`
    also zeroes atoms that do not exist, which af2 relies on downstream.

    kabsch comes from protein.py and is array-api generic, so it runs on torch unchanged. The
    rotation is built from detached coordinates, matching af2's jax.lax.stop_gradient: the
    gradient flows through the positions being moved, never through the frame they move into.
    """
    ca_atom_index = ATOM_INDEX["CA"]
    template_residue_mask = has_residue_flag(
        flags, ResidueFlags.TEMPLATE
    ) & ~has_residue_flag(flags, ResidueFlags.DESIGN)
    alignment_weights = (
        template_residue_mask.bool() & template_atom_mask[:, ca_atom_index].bool()
    ).float()
    predicted_ca_coordinates = (
        predicted_atom_positions[:, ca_atom_index].detach().float()
    )
    template_ca_coordinates = template_atom_positions[:, ca_atom_index].float()
    alignment_rotation, prediction_center, template_center = kabsch(
        predicted_ca_coordinates, template_ca_coordinates, alignment_weights
    )
    aligned_atom_positions = (
        predicted_atom_positions.float() - prediction_center
    ) @ alignment_rotation.T + template_center
    return torch.where(
        predicted_atom_mask[:, :, None].bool(), aligned_atom_positions, 0.0
    ).to(predicted_atom_positions.dtype)


def openfold_input_features(
    protein_complex: dict[str, Protein[A]],
    num_recycle: int = 1,
    softmax_weight: float = 1.0,
    one_hot_weight: float = 0.0,
    temperature: float = 1.0,
    logit_scale: float = 2.0,
    amino_acid_bias: A | None = None,
) -> dict[str, A]:
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
    sequence, flags, asym_id = arrays["sequence"], arrays["flags"], arrays["asym_id"]
    residues, device = sequence.shape[0], sequence.device
    scalar = lambda value: torch.as_tensor(np.asarray(value), dtype=sequence.dtype, device=device)
    designed = has_residue_flag(flags, ResidueFlags.DESIGN)[:, None]
    aatype = sequence.argmax(-1).long()
    residue_mask = real_residue_mask(flags).float()

    # the same two feature sets af2.py:102 prepares, from the same function: below softmax_weight the
    # linear path hands its logits over verbatim, so target_feat and the profile are not the same array
    sequence_features = torch.where(
        designed,
        sequence_features_from_logits(
            sequence,
            scalar(softmax_weight),
            scalar(one_hot_weight),
            scalar(temperature),
            scalar(logit_scale),
            amino_acid_bias,
        ),
        sequence,
    )
    sequence_profile = torch.where(
        designed,
        sequence_features_from_logits(
            sequence,
            scalar(softmax_weight),
            scalar(1.0),
            scalar(temperature),
            scalar(logit_scale),
            amino_acid_bias,
        ),
        sequence,
    )
    msa_feat = torch.zeros(
        (MSA_DEPTH, residues, MSA_FEATURE_CHANNELS), dtype=sequence.dtype, device=device
    )
    msa_feat[0, :, 0 : len(AMINO_ACIDS)] = sequence_features
    msa_feat[0, :, 25 : 25 + len(AMINO_ACIDS)] = sequence_profile
    # tf_dim is 21 for multimer and the designed sequence is 20 wide; af2's patched modules_multimer pads
    # the 21st column itself, openfold does not
    target_feat = torch.cat(
        [
            sequence_features,
            torch.zeros((residues, 1), dtype=sequence.dtype, device=device),
        ],
        dim=-1,
    )

    # atom14/atom37 bookkeeping is a lookup on the argmax, so it moves whenever the design does
    atom_maps = data_transforms.make_atom14_masks({"aatype": aatype})
    template_residues = has_residue_flag(flags, ResidueFlags.TEMPLATE)
    template_atom_mask = arrays["atom_mask"] * template_residues[:, None].float()
    template_positions = torch.where(
        template_residues[:, None, None], arrays["atoms"], 0.0
    )
    expand = lambda array, *shape: array[None].expand(*shape)

    features = {
        "aatype": aatype,
        "residue_index": arrays["residue_index"],
        "asym_id": asym_id,
        "entity_id": asym_id.clone(),
        "sym_id": torch.ones_like(asym_id),
        "seq_mask": residue_mask,
        "seq_length": torch.full((), residues, dtype=torch.long, device=device),
        "target_feat": target_feat,
        "msa_feat": msa_feat,
        "msa": expand(aatype, MSA_DEPTH, residues).contiguous(),
        "true_msa": expand(aatype, MSA_DEPTH, residues).contiguous(),
        "msa_mask": expand(residue_mask, MSA_DEPTH, residues).contiguous(),
        "msa_profile": torch.nn.functional.one_hot(aatype, 22).float(),
        "cluster_profile": expand(
            torch.nn.functional.one_hot(aatype, 23).float(), MSA_DEPTH, residues, 23
        ),
        "num_alignments": torch.full((), MSA_DEPTH, dtype=torch.long, device=device),
        # no alignment means no deletions, nothing clustered and nothing masked. the extra-MSA
        # stack is given one empty row rather than none, so its shapes exist
        "deletion_matrix": torch.zeros((MSA_DEPTH, residues), device=device),
        "cluster_deletion_mean": torch.zeros((MSA_DEPTH, residues), device=device),
        "bert_mask": torch.zeros((MSA_DEPTH, residues), device=device),
        "extra_msa": torch.zeros((1, residues), dtype=torch.long, device=device),
        "extra_msa_mask": torch.zeros((1, residues), device=device),
        "extra_deletion_matrix": torch.zeros((1, residues), device=device),
        "use_clamped_fape": torch.zeros((), device=device),
        **{
            name: atom_maps[name]
            for name in (
                "atom14_atom_exists",
                "atom37_atom_exists",
                "residx_atom14_to_atom37",
                "residx_atom37_to_atom14",
            )
        },
        "template_aatype": expand(aatype, TEMPLATE_COUNT, residues).contiguous(),
        "template_all_atom_mask": expand(
            template_atom_mask, TEMPLATE_COUNT, residues, len(ATOM_NAMES)
        ),
        "template_all_atom_positions": expand(
            template_positions, TEMPLATE_COUNT, residues, len(ATOM_NAMES), 3
        ),
    }
    return {
        name: value.unsqueeze(-1).expand(*value.shape, num_recycle + 1)
        for name, value in features.items()
    }


class OpenFoldDesignModel(DifferentiableProteinPredictor):
    def __init__(self, models, data_dir, num_recycle, amino_acid_bias, device="cuda",
                 key=None, dropout=True):
        self.device = device
        self.data_dir = data_dir
        self.num_recycle = 1 if num_recycle is None else int(num_recycle)
        self.dropout = dropout
        #af2.py:307 holds a PRNGKey and splits it; openfold's Dropout reads torch's global
        #RNG and takes no generator (dropout.py has no generator argument), so the key seeds
        #a Generator we own and advance ourselves, one draw per forward. Kept on the host so
        #it is device-independent: it only ever produces python ints.
        self.generator = torch.Generator()
        self.key = key
        # the same per-amino-acid bias vector af2.py:314 builds, as a torch tensor
        self.amino_acid_bias = None if not amino_acid_bias else torch.as_tensor(
            [amino_acid_bias.get(amino_acid, 0.0) for amino_acid in AMINO_ACIDS],
            dtype=torch.float32, device=self.device)

        self.models = models
        self.model_families = dict()
        self.model_family_models = dict()
        self.runners = dict()
        self.model_parameters = dict()

        self.attention_backend = None
        self.last_attention_backend = None
        self.last_used_model = None

        for model_name in models:
            model_family = alphafold_model_family(model_name)
            self.model_families[model_name] = model_family
            self.model_family_models.setdefault(model_family, model_name)
            model = self._openfold_runner(model_family).to(device=self.device)
            model.eval()
            self._set_dropout(model, self.dropout)

            import_jax_weights_(
                model, alphafold_npz_path(data_dir, model_name), version=model_name
            )
            self.model_parameters[model_name] = {
                k: tensor.detach().clone().to(device=self.device) for k, tensor in model.state_dict().items()
            }

    def _openfold_runner(self, model_family):
        if model_family not in self.runners:
            config = model_config(self.model_family_models[model_family])
            # sequence_gradients needs the backward graph, and openfold gates activation
            # checkpointing on torch.is_grad_enabled() alone (evoformer.py:1053), so the
            # eval-mode default of blocks_per_ckpt=None would retain every activation.
            config.globals.blocks_per_ckpt = 1
            # openfold's multimer preset early-stops recycling once CA distances move less
            # than 0.5A (config.py:947, model.py:204), while af2.py:197 always runs every
            # cycle. Left on, of2 silently does fewer recycles than af2 on the same design.
            # It is also the data-dependent branch that defeats whole-model torch.compile,
            # and its .item() forces a host sync per cycle.
            config.model.recycle_early_stop_tolerance = -1.0
            with skip_openfold_init():
                self.runners[model_family] = AlphaFold(config)

        return self.runners[model_family]

    @property
    def key(self):
        return self._key

    @key.setter
    def key(self, key) -> None:
        """campaign.py:172 assigns a fresh jax key per trajectory, the same way it does to
        AlphaFoldDesignModel. Reseed our Generator from it, or the per-trajectory key is
        silently ignored and every trajectory reuses the same dropout stream."""
        self._key = key
        self.generator.manual_seed(0 if key is None else int(jax.random.randint(key, (), 0, 2 ** 31 - 1)))

    def _sample_design_model(self) -> str:
        return self.models[int(torch.randint(0, len(self.models), (), generator=self.generator))]

    def _set_dropout(self, model, enabled: bool) -> None:
        """Flip only the Dropout modules, not the whole module tree.

        model.train() would also arm the `assert(not self.training)` guards on openfold's
        offload paths (model.py:922,953,1193), and those are unrelated to dropout.
        """
        for module in model.modules():
            if isinstance(module, (OpenFoldDropout, torch.nn.Dropout)):
                module.train(enabled)

    @contextlib.contextmanager
    def _forked_rng(self):
        """Draw the forward's dropout masks from our own stream, leaving the global RNG as it was."""
        devices = [torch.device(self.device)] if torch.device(self.device).type == 'cuda' else []
        step_seed = int(torch.randint(0, 2 ** 31 - 1, (), generator=self.generator))
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(step_seed)
            yield

    def _forward_with_metrics(
        self,
        protein_states: ProteinStates,
        model: str | None = None,
        softmax_weight: float = 1.0,
        one_hot_weight: float = 0.0,
        temperature: float = 1.0,
        logit_scale: float = 2.0,
    ):

        protein_states, chain_seqs = self.prep_protein_states(protein_states)
        structure_predictions = {}

        if model is None:
            model = self._sample_design_model()
        model_family = self.model_families[model]

        runner = self._openfold_runner(model_family)

        if model != self.last_used_model:
            runner.load_state_dict(self.model_parameters[model])
            self.last_used_model = model

        for state_name, protein_complex in protein_states.items():
            features = openfold_input_features(
                protein_complex,
                self.num_recycle,
                softmax_weight,
                one_hot_weight,
                temperature,
                logit_scale,
                self.amino_acid_bias,
            )

            with self._forked_rng():
                out = runner(features)

            # af2.py:480 aligns the whole complex at once, before splitting it per chain, so the
            # alignment set spans every target chain. Do the same here rather than per chain.
            input_arrays = complex_residue_arrays(protein_complex)
            aligned_atom_positions = align_prediction_to_target_template(
                out["final_atom_positions"],
                out["final_atom_mask"],
                input_arrays["atoms"],
                input_arrays["atom_mask"],
                input_arrays["flags"],
            )

            out_protein_complex = {}
            residue_offset = 0
            for chain in sorted(protein_complex):
                protein = protein_complex[chain]
                n_res = protein.atoms.shape[0]
                atoms = aligned_atom_positions[residue_offset:residue_offset+n_res]
                atom_mask = out["final_atom_mask"][residue_offset:residue_offset+n_res]
                residue_offset += n_res
                
                out_protein_complex[chain] = Protein(
                    sequence=chain_seqs[chain],
                    atoms=atoms,
                    atom_mask=atom_mask,
                    flags=protein.flags,
                    residue_index=protein.residue_index,
                )

            interface_asym_ids = features["asym_id"][..., 0].clone()
            res_offset = 0
            all_binders = all(k=='binder' or k.startswith('binder_') for k in protein_complex)
            for chain in sorted(protein_complex):
                n_res = protein_complex[chain].sequence.shape[0]
                if not all_binders and (chain=='binder' or chain.startswith('binder_')):
                    interface_asym_ids[res_offset:res_offset+n_res] = 1000
                res_offset += n_res

            metrics = self.openfold_prediction_metrics(out, features["seq_mask"][..., 0], interface_asym_ids)

            structure_predictions[state_name] = StructurePrediction(
                out_protein_complex, metrics
            )

        return structure_predictions, protein_states, chain_seqs

    def predict(
        self, protein_states: ProteinStates, model: str | None = None
    ) -> StructurePredictions:
        with torch.no_grad():
            structure_predictions, _, _ = self._forward_with_metrics(
                protein_states, model, one_hot_weight=1.0
            )


        structure_predictions = nested_map(structure_predictions, torch_to_jax)

        return structure_predictions
    

    def prep_protein_states(self, protein_states: ProteinStates):
        chain_seqs = {}
        out_protein_states = {}

        jax_to_torch = lambda x: torch.as_tensor(np.array(x, copy=True), device=self.device)

        for state_name, protein_state in protein_states.items():
            out_protein_states[state_name] = {}
            for chain, protein in protein_state.items():
                if chain not in chain_seqs:
                    chain_seqs[chain] = jax_to_torch(protein.sequence)
                    chain_seqs[chain].requires_grad_(True)

                out_protein_states[state_name][chain] = nested_map(protein, jax_to_torch)

                out_protein_states[state_name][chain].sequence = chain_seqs[chain]

        return out_protein_states, chain_seqs

    def sequence_gradients(
        self,
        protein_states: ProteinStates,
        losses: dict[str, DesignLoss],
        model: str | None = None,
        softmax_weight: float = 1.0,
        one_hot_weight: float = 0.0,
        temperature: float = 1.0,
        logit_scale: float = 2.0,
        reference_predictions: StructurePredictions | None = None,
        compile_only=False,
    ) -> tuple[StructurePredictions, dict[str, A], A]:

        if compile_only:
            # TODO: Think whether i actually want to compile
            return {}, {}, jnp.array(0.0)

        # TODO: Use reference_predictions (dont know how they should be used yet)


        structure_predictions, protein_states, chain_seqs = (
            self._forward_with_metrics(
                protein_states,
                model,
                softmax_weight,
                one_hot_weight,
                temperature,
                logit_scale,
            )
        )

        weighted_losses = {
            name: entry.weight * entry.function(protein_states, structure_predictions)
            for name, entry in losses.items()
        }
        total_loss = sum(
            weighted_losses.values(), torch.tensor(0.0, device=self.device)
        )
        total_loss.backward()

        # af2.py:502 records each weighted term on the state's metrics, which is what the
        # trajectory log reads its per-loss columns from. Without this of2 campaigns logged a
        # total and nothing else, and the two backends could not be compared term by term.
        # The overwrite guard is af2.py:503's, kept so a loss named after a metric fails loudly
        # on this backend too rather than silently replacing the model's own value.
        for name, entry in losses.items():
            for state_name in (
                entry.required_states or structure_predictions.keys()
            ) & structure_predictions.keys():
                metrics = structure_predictions[state_name].metrics
                if name in metrics:
                    raise ValueError(
                        f"loss {name!r} would overwrite an existing metric on {state_name!r}"
                    )
                metrics[name] = weighted_losses[name].detach()

        sequence_gradients = {
            chain: seq.grad if seq.grad is not None else torch.zeros_like(seq)
            for chain, seq in chain_seqs.items()
        }

        structure_predictions = nested_map(structure_predictions, torch_to_jax)
        sequence_gradients = nested_map(sequence_gradients, torch_to_jax)
        total_loss = nested_map(total_loss, torch_to_jax)

        return structure_predictions, sequence_gradients, total_loss

    def openfold_prediction_metrics(self, openfold_outputs: dict, seq_mask: A, interface_asym_id: A) -> dict[str, A]:
        metrics: dict[str, torch.Tensor] = {}
        if "plddt" in openfold_outputs:
            metrics["plddt"] = openfold_outputs["plddt"] / 100
        if "predicted_aligned_error" in openfold_outputs:
            pae = openfold_outputs["predicted_aligned_error"]
            metrics["pae"] = (pae + pae.T) / 2
            
            metrics["ptm"] = compute_tm(openfold_outputs["tm_logits"], residue_weights=seq_mask)
            metrics["iptm"], metrics["iptm_per_residue"] = compute_tm(openfold_outputs["tm_logits"], residue_weights=seq_mask, asym_id=interface_asym_id,
                                         interface=True, return_per_alignment=True)
        # af2.py:221's metric. loss.py:228 (the 'experimentally_resolved' loss) indexes it, so
        # leaving it out made that loss a KeyError on this backend. openfold's AuxiliaryHeads
        # exposes the same logits under a flat name (heads.py:72).
        if "experimentally_resolved_logits" in openfold_outputs:
            metrics["experimentally_resolved_ca"] = torch.sigmoid(
                openfold_outputs["experimentally_resolved_logits"][:, ATOM_INDEX["CA"]]
            )
        if "distogram_logits" in openfold_outputs:
            metrics["distogram"] = openfold_outputs["distogram_logits"]

        return metrics


def torch_to_jax(array: torch.Tensor):
    """numpy has no bfloat16, so widen only that dtype.

    Casting unconditionally would turn flags (uint8) and residue_index (int32) into
    float32, and has_residue_flag's `flags & flag` (protein.py:61) rejects floats.
    """
    if array.dtype == torch.bfloat16:
        array = array.float()
    return jnp.asarray(array.detach().cpu().numpy())


def nested_map(obj, f):
    if isinstance(obj, torch.Tensor) or isinstance(obj, jax.Array):
        return f(obj)

    if isinstance(obj, dict):
        return {
            k: nested_map(v, f) for k,v in obj.items()
        }

    if isinstance(obj, StructurePrediction):
        return StructurePrediction(
            protein_complex=nested_map(obj.protein_complex, f),
            metrics=nested_map(obj.metrics, f),
        )

    if isinstance(obj, Protein):
        return Protein(
            sequence=nested_map(obj.sequence, f),
            atoms=nested_map(obj.atoms, f),
            atom_mask=nested_map(obj.atom_mask, f),
            flags=nested_map(obj.flags, f),
            residue_index=nested_map(obj.residue_index, f),
        )

    raise ValueError(f'Mapping for object {type(obj)} is not implemented.')

    