"""Call the loss helpers that build their own arrays with torch CUDA tensors, and check
that what comes back is on the card.

    BC_TREE=/workspace/BindCraft2 .venv/bin/python dev/parity/torch_device.py

Nothing else in this folder can catch a missing `device=`: under JAX an array creation with
no device lands on the default device, which is the GPU, so the tree is silent about it.
Under torch it lands on the host, and then either the op raises ("found at least two
devices") or — worse — torch quietly copies, as it does for a CPU index tensor into a CUDA
tensor, and the cost hides inside the step. Run this against the reference tree to see a
regression fail: pre-fix, two of the three raise and `chain_group_rows` returns `cpu`.
"""
import os, sys
_ROOT = os.environ.get('BC_TREE')
if _ROOT:
    sys.meta_path = [f for f in sys.meta_path
                     if 'editable' not in getattr(f, '__module__', '').lower()
                     and 'Editable' not in type(f).__name__]
    sys.path.insert(0, _ROOT)
import torch
from bindcraft.protein import Protein, ResidueFlags
import bindcraft.loss as L

DEVICE = os.environ.get('BC_TORCH_DEVICE', 'cuda')

def protein(length: int, designed: bool) -> Protein:
    flags = torch.full((length,), int(ResidueFlags.SEQUENCE | ResidueFlags.TEMPLATE), dtype=torch.uint8, device=DEVICE)
    flags[:] = int(ResidueFlags.DESIGN) if designed else flags[0]
    flags[-2:] = int(ResidueFlags.PADDING)
    generator = torch.Generator(device=DEVICE).manual_seed(0)
    return Protein(sequence=torch.randn((length, 20), generator=generator, device=DEVICE, dtype=torch.float16),
                   atoms=torch.randn((length, 37, 3), generator=generator, device=DEVICE, dtype=torch.float16) * 8,
                   atom_mask=torch.rand((length, 37), generator=generator, device=DEVICE) > 0.2,
                   flags=flags,
                   residue_index=torch.arange(1, length + 1, dtype=torch.int32, device=DEVICE))

binder, target = protein(14, True), protein(11, False)
protein_complex = {'binder': binder, 'target': target}
coordinates = binder.atoms[:, 1].to(torch.float32)
residue_weights = L.real_residue_weights(binder.flags).to(torch.float32)

CASES = {
 'chain_group_rows':          lambda: L.chain_group_rows(protein_complex, ('binder', 'target')),
 'binder_domain_membership':  lambda: L.binder_domain_membership(binder, (0, 0, 0, 1, 1, 1), 2),
 'elastic_network_covariance': lambda: L.elastic_network_covariance(coordinates, residue_weights, 8.0, 0.01),
}
failed = 0
for name, case in CASES.items():
    try:
        result = case()
    except Exception as error:
        print(f'{name:28s} RAISED {type(error).__name__}: {str(error)[:90]}')
        failed += 1
        continue
    on_device = result.device.type == torch.device(DEVICE).type
    print(f'{name:28s} {"ok  " if on_device else "HOST"} device={result.device} dtype={result.dtype}')
    failed += not on_device

gradient_input = coordinates.clone().requires_grad_(True)
try:
    L.elastic_network_covariance(gradient_input, residue_weights, 8.0, 0.01).sum().backward()
    on_device = gradient_input.grad.device.type == torch.device(DEVICE).type
    print(f'{"elastic gradient":28s} {"ok  " if on_device else "HOST"} device={gradient_input.grad.device}')
    failed += not on_device
except Exception as error:
    print(f'{"elastic gradient":28s} RAISED {type(error).__name__}: {str(error)[:90]}')
    failed += 1
print(f'{failed} of {len(CASES) + 1} checks off the device' if failed else 'all on the device')
sys.exit(1 if failed else 0)
