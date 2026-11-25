"""Run the CLI from BC_TREE with the losses.csv rounding disabled, so values keep full precision."""
import sys, os
root = os.environ['BC_TREE']
sys.meta_path = [f for f in sys.meta_path
                 if 'editable' not in getattr(f, '__module__', '').lower()
                 and 'Editable' not in type(f).__name__]
sys.path.insert(0, root)
import bindcraft
assert bindcraft.__file__.startswith(root), bindcraft.__file__
import bindcraft.trajectory_output as to
to.recorded_row = lambda row: row          # keep every digit
print(f'raw-precision run from {bindcraft.__file__}', flush=True)
from bindcraft.cli import main
sys.exit(main())
