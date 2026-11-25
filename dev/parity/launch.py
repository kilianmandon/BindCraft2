"""Run the bindcraft CLI out of the tree named by BC_TREE, past the editable install's finder."""
import sys, os
root = os.environ['BC_TREE']
sys.meta_path = [f for f in sys.meta_path
                 if 'editable' not in getattr(f, '__module__', '').lower()
                 and 'Editable' not in type(f).__name__]
sys.path.insert(0, root)
import bindcraft
assert bindcraft.__file__.startswith(root), f'wrong tree: {bindcraft.__file__}'
print(f'running bindcraft from {bindcraft.__file__}', flush=True)
from bindcraft.cli import main
sys.exit(main())
