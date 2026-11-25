"""Log the avals and device-commitment of every argument to the compiled gradient, per round."""
import sys, os
root = os.environ['BC_TREE']
sys.meta_path = [f for f in sys.meta_path
                 if 'editable' not in getattr(f, '__module__', '').lower()
                 and 'Editable' not in type(f).__name__]
sys.path.insert(0, root)
import bindcraft
assert bindcraft.__file__.startswith(root), bindcraft.__file__
import bindcraft.trajectory_output as to
to.recorded_row = lambda row: row
import bindcraft.af2 as af2

def describe(a):
    if isinstance(a, dict):
        return '{' + ','.join(f'{k}:{describe(v)}' for k, v in sorted(a.items())) + '}'
    if isinstance(a, (list, tuple)):
        return '[' + ','.join(describe(v) for v in a) + ']'
    if hasattr(a, 'shape') and hasattr(a, 'dtype'):
        wt = getattr(getattr(a, 'aval', None), 'weak_type', None)
        com = getattr(a, '_committed', None)
        return f'{a.dtype}{tuple(a.shape)}w{int(bool(wt))}c{int(bool(com))}'
    return type(a).__name__

orig = af2.AlphaFoldDesignModel._compiled_sequence_gradients
calls = {'n': 0, 'prev': None}
def patched(self, *a, **k):
    fn = orig(self, *a, **k)
    class Proxy:
        def __getattr__(self, name): return getattr(fn, name)
        def __call__(self, *args, **kwargs): return wrapper(*args, **kwargs)
    def wrapper(*args, **kwargs):
        calls['n'] += 1
        sig = [describe(x) for x in args]
        if calls['prev'] is not None and sig != calls['prev']:
            diffs = [(i, calls['prev'][i], sig[i]) for i in range(min(len(sig), len(calls['prev']))) if sig[i] != calls['prev'][i]]
            print(f'AVALS CHANGED at call {calls["n"]}: {diffs}', flush=True)
        calls['prev'] = sig
        print(f'AVALS call {calls["n"]}: {" ".join(sig)[:600]}', flush=True)
        return fn(*args, **kwargs)
    return Proxy()
af2.AlphaFoldDesignModel._compiled_sequence_gradients = patched
from bindcraft.cli import main
sys.exit(main())
