"""Compare two parity dumps with a float32-appropriate tolerance."""
import json, sys, math
RTOL, ATOL = 2e-5, 1e-6
def close(x, y):
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        if math.isnan(x) and math.isnan(y): return True
        return abs(x - y) <= ATOL + RTOL * max(abs(x), abs(y))
    return x == y
def cmp(x, y, path=''):
    if isinstance(x, dict) and isinstance(y, dict):
        if set(x) != set(y): return [f'{path}: key mismatch']
        return [m for k in x for m in cmp(x[k], y[k], f'{path}.{k}')]
    if isinstance(x, list) and isinstance(y, list):
        if len(x) != len(y): return [f'{path}: length {len(x)} vs {len(y)}']
        return [m for i,(p,q) in enumerate(zip(x,y)) for m in cmp(p,q,f'{path}[{i}]')]
    return [] if close(x, y) else [f'{path}: {x} vs {y}']
a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
label = sys.argv[3] if len(sys.argv) > 3 else 'compare'
exact = [k for k in a if a[k] == b[k]]
within, outside = [], {}
for k in a:
    if k in exact: continue
    issues = cmp(a[k], b[k], k)
    (within if not issues else outside.setdefault(k, issues)) and None
    if issues: outside[k] = issues
    else: within.append(k)
print(f'{label}: {len(a)} cases | {len(exact)} exact | {len(within)} within float32 tol | {len(outside)} outside')
for k, issues in outside.items():
    print(f'    OUTSIDE {k}')
    for m in issues[:3]: print(f'        {m}')
sys.exit(1 if outside else 0)
