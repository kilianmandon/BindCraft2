#!/bin/bash
# Run the same campaign out of two trees, trajectory for trajectory, and leave the
# results where paired_stats.py can diff them.
#
#   BC_REF=/somewhere/ref OUT=/scratch/paired dev/parity/paired_campaigns.sh short hPDL1 hIL7RA
#
# One campaign of N trajectories costs one compile, and trajectory n draws
# jax.random.fold_in(campaign_key, n) in either tree, so the trajectories pair up by
# number. That is what makes 24 paired samples affordable: ~8 s each after the first.
#
# Every stage gate is set to 0.0, so the whole step schedule always runs and the two trees
# are compared on the metrics rather than on where a trajectory happened to be cut off.
set -u
SCHEDULE=${1:?usage: paired_campaigns.sh <short|long> <target>...}; shift
REF=${BC_REF:?set BC_REF to the reference worktree}
NEW=${BC_NEW:-$(cd "$(dirname "$0")/../.." && pwd)}
OUT=${OUT:?set OUT to a scratch folder}
export XLA_FLAGS=${XLA_FLAGS:---xla_gpu_autotune_level=0}   # trap 1 in dev/PORT-NOTES.md
mkdir -p "$OUT/configs" "$OUT/logs" "$OUT/runs"

case $SCHEDULE in
  short) STEPS='"screen_steps":4,"refine_steps":2,"anneal_steps":2,"harden_steps":1,"mutate_steps":1'; TRAJ=${TRAJ:-24} ;;
  long)  STEPS='"screen_steps":50,"refine_steps":25,"anneal_steps":20,"harden_steps":5,"mutate_steps":3'; TRAJ=${TRAJ:-4} ;;
  *) echo "unknown schedule $SCHEDULE" >&2; exit 2 ;;
esac
GATES='"min_plddt_screen":0.0,"min_plddt_refine":0.0,"min_plddt_anneal":0.0,"min_plddt_harden":0.0,"min_plddt_mutate":0.0,"min_iptm_anneal":0.0,"min_iptm_harden":0.0,"min_iptm_mutate":0.0'

for target in "$@"; do
  exp=${SCHEDULE}_${target}
  cat > "$OUT/configs/$exp.json" <<JSON
{"modality":"binder","target":"$target","campaign_name":"$exp","binder_lengths":[55,55],
 "max_trajectories":$TRAJ,"number_of_final_designs":1,"trajectory_only":true,
 $STEPS,$GATES}
JSON
  for tree in new head; do
    [ "$tree" = new ] && root=$NEW || root=$REF
    run=$OUT/runs/${exp}_${tree}
    [ -f "$run/summary.csv" ] && { echo "skip $exp $tree"; continue; }
    echo "=== $exp $tree $(date +%T)"
    # a compile cache per tree: trap 1's second half
    JAX_COMPILATION_CACHE_DIR=$OUT/cc_$tree BC_TREE=$root \
      "$NEW/.venv/bin/python" "$NEW/dev/parity/launch_raw.py" design "$OUT/configs/$exp.json" \
      --core benchmark --set project_folder="$run" > "$OUT/logs/${exp}_${tree}.log" 2>&1
    echo "    exit $? $(date +%T)"
  done
done
echo "ALLDONE $(date +%T)"
