#!/bin/bash
# Regenerate the convergence studies (docs/convergence.md): every study's levels through bin/tsafe, one process at
# a time, then the curves (docs/img/convergence/*.svg) and the tables of docs/convergence.md.
#
#   tools/convergence/run_all.sh [--root DIR] [--fresh] [study ...]
#
# Without study names every study runs (about one CPU hour on the i5-4690, half of it the whole-loop studies; the
# 2 ms macro step ran out of the 1.8 GB cap until the fix of docs/convergence.md section 9 item 10).
# --root DIR runs a frozen copy of the tree (cnctwin/, db/, examples/) with its own Numba cache (tools/convergence/
# env.py); compile it first with: bin/tsafe python tools/convergence/warm.py --root DIR. --fresh recomputes levels
# already in tools/convergence/results/. Deterministic: the same tree gives the same results, curves and tables.
set -uo pipefail
app=/opt/plint/apps/cnc-twin
here=$app/tools/convergence
args=()
studies=()
while [ $# -gt 0 ]; do
  case "$1" in
    --root) args+=(--root "$2"); shift 2 ;;
    --root=*) args+=("$1"); shift ;;
    --fresh) args+=(--fresh); shift ;;
    *) studies+=("$1"); shift ;;
  esac
done
if [ ${#studies[@]} -eq 0 ]; then
  studies=(stride_slot stride_side lattice_side lattice_corner element_flat element_ball bins_stride bins_runout
           bins_chatter turn_theta turn_dz lag thermal_cell thermal_every macro)
fi
failed=()
for s in "${studies[@]}"; do
  mem=1700M
  case "$s" in macro|lag|thermal_cell|thermal_every) mem=1800M ;; esac
  echo "== $s $(date +%H:%M:%S)"
  (cd "$app" && TSAFE_MEM=$mem bin/tsafe python "$here/study.py" "$s" "${args[@]}") || failed+=("$s")
done
(cd "$app" && bin/tsafe python "$here/plots.py") || failed+=(plots)
(cd "$app" && bin/tsafe python "$here/report.py") || failed+=(report)
if [ ${#failed[@]} -gt 0 ]; then echo "FAILED: ${failed[*]}"; exit 1; fi
echo "done $(date +%H:%M:%S)"
