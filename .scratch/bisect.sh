#!/bin/bash
# bisect.sh <maxs> variant... : resume each from r0's 4.3 s checkpoint with SW=variant
S=/opt/plint/.claude/worktrees/thinwall/.scratch
cd /opt/plint/.claude/worktrees/thinwall/apps/cnc-twin
maxs=$1; shift
for v in "$@"; do
  rm -rf $S/rb_$v; mkdir -p $S/rb_$v; cp -r $S/r0/* $S/rb_$v/
  rm -f $S/b_$v.steps
  TSAFE_MEM=1800M bin/tsafe env PYTHONPATH=$PWD SW=$v STEPLOG=$S/b_$v.steps STEPLOG_T0=4.4 python $S/run.py $S/rb_$v "{\"max_sim_s\": $maxs, \"resume_from\": \"$S/rb_$v/checkpoints/ck_4300000.json\", \"checkpoint_interval_s\": 1000, \"outputs\": false, \"sanity\": false}" sw > $S/b_$v.log 2>&1
  echo "== $v: $(grep -E 'TwinError|RESULT|Error' $S/b_$v.log | tail -1 | cut -c1-250)"; tail -1 $S/b_$v.steps | cut -c1-200
  rm -rf $S/rb_$v
done
