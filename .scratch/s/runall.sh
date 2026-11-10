#!/bin/bash
for q in cutting_force drilling_thrust drilling_torque shear_angle tool_temperature chip_temperature coolant_reach taylor_life chip_ratio; do
  for m in after before; do
    [ -s val_${m}_${q}.json ] && [ "$q" = chip_ratio ] && [ "$m" = after ] && continue
    /opt/plint/.claude/worktrees/softening/.scratch/s/val.sh $m $q
  done
done
echo ALLDONE
