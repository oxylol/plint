#!/bin/bash
# usage: val.sh <before|after> <quantity>
cd /opt/plint/.claude/worktrees/softening/apps/cnc-twin
pre=""
[ "$1" = before ] && pre="sys.path.insert(0, '/opt/plint/.claude/worktrees/softening/.scratch/s'); import nosoft; "
TSAFE_MEM=1800M bin/tsafe env PYTHONPATH=$PWD python -c "import sys; ${pre}import cnctwin.validation as v; sys.exit(v.main(['--quantity','$2','--no-write','--json']))" > /opt/plint/.claude/worktrees/softening/.scratch/s/val_$1_$2.json 2> /opt/plint/.claude/worktrees/softening/.scratch/s/val_$1_$2.err
echo "$1 $2 exit $?"
