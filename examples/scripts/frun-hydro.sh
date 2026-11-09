#!/bin/bash
source ../../builds/setup.frontier.cce.sh
module -t list
set -x
EXE=$(readlink -f ../../bin/cholla.hydro.frontier)
ldd $EXE
PARAMFILE=$(readlink -f ../$2)
OUTDIR="run/$2.$(date +%m%d.%H%M%S)"
mkdir -p ${OUTDIR}
cd ${OUTDIR}
NODES=$1
TASKS=$(( 8 * NODES ))
echo $NODES $TASKS
srun -t5:00 -n1 -c7 -N1 --exclusive $EXE $PARAMFILE |& tee tee
