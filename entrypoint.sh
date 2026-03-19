#!/bin/bash
# Source the LCG environment

export LCG_VERSION=107
export LCG_PLATFORM=el9
. /cvmfs/sft.cern.ch/lcg/views/LCG_${LCG_VERSION}/x86_64-${LCG_PLATFORM}-gcc13-opt/setup.sh
cd core_dependencies/OpenDataDetector
source acts-install/bin/this_acts.sh 
source odd-install/bin/this_odd.sh 
source acts-install/python/setup.sh
cd ../../
# Run any script passed as argument, or drop to shell
if [ -n "$1" ]; then
    exec "$@"
else
    exec /bin/bash
fi