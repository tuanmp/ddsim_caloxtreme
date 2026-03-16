# ddsim_caloxtreme

## Simulation

1. Setup environment

Clone the repository
```
git clone https://github.com/tuanmp/ddsim_caloxtreme.git
cd ddsim_caloxtreme
```

The most convenient setup is to start from an LCG build. On a machine with `cvmfs`:
```
source /global/cfs/projectdirs/atlas/scripts/setupATLAS.sh # not necessary on lxplus. Need to change this line depending on the machine
setupATLAS -c el9
source lsetup.sh # setup a view from LCG_107
source build_all.sh
```

In a new shell
```
cd core_dependencies/OpenDataDetector
source acts-install/bin/this_acts.sh
source odd-install/bin/this_odd.sh
source acts-install/python/setup.sh
```


