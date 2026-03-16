#!/bin/bash

# lsetup "views LCG_107 x86_64-el9-gcc13-opt" # setup a view from LCG_107
mkdir core_dependencies
cd core_dependencies
git clone https://gitlab.cern.ch/acts/OpenDataDetector.git
cd OpenDataDetector
git clone --depth 1 https://github.com/acts-project/acts.git --branch v39.2.1

set -e
set -u

source_dir=$PWD/acts
install_dir=$PWD/acts-install
build_dir=$PWD/acts-build

cmake -S $source_dir -B $build_dir -GNinja \
-DCMAKE_BUILD_TYPE=Release  \
-DCMAKE_INSTALL_PREFIX=$install_dir \
-DACTS_BUILD_EXAMPLES=ON \
-DACTS_BUILD_EXAMPLES_DD4HEP=ON \
-DACTS_BUILD_EXAMPLES_GEANT4=ON \
-DACTS_BUILD_PLUGIN_DD4HEP=ON \
-DACTS_BUILD_ANALYSIS_APPS=ON \
-DACTS_BUILD_EXAMPLES_PYTHON_BINDINGS=ON

cmake --build $build_dir -- -j16

cmake --install $build_dir

cmake -S . -B odd-build -GNinja -DCMAKE_INSTALL_PREFIX=$PWD/odd-install 
cmake --build odd-build -- -j3
cmake --install odd-build