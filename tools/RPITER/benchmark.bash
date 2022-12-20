#!/usr/bin/env bash

# START OF CONFIGURATION

DATASETS=( 'rpi2241' 'rpi1807' 'rpi488' 'rpi369' )

# END OF CONFIGURATION

source env/bin/activate

cd RPITER
for (( i=0; i<${#DATASETS[@]}; i++ )); do
    for (( j=0; j<${#DATASETS[@]}; j++ )); do
        python rpiter.py -d ${DATASETS[$i]} -m ${DATASETS[$j]} 1> /dev/null
    done
    echo
    echo
done
