#!/usr/bin/env bash

# START OF CONFIGURATION

DATASETS=( 'rpi2241' 'rpi1807' 'rpi488' 'rpi369' 'npinter-v5' )

# END OF CONFIGURATION

source env/bin/activate

for (( i=0; i<${#DATASETS[@]}; i++ )); do
    python RPITER/rpiter.py -d ${DATASETS[$i]} -m ${DATASETS[$i]} 1> /dev/null
done
echo
for (( i=0; i<${#DATASETS[@]}-1; i++ )); do
    python RPITER/rpiter.py -d npinter-v5 -m ${DATASETS[$i]} -p 1> /dev/null
done
