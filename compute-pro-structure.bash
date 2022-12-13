#!/usr/bin/env bash

# START OF CONFIGURATION

INPUT=pro.fa
OUTPUT=pro-struct.fa
TMP=pro.tmp

# END OF CONFIGURATION

[ ! -f $INPUT ] && echo "Input file does not exist" && exit 1
[ -f $OUTPUT ] && rm -rf $OUTPUT

predator -a -f$TMP -bconst/stride.dat $INPUT
python3 utils/process_predator_output.py $TMP > $OUTPUT

rm -rf $TMP
