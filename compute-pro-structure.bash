#!/usr/bin/env bash

# START OF CONFIGURATION

INPUT=pro.fa
OUTPUT=pro-struct.fa
TMP=pro.tmp

# END OF CONFIGURATION

[[ ! -f $INPUT ]] && echo -e "\033[0;31mError:\033[0m input file does not exist" && exit 1
[[ -f $OUTPUT ]] && rm -rf $OUTPUT
[[ -f $TMP ]] && rm -rf $TMP

predator -a -f$TMP -bconst/stride.dat $INPUT
python3 utils/process_predator_output.py $TMP > $OUTPUT

[[ -f $TMP ]] && rm -rf $TMP
