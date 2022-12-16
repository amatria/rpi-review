#!/usr/bin/env bash

# START OF CONFIGURATION

INPUT=rna.fa
OUTPUT=rna-struct.fa
TMP=rna.tmp

# END OF CONFIGURATION

[[ ! -f $INPUT ]] && echo -e "\033[0;31mError:\033[0m input file does not exist" && exit 1

[[ -f $OUTPUT ]] && rm -rf $OUTPUT
[[ -f $TMP ]] && rm -rf $TMP

RNAsubopt -p 1 --infile=$INPUT --outfile=$TMP

while IFS= read -r LINE; do
    if [[ "${LINE:0:1}" == ">" ]]; then
        echo $LINE >> $OUTPUT
    elif [[ "${LINE:0:1}" == "(" ]] || [[ "${LINE:0:1}" == ")" ]] || [[ "${LINE:0:1}" == "." ]]; then
        for CHUNK in $LINE; do
            echo $CHUNK >> $OUTPUT
            break
        done
    fi
done < $TMP

[[ -f $TMP ]] && rm -rf $TMP
