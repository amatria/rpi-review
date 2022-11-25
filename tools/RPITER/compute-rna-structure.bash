#!/bin/bash

INPUT=prueba.fa
OUTPUT=prueba.out
TMP=prueba.tmp

[ ! -f $INPUT ] && echo "Input file does not exist" && exit 1
[ -f $OUTPUT ] && rm -rf $OUTPUT

RNAfold --jobs=16 --noPS --outfile=$TMP $INPUT

while IFS= read -r LINE
do
    if [ "${LINE:0:1}" == ">" ]
    then
        echo $LINE >> $OUTPUT
    elif [ "${LINE:0:1}" == "(" ] || [ "${LINE:0:1}" == ")" ] || [ "${LINE:0:1}" == "." ]
    then
        for CHUNK in $LINE
        do
            echo $CHUNK >> $OUTPUT
            break
        done
    fi
done < $TMP

rm -rf $TMP
