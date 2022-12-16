#!/usr/bin/env bash

# START OF CONFIGURATION

INPUT=pairs.txt
OUTPUT=negative-pairs.txt

# END OF CONFIGURATION

[[ ! -f $INPUT ]] && echo -e "\033[0;31mError:\033[0m input file does not exist" && exit 1

[[ -f $OUTPUT ]] && rm -rf $OUTPUT

touch $OUTPUT

POSITIVE_PAIRS=`wc -l $INPUT | awk '{print $1}'`

PROS=`awk '{print $1}' $INPUT | sort -u`
RNAS=`awk '{print $2}' $INPUT | sort -u`

NEGATIVE_PAIRS=0
while [[ "${NEGATIVE_PAIRS}" -lt "${POSITIVE_PAIRS}" ]]; do
    RANDOM_PRO=`shuf -n 1 <<< ${PROS}`
    RANDOM_RNA=`shuf -n 1 <<< ${RNAS}`

    RANDOM_INTERACTION="${RANDOM_PRO}\t${RANDOM_RNA}"

    if grep -q -P "${RANDOM_INTERACTION}" $INPUT || grep -q -P "${RANDOM_INTERACTION}" $OUTPUT; then
        continue
    fi
    echo -e "${RANDOM_INTERACTION}\t0" >> $OUTPUT

    NEGATIVE_PAIRS=$((NEGATIVE_PAIRS+1))
done
