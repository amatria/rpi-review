#!/usr/bin/env bash

INPUT=my-npinter-v5.txt
PROTEIN_OUTPUT=protein.fa
INTERACTS_OUTPUT=protein-interacts.txt

TMP_FILE=tmp_file.txt

[ ! -f $INPUT ] && exit 1
[ -f $PROTEIN_OUTPUTS ] && rm -rf $PROTEIN_OUTPUT
[ -f $INTERACTS_OUTPUT ] && rm -rf $INTERACTS_OUTPUT

[ -f $TMP_FILE ] && rm -rf $TMP_FILE

declare -A RNAS
declare -A PROTEINS

find_protein_id () {
    PROTID=$1
    ORGANISM=$2

    if [ "${PROTID}" == "-" ]
    then
        echo "Skipping protein with ID: -"
        return 1
    fi

    curl -s -H "Accept: text/plain; format=tsv" "https://rest.uniprot.org/uniprotkb/search?query=${PROTID}" --output $TMP_FILE

    NUM_LINES=`wc -l ${TMP_FILE} | awk '{print $1}'`
    if [ ${NUM_LINES} -gt 1 ]
    then
        FOUND=0
        while IFS= read -r LINE
        do
            IFS=$'\t' read -r -a WWORDS <<< "$LINE"
            TMP_ORGANISM="${WWORDS[5]}"
            if grep -q "${ORGANISM}" <<< $TMP_ORGANISM; then
                PROTID="${WWORDS[0]}"
                FOUND=1
                break
            fi
        done < $TMP_FILE

        if [ $FOUND -eq 0 ]
        then
            echo "Cannot find protein with ID: ${PROTID} and ORGANISM: ${ORGANISM}"
            return 1
        fi
    else
        echo "Cannot find protein with ID: ${PROTID}"
        return 1
    fi

    return 0
}

# find valid lncRNA-protein pairs
while IFS= read -r LINE
do
    IFS=$'\t' read -r -a WORDS <<< "$LINE"
    if [ "${WORDS[3]}" = "lncRNA" ] && [ "${WORDS[6]}" = "protein" ]
    then
        PROTID="${WORDS[5]}"
        NONCODEID="${WORDS[2]}"
        ORGANISM=${WORDS[10]} && [[ ${#WORDS[@]} -lt 16 ]] && ORGANISM=${WORDS[9]}

        if ! grep -q "NON" <<< $NONCODEID; then
            if ! grep -q "ENS" <<< $NONCODEID; then
                continue
            fi
        fi

        if [ ! "${PROTEINS[$PROTID]+abc}" ]
        then
            find_protein_id $PROTID "${ORGANISM}"
            if [ $? -gt 0 ]
            then
                PROTID=${WORDS[4]}
                echo " -> Retrying with ID: ${PROTID}"
                find_protein_id $PROTID "${ORGANISM}"
                [ $? -gt 0 ] && continue
            fi
        fi

        echo "${PROTID} ${NONCODEID} ${ORGANISM}" >> $INTERACTS_OUTPUT

        PROTEINS[$PROTID]=$ORGANISM
        RNAS[$NONCODEID]=$ORGANISM
    fi
done < $INPUT

# download protein sequences
for PROTID in "${!PROTEINS[@]}"
do
    curl -s -H "Accept: text/plain; format=fasta" "https://rest.uniprot.org/uniprotkb/${PROTID}" --output $TMP_FILE
    if grep -q "Error messages" $TMP_FILE; then
        echo "Cannot download protein with ID: ${PROTID} and ORGANISM: ${PROTEINS[$PROTID]}"
    else
        cat $TMP_FILE >> $PROTEIN_OUTPUT
    fi
done

[ -f $TMP_FILE ] && rm -rf $TMP_FILE
