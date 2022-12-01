#!/usr/bin/env bash

INPUT=npinter-v5.txt
NONCODEIDS=ids-noncode-v5.txt

RNA_OUTPUT=rna.fa
PROTEIN_OUTPUT=protein.fa
INTERACTS_OUTPUT=protein-interacts.txt

TMP_FILE=tmp_file.txt
TMP2_FILE=tmp2_file.txt

[ ! -f $INPUT ] && exit 1
[ ! -f $NONCODEIDS ] && exit 1

[ -f $RNA_OUTPUT ] && rm -rf $RNA_OUTPUT
[ -f $PROTEIN_OUTPUTS ] && rm -rf $PROTEIN_OUTPUT
[ -f $INTERACTS_OUTPUT ] && rm -rf $INTERACTS_OUTPUT

[ -f $TMP_FILE ] && rm -rf $TMP_FILE
[ -f $TMP2_FILE ] && rm -rf $TMP2_FILE

mkdir -p cache

declare -A RNAS
declare -A PROTEINS

declare -A ALLOWED_ORGANISMS=( ["Homo sapiens"]=0 ["Mus musculus"]=0 ["Drosophila melanogaster"]=0 )

find_protein_id () {
    PROTID=$1
    ORGANISM=$2

    [ "${PROTID}" == "-" ] && echo ">>Skipping protein with ID: -" && return 1
    [[ "${PROTID}" == *"heterodimer"* ]] && echo ">>Skipping protein with ID: ${PROTID} because it is a heterodimer" && return 1

    PROTID_LINK=$(sed -e "s/ /%20/g" <<< $PROTID)
    [ ! -f "cache/${PROTID}.txt" ] && curl -s -H "Accept: text/plain; format=tsv" "https://rest.uniprot.org/uniprotkb/search?query=${PROTID_LINK}" --output "cache/${PROTID}.txt" || echo "Using cached protein ID: ${PROTID}"

    NUM_LINES=`wc -l cache/${PROTID}.txt | awk '{print $1}'`
    if [ $NUM_LINES -gt 1 ]
    then
        FOUND=0
        while IFS= read -r LINE
        do
            IFS=$'\t' read -r -a WWORDS <<< $LINE
            TMP_PROTID="${WWORDS[0]}"
            TMP_ORGANISM="${WWORDS[5]}"
            if grep -q "${ORGANISM}" <<< $TMP_ORGANISM; then
                PROTID="${TMP_PROTID}"
                FOUND=1
                break
            fi
        done < "cache/${PROTID}.txt"

        if [ $FOUND -eq 0 ]
        then
            echo ">>Cannot find protein with ID: ${PROTID} and ORGANISM: ${ORGANISM}"
            return 1
        fi
    else
        echo ">>Cannot find protein with ID: ${PROTID} and ORGANISM: ${ORGANISM}"
        return 1
    fi

    echo "!!Found protein with ID: ${PROTID} and ORGANISM: ${ORGANISM}"

    return 0
}

find_lncRNA_id () {
    NONCODEID=$1
    ORGANISM=$2

    [ "${NONCODEID}" == "-" ] && echo ">>Skipping RNA with ID: -" && return 1

    grep -i -w "${NONCODEID}" ${NONCODEIDS} > $TMP_FILE
    [ ! -s $TMP_FILE ] && echo ">>Cannot find RNA with ID: ${NONCODEID}" && return 1

    while IFS= read -r LINE
    do
        IFS=$'\t' read -r -a WWORDS <<< "$LINE"
        TMP_NAME="${WWORDS[0]}"
        IFS=$'.' read -r -a WWWORDS <<< "${TMP_NAME}"
        [ ! -f "cache/${TMP_NAME}.html" ] && curl -s "http://www.noncode.org/show_rna.php?id=${WWWORDS[0]}&version=${WWWORDS[1]}" --output "cache/${TMP_NAME}.html"
        python3 parser.py "${TMP_NAME}.html" > $TMP2_FILE
        if grep -q "${ORGANISM}" $TMP2_FILE; then
            NONCODEID="${TMP_NAME}"
            if [ ! "${RNAS[$NONCODEID]+abc}" ]
            then
                echo ">${NONCODEID}" >> $RNA_OUTPUT
                sed -n '2{p;q}' $TMP2_FILE >> $RNA_OUTPUT
            fi
            echo "!!Found RNA with ID: ${NONCODEID} and ORGANISM: ${ORGANISM}"
            return 0
        fi
    done < $TMP_FILE

    echo ">>Cannot find RNA with ID: ${NONCODEID} and ORGANISM: ${ORGANISM}"

    return 1
}

# find valid lncRNA-protein pairs
while IFS= read -r LINE
do
    IFS=$'\t' read -r -a WORDS <<< $LINE
    if [ "${WORDS[3]}" = "lncRNA" ] && [ "${WORDS[6]}" = "protein" ]
    then
        PROTID="${WORDS[5]}"
        NONCODEID="${WORDS[2]}"
        ORGANISM="${WORDS[10]}" && [[ ${#WORDS[@]} -lt 16 ]] && ORGANISM="${WORDS[9]}"

        [ ! ${ALLOWED_ORGANISMS[$ORGANISM]+abc} ] && continue

        find_lncRNA_id "${NONCODEID}" "${ORGANISM}"
        if [ $? -gt 0 ]
        then
            NONCODEID="${WORDS[1]}"
            echo "Retrying with RNA ID: ${NONCODEID}"
            find_lncRNA_id "${NONCODEID}" "${ORGANISM}"
            [ $? -gt 0 ] && continue
        fi

        find_protein_id "${PROTID}" "${ORGANISM}"
        if [ $? -gt 0 ]
        then
            PROTID="${WORDS[4]}"
            echo "Retrying with protein ID: ${PROTID}"
            find_protein_id "${PROTID}" "${ORGANISM}"
            [ $? -gt 0 ] && continue
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
        echo -e "\033[0;31mFatal error: cannot download protein with ID: ${PROTID} and ORGANISM: ${PROTEINS[$PROTID]}\033[0m"
    else
        echo ">${PROTID}" >> $PROTEIN_OUTPUT
        sed 1d $TMP_FILE >> $PROTEIN_OUTPUT
    fi
done

[ -f $TMP_FILE ] && rm -rf $TMP_FILE
[ -f $TMP2_FILE ] && rm -rf $TMP2_FILE
