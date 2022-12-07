#!/usr/bin/env bash

INPUT=npinter-v5.txt
NONCODEIDS=ids-noncode-v5.txt

# end of configuration

RNA_OUTPUT=rna.fa
PROTEIN_OUTPUT=protein.fa
INTERACTS_OUTPUT=protein-rna-pairs.txt

CACHE=cache

TMP_FILE=/var/tmp/tmp_file.txt
TMP2_FILE=/var/tmp/tmp2_file.txt

[[ ! -f $INPUT ]] || [[ ! -f $NONCODEIDS ]] && echo -e "\033[0;31mFatal error:\033[0m missing input files" && exit 1

[[ -f $RNA_OUTPUT ]] && rm -rf $RNA_OUTPUT
[[ -f $PROTEIN_OUTPUT ]] && rm -rf $PROTEIN_OUTPUT
[[ -f $INTERACTS_OUTPUT ]] && rm -rf $INTERACTS_OUTPUT

mkdir -p $CACHE

touch $RNA_OUTPUT
touch $PROTEIN_OUTPUT
touch $INTERACTS_OUTPUT

declare -A ALLOWED_ORGANISMS=( ["Homo sapiens"]=0 ["Mus musculus"]=0 ["Drosophila melanogaster"]=0 )

find_protein () {
    [[ "${PROTID}" == "-" ]] || \
    [[ "${PROTID}" == *"heterodimer"* ]] || \
    [[ "${PROTID}" == *"#"* ]] || \
    [[ "${PROTID}" == *[![:ascii:]]* ]] && \
    return 1

    local PROTID_LINK=$(sed -e "s/ /%20/g" <<< "${PROTID}")
    [[ ! -f "${CACHE}/'${PROTID}'.txt" ]] && curl -s -H "Accept: text/plain; format=tsv" "https://rest.uniprot.org/uniprotkb/search?query=${PROTID_LINK}" --output "${CACHE}/'${PROTID}'.txt"

    local NUM_LINES=`wc -l "${CACHE}/'${PROTID}'.txt" | awk '{print $1}'`
    [[ "${NUM_LINES}" -lt "2" ]] && return 1

    local LINE
    local WORDS
    while IFS= read -r LINE
    do
        IFS=$'\t' read -r -a WORDS <<< $LINE
        local TMP_PROTID="${WORDS[0]}"
        local TMP_ORGANISM="${WORDS[5]}"
        if grep -q "${ORGANISM}" <<< "${TMP_ORGANISM}"; then
            PROTID="${TMP_PROTID}"
            return 0
        fi
    done < "${CACHE}/'${PROTID}'.txt"

    return 1
}

find_lncRNA () {
    [[ "${NONCODEID}" == "-" ]] && return 1

    grep -i -w "${NONCODEID}" $NONCODEIDS > $TMP_FILE
    [[ ! -s $TMP_FILE ]] && return 1

    local LINE
    local WORDS
    local NAMES
    while IFS= read -r LINE; do
        IFS=$'\t' read -r -a WORDS <<< $LINE
        local TMP_NAME="${WORDS[0]}"
        IFS=$'.' read -r -a NAMES <<< "${TMP_NAME}"

        [[ ! -f "${CACHE}/${TMP_NAME}.html" ]] && curl -s "http://www.noncode.org/show_rna.php?id=${NAMES[0]}&version=${NAMES[1]}" --output "${CACHE}/${TMP_NAME}.html"
        python3 noncode_html_parser.py "${CACHE}/${TMP_NAME}.html" > $TMP2_FILE

        if grep -q "${ORGANISM}" $TMP2_FILE; then
            NONCODEID="${TMP_NAME}"
            return 0
        fi
    done < $TMP_FILE

    return 1
}

check_duplicate () {
    if grep -q -P "${PROTID}\t${NONCODEID}\t${ORGANISM}" $INTERACTS_OUTPUT; then
        return 1
    fi
    return 0
}

download_protein () {
    if grep -q "${PROTID}" $INTERACTS_OUTPUT; then
        return 0
    fi

    [[ ! -f "${CACHE}/'${PROTID}'.fa" ]] && curl -s -H "Accept: text/plain; format=fasta" "https://rest.uniprot.org/uniprotkb/${PROTID}" --output "${CACHE}/'${PROTID}'.fa"

    if grep -q "Error messages" "${CACHE}/'${PROTID}'.fa"; then
        echo -e "\033[0;33mWarning:\033[0m could not download protein with ID: ${PROTID} and ORGANISM: ${ORGANISM}"
        return 1
    fi

    echo ">${PROTID}" >> $PROTEIN_OUTPUT
    sed 1d "${CACHE}/'${PROTID}'.fa" >> $PROTEIN_OUTPUT

    return 0
}

download_lncRNA () {
    if grep -q "${NONCODEID}" $INTERACTS_OUTPUT; then
        return
    fi

    python3 noncode_html_parser.py "${CACHE}/${NONCODEID}.html" > $TMP_FILE
    echo ">${NONCODEID}" >> $RNA_OUTPUT
    sed -n '2{p;q}' $TMP2_FILE >> $RNA_OUTPUT
}

echo -e "\033[1mlncRNA--protein downloader\033[0m -- by Iñaki Amatria Barral"

ANALYZED=0
TOTAL_LINES=`wc -l $INPUT | awk '{print $1}'`

while IFS= read -r LINE; do
    ANALYZED=$((ANALYZED + 1))
    echo -ne "\r\033[1mProgress:\033[0m ${ANALYZED}/${TOTAL_LINES}"

    IFS=$'\t' read -r -a WORDS <<< $LINE

    [[ "${LINE}" = "#"* ]] && continue

    if [[ "${WORDS[3]}" == "lncRNA" ]] && [[ "${WORDS[6]}" == "protein" ]]; then
        PROTID="${WORDS[5]}"
        NONCODEID="${WORDS[2]}"
        ORGANISM="${WORDS[10]}" && [[ "${#WORDS[@]}" -lt "16" ]] && ORGANISM="${WORDS[9]}"

        [[ ! "${ALLOWED_ORGANISMS[${ORGANISM}]+abc}" ]] && continue

        find_protein
        if [[ "$?" -gt "0" ]]; then
            PROTID="${WORDS[4]}"
            find_protein
            [[ "$?" -gt "0" ]] && continue
        fi

        find_lncRNA
        if [[ "$?" -gt "0" ]]; then
            NONCODEID="${WORDS[1]}"
            find_lncRNA
            [[ "$?" -gt "0" ]] && continue
        fi

        check_duplicate
        [[ "$?" -gt "0" ]] && continue

        download_protein
        [[ "$?" -gt "0" ]] && continue
        download_lncRNA

        echo -e "${PROTID}\t${NONCODEID}\t${ORGANISM}" >> $INTERACTS_OUTPUT
    fi
done < $INPUT

echo -e "\r\033[2K\033[1mlncRNA-protein downloader\033[0m -- done!"
