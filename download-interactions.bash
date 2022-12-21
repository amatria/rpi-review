#!/usr/bin/env bash

# START OF CONFIGURATION

MAXIMUM_RNA_LENGTH=10000
MINIMUM_PROTEIN_LENGTH=30

INPUT=const/npinter-v5.txt
NONCODEIDS=const/ids-noncode-v5.txt
RNA_OUTPUT=rna.fa
PROTEIN_OUTPUT=pro.fa
INTERACTIONS_OUTPUT=pairs.txt

# END OF CONFIGURATION

[[ ! -f $INPUT ]] || [[ ! -f $NONCODEIDS ]] && echo -e "\033[0;31mError:\033[0m missing input files" && exit 1

[[ -f $RNA_OUTPUT ]] && rm -rf $RNA_OUTPUT
[[ -f $PROTEIN_OUTPUT ]] && rm -rf $PROTEIN_OUTPUT
[[ -f $INTERACTIONS_OUTPUT ]] && rm -rf $INTERACTIONS_OUTPUT

CACHE=cache
declare -A ALLOWED_ORGANISMS=( ["Homo sapiens"]=0 ["Mus musculus"]=0 ["Drosophila melanogaster"]=0 )

mkdir -p $CACHE

touch $RNA_OUTPUT
touch $PROTEIN_OUTPUT
touch $INTERACTIONS_OUTPUT

find_protein_uniprot () {
    local PROTID_LINK=$(sed -e "s/ /%20/g" <<< "${PROTID}")
    curl -s -f -H "Accept: text/plain; format=tsv" "https://rest.uniprot.org/uniprotkb/search?query=${PROTID_LINK}" --output "${CACHE}/'${PROTID}'.txt"
    [[ "$?" -gt "0" ]] && echo -e "\r\033[2K\033[1m\033[0;33mWarning:\033[0m could not download information for protein with ID: ${PROTID} and ORGANISM: ${ORGANISM}" && return 1
    return 0
}

find_protein () {
    [[ "${PROTID}" == "-" ]] || \
    [[ "${PROTID}" == *"heterodimer"* ]] || \
    [[ "${PROTID}" == *"#"* ]] || \
    [[ "${PROTID}" == *[![:ascii:]]* ]] && \
    return 1

    if [[ ! -f "${CACHE}/'${PROTID}'.txt" ]]; then
        find_protein_uniprot || return 1
    fi

    local NUM_LINES=`wc -l "${CACHE}/'${PROTID}'.txt" | awk '{print $1}'`
    [[ "${NUM_LINES}" -lt "2" ]] && return 1

    local LINE
    local WORDS
    while IFS= read -r LINE
    do
        IFS=$'\t' read -r -a WORDS <<< $LINE
        local TMP_PROTID="${WORDS[0]}"
        local TMP_ORGANISM="${WORDS[5]}"
        if grep -i -q "${ORGANISM}" <<< "${TMP_ORGANISM}"; then
            PROTID="${TMP_PROTID}"
            return 0
        fi
    done < "${CACHE}/'${PROTID}'.txt"

    return 1
}

find_lncRNA_noncode () {
    curl -s -f "http://www.noncode.org/show_rna.php?id=${NAMES[0]}&version=${NAMES[1]}" --output "${CACHE}/${TMP_NAME}.html"
    [[ "$?" -gt "0" ]] && echo -e "\r\033[2K\033[1m\033[0;33mWarning:\033[0m could not download information for lncRNA with ID: ${TMP_NAME} and ORGANISM: ${ORGANISM}" && return 1
    return 0
}

find_lncRNA () {
    [[ "${NONCODEID}" == "-" ]] && return 1

    local LINE
    local WORDS
    local NAMES
    while IFS= read -r LINE; do
        IFS=$'\t' read -r -a WORDS <<< $LINE
        local TMP_NAME="${WORDS[0]}"
        IFS="." read -r -a NAMES <<< "${TMP_NAME}"

        if [[ ! -f "${CACHE}/${TMP_NAME}.html" ]]; then
            find_lncRNA_noncode || return 1
        fi

        local TMP_ORGANISM=`python3 utils/noncode_html_parser.py "${CACHE}/${TMP_NAME}.html" organism`
        if  grep -i -q "${ORGANISM}" <<< "${TMP_ORGANISM}"; then
            NONCODEID="${TMP_NAME}"
            return 0
        fi
    done < <(grep -i -w "${NONCODEID}" $NONCODEIDS)

    return 1
}

check_duplicate () {
    if grep -q -P "${PROTID}\t${NONCODEID}\t${ORGANISM}" $INTERACTIONS_OUTPUT; then
        return 1
    fi
    return 0
}

download_protein_uniprot () {
    curl -s -f -H "Accept: text/plain; format=fasta" "https://rest.uniprot.org/uniprotkb/${PROTID}" --output "${CACHE}/'${PROTID}'.fa"
    [[ "$?" -gt "0" ]] && echo -e "\r\033[2K\033[1m\033[0;33mWarning:\033[0m could not download protein with ID: ${PROTID} and ORGANISM: ${ORGANISM}" && return 1
    return 0
}

download_protein () {
    if grep -q "${PROTID}" $INTERACTIONS_OUTPUT; then
        return 0
    fi

    if [[ ! -f "${CACHE}/'${PROTID}'.fa" ]]; then
        download_protein_uniprot || return 1
    fi

    if grep -q "Error messages" "${CACHE}/'${PROTID}'.fa"; then
        echo -e "\r\033[2K\033[1m\033[0;33mWarning:\033[0m error with downloaded protein with ID: ${PROTID} and ORGANISM: ${ORGANISM}"
        return 1
    fi

    local SEQUENCE=`sed 1d "${CACHE}/'${PROTID}'.fa"`
    local SEQUENCE_LENGTH=`echo "${SEQUENCE}" | wc -c`
    [[ "${SEQUENCE_LENGTH}" -lt "${MINIMUM_PROTEIN_LENGTH}" ]] && echo -e "\r\033[2K\033[1m\033[0;33mWarning:\033[0m skipping protein with ID: ${PROTID} and ORGANISM: ${ORGANISM} because it is too small" && return 1

    echo -e ">${PROTID}\n${SEQUENCE}" >> $PROTEIN_OUTPUT

    return 0
}

download_lncRNA () {
    if grep -q "${NONCODEID}" $INTERACTIONS_OUTPUT; then
        return 0
    fi

    local SEQUENCE=`python3 utils/noncode_html_parser.py "${CACHE}/${NONCODEID}.html" sequence`
    local SEQUENCE_LENGTH=`echo "${SEQUENCE}" | wc -c`
    [[ "${SEQUENCE_LENGTH}" -gt "${MAXIMUM_RNA_LENGTH}" ]] && echo -e "\r\033[2K\033[1m\033[0;33mWarning:\033[0m skipping lncRNA with ID: ${NONCODEID} and ORGANISM: ${ORGANISM} because it is too big" && return 1
    
    echo -e ">${NONCODEID}\n${SEQUENCE}" >> $RNA_OUTPUT

    return 0
}

echo -e "\033[1mlncRNA-protein downloader\033[0m -- by Iñaki Amatria Barral"

ANALYZED=0
TOTAL_LINES=`wc -l $INPUT | awk '{print $1}'`

while IFS= read -r LINE; do
    ANALYZED=$((ANALYZED + 1))
    PROGRESS=`echo "${ANALYZED} ${TOTAL_LINES}" | awk '{printf "%.2f", $1 / $2 * 100}'`
    echo -ne "\r\033[2K\033[1m\033[1mProgress:\033[0m ${PROGRESS}% (${ANALYZED}/${TOTAL_LINES})"

    [[ "${LINE}" == "#"* ]] && continue

    IFS=$'\t' read -r -a WORDS <<< $LINE
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
        [[ "$?" -gt "0" ]] && continue

        echo -e "${PROTID}\t${NONCODEID}\t${ORGANISM}" >> $INTERACTIONS_OUTPUT
    fi
done < $INPUT

echo -e "\r\033[2K\033[1mlncRNA-protein downloader\033[0m -- done!"
