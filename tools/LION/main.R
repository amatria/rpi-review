library("LION")
library("seqinr")

rna_sequences <- read.fasta("rna.fa")
pro_sequences <- read.fasta("pro.fa")

run_LION(rna_sequences, pro_sequences)