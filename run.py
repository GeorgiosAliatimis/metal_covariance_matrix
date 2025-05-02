from metal.estimators import *

fasta_filepath = "gene_data/concatenated_seq_alignment.nex"

metal_estimator(fasta_filepath, "best_tree_metal.tre")
#metal_bootstrap_estimators(fasta_filepath, 1000)
multivariate_normal_bootstrap_estimators(fasta_filepath,1000,mutation_rate=1, sites_per_gene=100)
