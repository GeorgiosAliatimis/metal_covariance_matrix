from metal.estimators import *
from plot_tree import *
from split_frequencies import *

fasta_filepath = "gene_data/concatenated_seq_alignment.fasta"

metal_tree, _ = metal_estimator(fasta_filepath, "best_tree_metal.tre")
metal_bootstrap_estimators(fasta_filepath,n_bootstraps = 1000,output_filepath = "boot_trees_metal.tre")
multivariate_normal_bootstrap_estimators(fasta_filepath,n_bootstraps = 1000,output_filepath = "boot_trees_mvt.tre",mutation_rate=1, sites_per_gene=100)

compute_split_frequencies("best_tree_metal.tre", "boot_trees_metal.tre", output_filepath = "best_tree_metal_support.tre")
compute_split_frequencies("best_tree_metal.tre", "boot_trees_mvt.tre", output_filepath = "best_tree_mvt_support.tre")

plot_tree_with_support("best_tree_metal_support.tre")
plot_tree_with_support("best_tree_mvt_support.tre")

