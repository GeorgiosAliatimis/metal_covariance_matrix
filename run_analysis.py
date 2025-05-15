from metal import Metal 
from utils.plot_tree import plot_tree_from_file, print_tree
from split_frequencies import compute_split_frequencies
from utils.treetools import save_trees
import matplotlib.pyplot as plt

fasta_filepath = "gene_data/concatenated_seq_alignment.fasta"

est = Metal(fasta_filepath, seed = 0)
est.estimate_tree()
print_tree(est.metal_tree, show_support=False, show_lengths=False)
save_trees([est.metal_tree], "metal_tree.tre","newick")
boot_trees = est.bootstrap_hamming(n_bootstraps = 100)
save_trees(boot_trees, "boot_trees.tre","newick")
gauss_trees = est.gaussian_sampling(n_bootstraps = 1000, mutation_rate=0.1, sites_per_gene=100)
save_trees(gauss_trees, "gauss_trees.tre","newick")

metal_boot_support = compute_split_frequencies(est.metal_tree, boot_trees)
save_trees([metal_boot_support], "metal_boot_support.tre", "newick")
metal_gauss_support = compute_split_frequencies(est.metal_tree, gauss_trees)
save_trees([metal_gauss_support], "metal_gauss_support.tre", "newick")

print_tree(metal_boot_support,show_lengths=False)
print_tree(metal_gauss_support,show_lengths=False)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
plot_tree_from_file("metal_boot_support.tre", schema= "newick", ax = ax1)
plot_tree_from_file("metal_gauss_support.tre", schema= "newick", ax = ax2)
plot_tree_from_file("gene_data/species_tree.nex", schema = "nexus", ax = ax3)

ax1.set_title("Bootstrap approach")
ax2.set_title("Gaussian sampling approach")
ax3.set_title("Correct species tree")

plt.show()
