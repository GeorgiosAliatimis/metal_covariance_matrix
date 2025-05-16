from metal import Metal 
from utils.treetools import save_trees, compute_split_frequencies, get_bipartitions
from data_generation import TreeSimulator, SequenceSimulator
import random
import os 
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import numpy as np

def auc_comparison(num_gene_trees = 100,
    ntax= 20,
    num_sites_per_gene = 100,
    mutation_rate = 0.1,
    species_tree_diameter = 1.0,
    dir_name = "data",
    seed=0
):
    rng = random.Random(seed)

    tree_sim = TreeSimulator(ntax=ntax, tree_diameter=species_tree_diameter, rng=rng)

    # Generate species and gene trees
    tree_sim.generate_species_tree()
    gene_trees = tree_sim.generate_multiple_gene_trees(num_gene_trees)

    #Generate a fasta file of the concatenated sequences of gene trees

    seq_sim = SequenceSimulator(seq_length=num_sites_per_gene, mutation_rate=mutation_rate, seed=0)

    fasta_filepath = f"{dir_name}/concatenated_seq_alignment.fasta"
    seq_sim.write_concatenated(gene_trees, fasta_filepath)

    est = Metal(fasta_filepath, seed = seed)
    est.estimate_tree()

    boot_trees = est.bootstrap_hamming(n_bootstraps = 100)
    gauss_trees = est.gaussian_sampling(n_bootstraps = 100, mutation_rate=mutation_rate, sites_per_gene=num_sites_per_gene)

    boot_freqs = compute_split_frequencies(est.metal_tree, boot_trees)
    gauss_freqs = compute_split_frequencies(est.metal_tree, gauss_trees)

    true_splits = get_bipartitions(tree_sim.species_tree)

    def auc(freqs):
        splits = list(freqs.keys())
        y_true = [split in true_splits for split in splits]
        y_scores = [freqs[split] for split in splits]
        return roc_auc_score(y_true, y_scores)
    
    boot_score = auc(boot_freqs)
    gauss_score = auc(gauss_freqs)

    print(f"Auc for boot is {boot_score:.2f}")
    print(f"Auc for gauss is {gauss_score:.2f}")
    return boot_score, gauss_score

n_experiments = 100
aucs = {key: np.zeros( (n_experiments) ) for key in ["boot","gauss"] }
for i in range(n_experiments):
    boot_score, gauss_score = auc_comparison(ntax = 40, seed=i, mutation_rate=1)
    aucs["boot"][i]= boot_score   
    aucs["gauss"][i]= gauss_score
np.save("aucs.npy",aucs)