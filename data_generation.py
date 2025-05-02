from data_generation.tree_sims import generate_species_tree, generate_gene_tree_from_species_tree
from data_generation.dna_seq_generation import generate_concatenated_fasta_file_from_gene_trees
import random
import dendropy
import os 

num_gene_trees = 1000
num_sites_per_gene = 100

dir_name = "gene_data"

if dir_name not in os.listdir("."):  
    os.mkdir(dir_name)

# Fix seed
rng = random.Random(0)

#Generate the species tree
species_tree = generate_species_tree(ntax=10,tree_depth = 1,normalize = True, rng=rng)
species_tree.write(path=f"{dir_name}/species_tree.nex",schema = "nexus")

#Generate gene trees from the species tree
gene_trees = [generate_gene_tree_from_species_tree(species_tree, normalize = False) for _ in range(num_gene_trees)] 
gene_trees = dendropy.TreeList(gene_trees)
gene_trees.write(path=f"{dir_name}/gene_trees.nex", schema = "nexus")

#Generate a fasta file of the concatenated sequences of gene trees
generate_concatenated_fasta_file_from_gene_trees(trees = gene_trees, \
                        output_filepath = f"{dir_name}/concatenated_seq_alignment.nex", \
                        seq_length = num_sites_per_gene)