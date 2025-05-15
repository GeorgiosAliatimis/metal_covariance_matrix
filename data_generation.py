from data_generation import TreeSimulator, SequenceSimulator
import random
import os 

num_gene_trees = 500
num_sites_per_gene = 100
mutation_rate = 0.1

dir_name = "gene_data"

if dir_name not in os.listdir("."):  
    os.mkdir(dir_name)

# Fix seed
rng = random.Random(0)

tree_sim = TreeSimulator(ntax=10, tree_diameter=1.0, rng=rng)

# Generate species tree
tree_sim.generate_species_tree()
tree_sim.get_species_tree().write(path=f"{dir_name}/species_tree.nex", schema="nexus")

# Generate gene trees
gene_trees = tree_sim.generate_multiple_gene_trees(num_gene_trees)
gene_trees.write(path=f"{dir_name}/gene_trees.nex", schema="nexus")


#Generate a fasta file of the concatenated sequences of gene trees

seq_sim = SequenceSimulator(seq_length=num_sites_per_gene, mutation_rate=mutation_rate, seed=42)

# Concatenated FASTA from multiple gene trees
seq_sim.write_concatenated(gene_trees, f"{dir_name}/concatenated_seq_alignment.fasta")
