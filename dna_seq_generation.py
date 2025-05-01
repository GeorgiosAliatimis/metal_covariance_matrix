import pyvolve
import os
from tqdm import tqdm
from collections import defaultdict

def scale_tree_branch_lengths(tree, mutation_rate):
    """Scales branch lengths of a DendroPy tree by mutation rate."""
    for node in tree.preorder_node_iter():
        if node.edge_length is not None:
            node.edge_length *= mutation_rate  # Scale branch lengths

def generate_sequences(tree, seq_length=100, model="Jukes-Cantor", mutation_rate = 1):
    """Simulates DNA sequences along a gene tree using the Jukes-Cantor model."""
    scale_tree_branch_lengths(tree, mutation_rate)

    # Convert DendroPy tree to Newick (needed for Pyvolve)
    newick_str = tree.as_string(schema="newick").replace("[&R]","")

    # Load tree into Pyvolve
    tree_pyvolve = pyvolve.read_tree(tree=newick_str)

    # Define evolutionary model (Jukes-Cantor)
    model = pyvolve.Model("nucleotide", {"rate_matrix": "JC"})  

    # Create partition for sequence evolution
    partition = pyvolve.Partition(models=model, size=seq_length)

    # Simulate sequences
    evolving_tree = pyvolve.Evolver(tree=tree_pyvolve, partitions=partition)
    evolving_tree(seqfile=None) 

    # Extract sequences as a dictionary
    sequences = evolving_tree.get_sequences()
    
    # Cleanup; remove temporary files
    os.remove("site_rates_info.txt")
    os.remove("site_rates.txt")
    return sequences

def generate_fasta_file_from_sequence(sequences, output_filepath = "./unnamed.fasta", **kwargs):
    with open(output_filepath, "w") as fasta_file:
        for id, seq in sequences.items():
            fasta_file.write(f">{id}\n")
            fasta_file.write(f"{seq}\n")

def generate_fasta_file_from_gene_tree(tree, output_filepath = "./unnamed.fasta", **kwargs):
    sequences = generate_sequences(tree, **kwargs)
    generate_fasta_file_from_sequence(sequence, output_filepath, **kwargs)
    

def generate_fasta_files_from_gene_trees(trees,output_directory = "fasta_files", **kwargs):
    for i, tree in tqdm(enumerate(trees)):
        filepath = os.path.join(output_directory, f"{i}.fasta")
        generate_fasta_files_from_gene_trees(tree,filepath, **kwargs)

def generate_concatenated_fasta_file_from_gene_trees(trees,output_filepath = "./unnamed.fasta", **kwargs):
    concatenated_seq = defaultdict(list)
    for i, tree in tqdm(enumerate(trees), total = len(trees)):
        sequences = generate_sequences(tree, **kwargs)
        for k, v in sequences.items():
            concatenated_seq[k].extend(v)
    generate_fasta_file_from_sequence(concatenated_seq, output_filepath, **kwargs)
