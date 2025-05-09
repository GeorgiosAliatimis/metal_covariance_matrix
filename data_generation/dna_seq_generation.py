"""
DNA Sequence Simulation and FASTA File Utilities

This module provides tools to:
- Simulate nucleotide sequences along phylogenetic trees using Pyvolve.
- Export simulated sequences to FASTA files (individual or concatenated).
- Scale tree branch lengths according to a mutation rate.

Dependencies:
- pyvolve
- dendropy
- tqdm
- numpy
- collections (defaultdict)
- os
"""

import pyvolve
import os
from tqdm import tqdm
from collections import defaultdict


def scale_tree_branch_lengths(tree, mutation_rate):
    """
    Scales all branch lengths in a DendroPy tree by a mutation rate.

    Parameters:
        tree (dendropy.Tree): The input tree whose branch lengths will be scaled.
        mutation_rate (float): The multiplier for all edge lengths.
    """
    for node in tree.preorder_node_iter():
        if node.edge_length is not None:
            node.edge_length *= mutation_rate


def generate_sequences(tree, seq_length=100, model="Jukes-Cantor", mutation_rate=1, seed=None):
    """
    Simulates nucleotide sequences along a phylogenetic tree using Pyvolve.

    Parameters:
        tree (dendropy.Tree): Gene tree to evolve sequences along.
        seq_length (int): Length of DNA sequences to simulate.
        model (str): Evolutionary model to use. Currently supports only "Jukes-Cantor".
        mutation_rate (float): Mutation rate used to scale branch lengths.

    Returns:
        dict: A dictionary mapping taxon labels to simulated DNA sequences.
    """
    scale_tree_branch_lengths(tree, mutation_rate)

    # Convert the DendroPy tree to Newick format for Pyvolve
    newick_str = tree.as_string(schema="newick").replace("[&R]", "")
    tree_pyvolve = pyvolve.read_tree(tree=newick_str)

    # Define evolutionary model (currently fixed to Jukes-Cantor)
    model = pyvolve.Model("nucleotide", {"rate_matrix": "JC"})

    # Set up simulation
    partition = pyvolve.Partition(models=model, size=seq_length)
    evolving_tree = pyvolve.Evolver(tree=tree_pyvolve, partitions=partition)

    # Run simulation (in-memory)
    evolving_tree(seqfile=None, ratefile=None, infofile=None, seed=seed)

    # Extract simulated sequences
    sequences = evolving_tree.get_sequences()

    return sequences


def generate_fasta_file_from_sequence(sequences, output_filepath="./unnamed.fasta"):
    """
    Writes a dictionary of sequences to a FASTA file.

    Parameters:
        sequences (dict): Dictionary of {label: sequence} pairs.
        output_filepath (str): Path to the output FASTA file.
    """
    with open(output_filepath, "w") as fasta_file:
        for id, seq in sequences.items():
            fasta_file.write(f">{id}\n{seq}\n")


def generate_fasta_file_from_gene_tree(tree, output_filepath="./unnamed.fasta", **kwargs):
    """
    Simulates sequences from a gene tree and writes them to a FASTA file.

    Parameters:
        tree (dendropy.Tree): The gene tree to simulate sequences on.
        output_filepath (str): Destination FASTA file.
        **kwargs: Additional arguments passed to `generate_sequences()`.
    """
    sequences = generate_sequences(tree, **kwargs)
    generate_fasta_file_from_sequence(sequences, output_filepath)


def generate_fasta_files_from_gene_trees(trees, output_directory="fasta_files", **kwargs):
    """
    Simulates and writes separate FASTA files for a list of gene trees.

    Parameters:
        trees (list of dendropy.Tree): List of gene trees to process.
        output_directory (str): Directory where FASTA files will be saved.
        **kwargs: Additional arguments passed to `generate_sequences()`.
    """
    os.makedirs(output_directory, exist_ok=True)
    for i, tree in tqdm(enumerate(trees), desc="Generating FASTA files", total=len(trees)):
        filepath = os.path.join(output_directory, f"{i}.fasta")
        generate_fasta_file_from_gene_tree(tree, filepath, **kwargs)


def generate_concatenated_fasta_file_from_gene_trees(trees, output_filepath="./unnamed.fasta", **kwargs):
    """
    Simulates sequences from multiple gene trees and writes a concatenated FASTA file.

    Parameters:
        trees (list of dendropy.Tree): List of gene trees.
        output_filepath (str): Path to the output concatenated FASTA file.
        **kwargs: Additional arguments passed to `generate_sequences()`.
    """
    concatenated_seq = defaultdict(list)
    desc = "Generating & concatenating sequences"
    for i, tree in tqdm(enumerate(trees), desc=desc, total=len(trees)):
        sequences = generate_sequences(tree, **kwargs)
        for taxon, seq in sequences.items():
            concatenated_seq[taxon].append(seq)

    # Join sequences for each taxon
    final_seqs = {k: "".join(v) for k, v in concatenated_seq.items()}
    generate_fasta_file_from_sequence(final_seqs, output_filepath)
