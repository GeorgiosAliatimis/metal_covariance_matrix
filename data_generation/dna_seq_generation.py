import pyvolve
import os
from tqdm import tqdm
from collections import defaultdict

from utils.treetools import scale_tree
class SequenceSimulator:
    """
    A utility class to simulate nucleotide sequences along phylogenetic trees using Pyvolve,
    and export results to FASTA files (individual or concatenated).

    Attributes:
        seq_length (int): Length of the sequences to simulate.
        model (str): Evolutionary model to use. Currently only "Jukes-Cantor" is supported.
        mutation_rate (float): Mutation rate used to scale branch lengths of trees.
        seed (int or None): Random seed for reproducibility.
    """

    def __init__(self, seq_length=100, model="Jukes-Cantor", mutation_rate=1.0, seed=None):
        """
        Initialize a SequenceSimulator instance with simulation parameters.

        Args:
            seq_length (int): Length of DNA sequences to simulate.
            model (str): Evolutionary model. Only "Jukes-Cantor" is supported.
            mutation_rate (float): Mutation rate to scale tree branch lengths.
            seed (int or None): Optional random seed for reproducibility.
        """
        self.seq_length = seq_length
        self.model = model
        self.mutation_rate = mutation_rate
        self.seed = seed

    def _scale_tree(self, tree):
        """
        Scale all branch lengths of a given tree by the mutation rate.

        Args:
            tree (dendropy.Tree): The input tree whose branch lengths are to be scaled.
        """
        scale_tree(tree, self.mutation_rate)

    def simulate(self, tree):
        """
        Simulate nucleotide sequences along a phylogenetic tree.

        Args:
            tree (dendropy.Tree): The phylogenetic tree for sequence evolution.

        Returns:
            dict: A dictionary mapping taxon labels to simulated DNA sequences.
        """
        self._scale_tree(tree)
        newick_str = tree.as_string(schema="newick").replace("[&R]", "")
        tree_pyvolve = pyvolve.read_tree(tree=newick_str)
        model = pyvolve.Model("nucleotide", {"rate_matrix": "JC"})
        partition = pyvolve.Partition(models=model, size=self.seq_length)
        evolver = pyvolve.Evolver(tree=tree_pyvolve, partitions=partition)
        evolver(seqfile=None, ratefile=None, infofile=None, seed=self.seed)
        return evolver.get_sequences()

    def simulate_concatenated(self, trees):
        """
        Simulate sequences from multiple trees and returns concatenated sequences.

        Args:
            trees (list of dendropy.Tree): List of gene trees to simulate gene sequences from.
        """
        concatenated = defaultdict(list)
        for tree in tqdm(trees, desc="Generating & concatenating sequences"):
            seqs = self.simulate(tree)
            for taxon, seq in seqs.items():
                concatenated[taxon].append(seq)
        concatenated = {taxon: ''.join(parts) for taxon, parts in concatenated.items()}
        return concatenated

    def write_fasta(self, sequences, filepath):
        """
        Write a dictionary of sequences to a FASTA file.

        Args:
            sequences (dict): Dictionary of {label: sequence} pairs.
            filepath (str): Output path for the FASTA file.
        """
        with open(filepath, "w") as fasta_file:
            for taxon, seq in sequences.items():
                fasta_file.write(f">{taxon}\n{seq}\n")

    def simulate_and_write(self, tree, filepath):
        """
        Simulate sequences from a tree and write them to a FASTA file.

        Args:
            tree (dendropy.Tree): The phylogenetic tree to simulate sequences from.
            filepath (str): Output path for the FASTA file.
        """
        sequences = self.simulate(tree)
        self.write_fasta(sequences, filepath)

    def batch_write(self, trees, directory):
        """
        Simulate sequences for a list of trees and write each to a separate FASTA file.

        Args:
            trees (list of dendropy.Tree): List of gene trees to process.
            directory (str): Output directory to store generated FASTA files.
        """
        os.makedirs(directory, exist_ok=True)
        for i, tree in tqdm(enumerate(trees), desc="Generating FASTA files", total=len(trees)):
            path = os.path.join(directory, f"{i}.fasta")
            self.simulate_and_write(tree, path)

    def write_concatenated(self, trees, filepath):
        """
        Simulate sequences from multiple trees and write a single concatenated FASTA file.

        Args:
            trees (list of dendropy.Tree): List of gene trees to simulate.
            filepath (str): Output path for the concatenated FASTA file.
        """
        concatenated_seq = self.simulate_concatenated(trees)
        self.write_fasta(concatenated_seq, filepath)
