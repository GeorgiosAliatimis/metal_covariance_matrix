import warnings
import numpy as np
from Bio import SeqIO
from scipy.spatial.distance import pdist, squareform

from utils.treetools import tree_from_distance_matrix, distance_matrix_from_tree, transform_hamming_to_coalescent_distances

class Glass:
    """
    A class to estimate the species tree from aligned DNA sequences using the Glass model.
    
    Parameters
    ----------
    fasta_path : str, optional
        Path to aligned FASTA file.
    sequences : dict, optional
        Dictionary of {taxon: sequence} pairs.
    
    Raises
    ------
    ValueError
        If neither fasta_path nor sequences are provided.
    """

    def __init__(self, sites_per_gene,fasta_path=None, sequences=None):
        if fasta_path is not None:
            if sequences is not None:
                warnings.warn("Both fasta_path and sequences are given; sequences will be discarded and fasta_path will be used instead.")
            self.fasta_path = fasta_path
            self.sequences = self._load_sequences()
        elif sequences is not None:
            self.fasta_path = None
            self.sequences = sequences
        else:
            raise ValueError("You must provide either a FASTA file path or a sequences dictionary.")
        
        self._validate_sequences()

        self.labels = sorted(self.sequences.keys())
        self.N = len(self.labels)
        self.seq_len = len(next(iter(self.sequences.values())))
        self.seq_array = self._to_array()
        self.dist_matrix = None
        self.glass_tree = None
        self.sites_per_gene = sites_per_gene
        self.n_genes = self.seq_len // self.sites_per_gene
        
    
    def _validate_sequences(self):
        """
        Validates in-memory sequences for uniform length.

        Raises
        ------
        ValueError
            If no sequences are found or lengths are inconsistent.
        """
        lengths = {len(seq) for seq in self.sequences.values()}
        if not self.sequences:
            raise ValueError("No sequences provided.")
        if len(lengths) > 1:
            raise ValueError("Not all sequences are the same length.")


    def _load_sequences(self):
        """
        Loads and validates sequences from the FASTA file.

        Returns
        -------
        dict
            Dictionary of {taxon_name: sequence}.
        
        Raises
        ------
        ValueError
            If no sequences are found or lengths are inconsistent.
        """
        sequences = {}
        for record in SeqIO.parse(self.fasta_path, "fasta"):
            seq = str(record.seq)
            sequences[record.id.strip()] = seq
        return sequences

    def _to_array(self):
        """
        Converts sequences to a character matrix for pairwise comparison.

        Returns
        -------
        np.ndarray
            2D array of shape (n_taxa, sequence_length).
        """
        return np.array([list(self.sequences[t]) for t in self.labels], dtype='<U1')

    def _compute_hamming_distances(self, seq_array):
        """
        Computes normalized Hamming distances between all pairs of taxa.

        Returns
        -------
        np.ndarray
            Square distance matrix of shape (N, N).
        """
        return squareform(pdist(seq_array, metric=lambda x, y: np.mean(x != y)))

    def _build_tree(self, matrix):
        """
        Constructs a UPGMA tree from a distance matrix using DendroPy.

        Parameters
        ----------
        matrix : np.ndarray
            Square symmetric distance matrix.

        Returns
        -------
        dendropy.Tree
            UPGMA tree.
        """
        return tree_from_distance_matrix(matrix, self.labels)

    def estimate_tree(self):
        """
        Estimates a UPGMA tree from Hamming distances and saves it.

        Parameters
        ----------
        output_path : str
            Path to save the resulting tree (Newick or Nexus).

        Returns
        -------
        dendropy.Tree
            Inferred UPGMA tree.
        """
        if self.dist_matrix is None:
            self.dist_matrix= np.full( (self.N,self.N), float("inf") )
            for gene_id in range(self.n_genes):
                start = gene_id * self.sites_per_gene
                end = (gene_id + 1) * self.sites_per_gene
                seq_array = self.seq_array[:,start:end]
                gene_dist_matrix = self._compute_hamming_distances(seq_array)
                self.dist_matrix = np.minimum(self.dist_matrix, gene_dist_matrix)
        self.glass_tree = self._build_tree(self.dist_matrix)
        return self.glass_tree