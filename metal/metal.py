# utils/tree_estimator.py
import os
import warnings
import tempfile
import numpy as np
from Bio import SeqIO, Phylo
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
import dendropy

from .covariance_matrix import compute_covariance_matrix
from utils.treetools import tree_from_distance_matrix, distance_matrix_from_tree, transform_hamming_to_coalescent_distances

class Metal:
    """
    A class to estimate the species tree from aligned DNA sequences using the METAL model.
    It also provides alternative nonparametric and parametric estimates of the species tree, 
    which enables the computation of support values.
    
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

    def __init__(self, fasta_path=None, sequences=None, seed=None):
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
        self.metal_tree = None
        self.rng = np.random.default_rng(seed)
    
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
            self.dist_matrix = self._compute_hamming_distances(self.seq_array)
        self.metal_tree = self._build_tree(self.dist_matrix)
        return self.metal_tree

    def bootstrap_hamming(self, n_bootstraps):
        """
        Performs nonparametric bootstrap using Hamming distances.

        Parameters
        ----------
        n_bootstraps : int
            Number of bootstrap replicates.

        Returns
        -------
        dendropy.TreeList:
            List of UPGMA trees inferred from bootstrap replicates.
        """
        C = self.seq_array.shape[1]
        boot_trees = dendropy.TreeList()
        for _ in tqdm(range(n_bootstraps), desc="Bootstrapping"):
            idx = self.rng.choice(C, size=C, replace=True)
            boot_seq_array = self.seq_array[:, idx]
            boot_dist_matrix = self._compute_hamming_distances(boot_seq_array)
            boot_trees.append(self._build_tree(boot_dist_matrix))
        return boot_trees

    def gaussian_sampling(self, n_bootstraps, sites_per_gene, mutation_rate):
        """
        Performs parametric bootstrap using a multivariate normal model of distances.

        Parameters
        ----------
        n_bootstraps : int
            Number of bootstrap replicates.
        sites_per_gene : int
            Number of sites per gene.
        mutation_rate : float
            The mutation rate per unit time

        Returns
        -------
        dendropy.TreeList:
            Trees inferred from MVN distance matrices.
        """
        if self.metal_tree is None:
            self.estimate_tree()
        
        ultrametric_hd_distances = distance_matrix_from_tree(self.metal_tree)
        coalescent_distances = transform_hamming_to_coalescent_distances(ultrametric_hd_distances, mutation_rate)

        sigma = compute_covariance_matrix(coalescent_distances, sites_per_gene=sites_per_gene, mutation_rate = mutation_rate)
        sigma *= sites_per_gene / self.seq_len
        mu = self.dist_matrix[np.triu_indices(self.N, k=1)]
        samples = self.rng.multivariate_normal(mu, sigma, size=n_bootstraps)

        def to_dist_matrix(v):
            mat = np.zeros((self.N, self.N))
            i, j = np.triu_indices(self.N, k=1)
            mat[i, j] = mat[j, i] = v
            return mat

        trees = dendropy.TreeList([
            self._build_tree(to_dist_matrix(sample))
            for sample in tqdm(samples, desc="Gaussian sample tree reconstruction", total=n_bootstraps)
        ])

        return trees
