import numpy as np
from tqdm import tqdm
import dendropy
from utils.matrix_metrics import matrices_comparison
from .covariance_matrix import compute_covariance_matrix
from utils.treetools import distance_matrix_from_tree, transform_hamming_to_coalescent_distances
from .tree_estimator import TreeEstimator

class Metal(TreeEstimator):
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
        super().__init__(fasta_path=fasta_path, sequences= sequences)
        self.rng = np.random.default_rng(seed)
    
    def _compute_covariance_matrix(self, mutation_rate, sites_per_gene, mode = "total"):
        if self.tree is None:
            self.estimate_tree()
        
        ultrametric_hd_distances = distance_matrix_from_tree(self.tree)
        coalescent_distances = transform_hamming_to_coalescent_distances(ultrametric_hd_distances, mutation_rate)
        sigma = compute_covariance_matrix(
            dist_matrix=coalescent_distances, 
            sites_per_gene=sites_per_gene, 
            mutation_rate = mutation_rate,
            mode = mode
        )
        sigma *= sites_per_gene / self.seq_len
        return sigma

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
        self.tree = self._build_tree(self.dist_matrix)
        return self.tree

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
        
        sigma = self._compute_covariance_matrix(sites_per_gene=sites_per_gene, mutation_rate = mutation_rate)
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

    def ratio_of_coalescent_uncertainty(self,sites_per_gene, mutation_rate):
        sigma_coal = self._compute_covariance_matrix(
            sites_per_gene=sites_per_gene, 
            mutation_rate = mutation_rate,
            mode = "coal"
        )
        sigma_total = self._compute_covariance_matrix(
            sites_per_gene=sites_per_gene, 
            mutation_rate = mutation_rate,
            mode = "total"
        )
        rescaling_factor = 1/sigma_total.mean()
        sigma_coal *= rescaling_factor
        sigma_total*= rescaling_factor

        return matrices_comparison(sigma_coal,sigma_total)
