import numpy as np
from .tree_estimator import TreeEstimator

class Glass(TreeEstimator):
    """
    A class to estimate the species tree from aligned DNA sequences using the Glass model.
    
    Parameters
    ----------
    sites_per_gene: int
        Number of sites per gene
    fasta_path : str, optional
        Path to aligned FASTA file.
    sequences : dict, optional
        Dictionary of {taxon: sequence} pairs.
    
    Raises
    ------
    ValueError
        If neither fasta_path nor sequences are provided.
    """

    def __init__(
        self, 
        sites_per_gene: int,
        fasta_path: str =None, 
        sequences: dict[str,str]=None
    ):
        super().__init__(fasta_path=fasta_path, sequences= sequences)
        self.sites_per_gene = sites_per_gene
        self.n_genes = self.seq_len // self.sites_per_gene

    def estimate_tree(self):
        """
        Estimates a UPGMA tree from minima of gene tree Hamming distances and saves it.

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
        self.tree = self._build_tree(self.dist_matrix)
        return self.tree