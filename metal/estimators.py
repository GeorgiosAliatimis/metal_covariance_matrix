import os
import warnings
import numpy as np
from Bio import SeqIO
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
import dendropy
from covariance_matrix import compute_covariance_matrix

def sequences_from_fasta(fasta_filepath):
    """
    Reads sequences from a FASTA file.

    Parameters
    ----------
    fasta_filepath : str
        Path to the FASTA file.

    Returns
    -------
    dict
        A dictionary mapping sequence IDs to sequences as strings.
    """
    sequences = {}
    for record in SeqIO.parse(fasta_filepath, "fasta"):
        sequences[record.id.strip()] = str(record.seq)
    return sequences


def hamming_distance_from_sequences(sequences):
    """
    Computes the Hamming distance matrix for a dictionary of aligned sequences.

    Parameters
    ----------
    sequences : dict
        Dictionary mapping taxa labels to equal-length sequence strings.

    Returns
    -------
    np.ndarray
        A square matrix of normalized Hamming distances between taxa.
    """
    taxa = sorted(sequences.keys())
    seq_array = np.array([list(sequences[taxon]) for taxon in taxa])
    dist_matrix = pdist(seq_array, metric=lambda x, y: np.mean(x != y))
    return squareform(dist_matrix)


def bootstrap_hamming_distances(sequences, n_bootstraps):
    """
    Computes bootstrap replicate Hamming distance matrices.

    Parameters
    ----------
    sequences : dict
        Dictionary mapping taxa labels to equal-length sequence strings.
    n_bootstraps : int
        Number of bootstrap replicates to compute.

    Returns
    -------
    np.ndarray
        Array of shape (n_bootstraps, N, N) where N is number of taxa.
    """
    taxa = sorted(sequences.keys())
    seq_array = np.array([list(sequences[taxon]) for taxon in taxa])
    n, C = seq_array.shape
    dist_matrix = np.zeros((n_bootstraps, n, n))

    for i in tqdm(range(n_bootstraps), desc="Bootstrapping"):
        col_indices = np.random.choice(C, size=C, replace=True)
        D = pdist(seq_array[:, col_indices], metric=lambda x, y: np.mean(x != y))
        dist_matrix[i, :, :] = squareform(D)

    return dist_matrix


def get_tree_from_dissimilarities(dist_matrix):
    """
    Constructs a UPGMA tree from a given dissimilarity matrix.

    Parameters
    ----------
    dist_matrix : np.ndarray
        Square matrix of distances between taxa.

    Returns
    -------
    dendropy.Tree
        A phylogenetic tree inferred using UPGMA.
    """
    num_taxa = dist_matrix.shape[0]
    taxa_labels = [chr(ord('a') + i) for i in range(num_taxa)]
    taxa = dendropy.TaxonNamespace(taxa_labels)

    tmp_file_name = "tmp.csv"
    np.savetxt(tmp_file_name, dist_matrix, delimiter=",")
    
    with open(tmp_file_name) as src:
        pdm = dendropy.PhylogeneticDistanceMatrix.from_csv(
            src,
            taxon_namespace=taxa,
            is_first_row_column_names=False,
            is_first_column_row_names=False,
            is_allow_new_taxa=True,
            delimiter=",",
        )
    os.remove(tmp_file_name)
    return pdm.upgma_tree()


def save_trees(trees, output_filepath="./trees.tre"):
    """
    Saves a list of trees to a file in the appropriate format.

    Parameters
    ----------
    trees : list of dendropy.Tree
        List of phylogenetic trees to write.
    output_filepath : str
        Filepath where the trees will be saved.
    """
    _, ext = os.path.splitext(output_filepath)
    ext = ext.lower()
    if ext in [".nex", ".nexus"]:
        schema = "nexus"
    elif ext in [".nwk", ".newick", ".tre"]:
        schema = "newick"
    else:
        warnings.warn(f"Unrecognized extension '{ext}'. Defaulting to 'newick' schema.")
        schema = "newick"

    tree_list = dendropy.TreeList(taxon_namespace=trees[0].taxon_namespace)
    tree_list.extend(trees)
    tree_list.write(path=output_filepath, schema=schema, suppress_rooting=True)


def metal_estimator(fasta_filepath, output_filepath="./best_tree_metal.tre"):
    """
    Estimates a UPGMA tree from FASTA sequences using Hamming distances.

    Parameters
    ----------
    fasta_filepath : str
        Input FASTA file containing aligned sequences.
    output_filepath : str
        File to write the resulting tree.
    """
    sequences = sequences_from_fasta(fasta_filepath)
    dist_matrix = hamming_distance_from_sequences(sequences)
    metal_tree = get_tree_from_dissimilarities(dist_matrix)
    save_trees([metal_tree], output_filepath)


def metal_bootstrap_estimators(fasta_filepath, n_bootstraps, output_filepath="./boot_trees_metal.tre"):
    """
    Performs bootstrapping and builds multiple UPGMA trees from sequence resampling.

    Parameters
    ----------
    fasta_filepath : str
        Input FASTA file with aligned sequences.
    n_bootstraps : int
        Number of bootstrap trees to generate.
    output_filepath : str
        Path to save the bootstrap trees.
    """
    sequences = sequences_from_fasta(fasta_filepath)
    dist_matrices = bootstrap_hamming_distances(sequences, n_bootstraps)
    boot_trees = [get_tree_from_dissimilarities(dist_matrix) for dist_matrix in dist_matrices]
    save_trees(boot_trees, output_filepath)

def  multivariate_normal_bootstrap_estimators(fasta_filepath, n_bootstraps, output_filepath="./boot_trees_mvt.tre", **kwargs):
    sequences = sequences_from_fasta(fasta_filepath)
    dist_matrix = hamming_distance_from_sequences(sequences)
    metal_tree = get_tree_from_dissimilarities(dist_matrix)
    sigma, leaf_pairs, _ = compute_covariance_matrix(dist_matrix,**kwargs)
    mu = np.array([dist_matrix[*pair] for pair in leaf_pairs])   
    mvt_samples = np.random.multivariate_normal(mean = mu, cov = sigma, size=n_bootstraps)
    def vector_to_matrix(v):
        N = dist_matrix.shape[0]
        mat = np.zeros( (N,N) )
        triu_i, triu_j = np.triu_indices(N, k=1)
        mat[triu_i, triu_j] = v
        mat[triu_j, triu_i] = v  
        return mat
    boot_trees = [get_tree_from_dissimilarities(vector_to_matrix(sample)) for sample in mvt_samples]
    save_trees(boot_trees, output_filepath)