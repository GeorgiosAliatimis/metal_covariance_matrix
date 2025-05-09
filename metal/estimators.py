import os
import warnings
import numpy as np
from Bio import SeqIO, Phylo
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
import dendropy
import tempfile
from .covariance_matrix import compute_covariance_matrix

def sequences_from_fasta(fasta_filepath):
    """
    Reads sequences from a FASTA file and verifies they are all the same length.

    Parameters
    ----------
    fasta_filepath : str
        Path to the FASTA file.

    Returns
    -------
    dict
        Dictionary mapping sequence IDs to sequence strings.

    Raises
    ------
    ValueError
        If sequences are not all the same length.
    """
    sequences = {}
    lengths = set()
    for record in SeqIO.parse(fasta_filepath, "fasta"):
        seq = str(record.seq)
        sequences[record.id.strip()] = seq
        lengths.add(len(seq))

    if len(lengths) == 0:
        raise ValueError("No sequences found in the FASTA file.")
    if len(lengths) > 1:
        raise ValueError("Not all sequences are the same length.")

    return sequences

def get_sequence_length(fasta_filepath):
    """
    Determines the length of sequences in a FASTA file, ensuring all sequences are the same length.

    Parameters:
        fasta_filepath (str): Path to the input FASTA file.

    Returns:
        int: Length of the sequences in the FASTA file.
    """

    lengths = set()
    for record in SeqIO.parse(fasta_filepath, "fasta"):
        lengths.add(len(record.seq))
    if len(lengths) == 0:
        raise ValueError("No sequences found in the FASTA file.")
    if len(lengths) > 1:
        raise ValueError("Not all sequences are the same length.")
    return lengths.pop()

def hamming_distance_from_sequences(sequences):
    """
    Computes the normalized Hamming distance matrix from aligned sequences.

    Parameters
    ----------
    sequences : dict
        Dictionary of {taxon: sequence} pairs.

    Returns
    -------
    np.ndarray
        Square matrix of normalized Hamming distances.
    """
    taxa = sorted(sequences.keys())
    seq_array = np.array([list(sequences[taxon]) for taxon in taxa])
    seq_array = seq_array.astype('<U1')  # ensure string dtype for comparisons
    dist_matrix = pdist(seq_array, metric=lambda x, y: np.mean(np.array(x) != np.array(y)))
    return squareform(dist_matrix)


def bootstrap_hamming_distances(sequences, n_bootstraps):
    """
    Bootstraps Hamming distance matrices by resampling sequence columns.

    Parameters
    ----------
    sequences : dict
        Dictionary of aligned sequences.
    n_bootstraps : int
        Number of bootstrap replicates.

    Returns
    -------
    np.ndarray
        Array of shape (n_bootstraps, N, N) containing distance matrices.
    """
    taxa = sorted(sequences.keys())
    seq_array = np.array([list(sequences[taxon]) for taxon in taxa])
    seq_array = seq_array.astype('<U1')
    n, C = seq_array.shape
    dist_matrix = np.zeros((n_bootstraps, n, n))

    for i in tqdm(range(n_bootstraps), desc="Bootstrapping"):
        col_indices = np.random.choice(C, size=C, replace=True)
        D = pdist(seq_array[:, col_indices], metric=lambda x, y: np.mean(np.array(x) != np.array(y)))
        dist_matrix[i, :, :] = squareform(D)

    return dist_matrix


def get_tree_from_dissimilarities(dist_matrix, labels=None):
    """
    Constructs a UPGMA tree from a distance matrix using DendroPy.

    Parameters
    ----------
    dist_matrix : np.ndarray
        Square distance matrix.
    labels : list, optional
        Taxa labels corresponding to rows/columns.

    Returns
    -------
    dendropy.Tree
        UPGMA tree.
    """

    #Avoid having long terminal branches
    nonzero = dist_matrix > 0
    min_nonzero = dist_matrix[nonzero].min()

    N = dist_matrix.shape[0]
    if labels is None:
        labels = [chr(ord('a') + i) for i in range(N)]

    taxa = dendropy.TaxonNamespace(labels)

    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp:
        np.savetxt(tmp.name, dist_matrix - min_nonzero, delimiter=",")
        tmp.seek(0)
        pdm = dendropy.PhylogeneticDistanceMatrix.from_csv(
            tmp,
            taxon_namespace=taxa,
            is_first_row_column_names=False,
            is_first_column_row_names=False,
            is_allow_new_taxa=True,
            delimiter=","
        )
    os.remove(tmp.name)
    return pdm.upgma_tree()


def save_trees(trees, output_filepath="./trees.tre"):
    """
    Saves a list of DendroPy trees to a file.

    Parameters
    ----------
    trees : list of dendropy.Tree
        Trees to save.
    output_filepath : str
        Destination file path.
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
        Path to FASTA input.
    output_filepath : str
        Output path for the inferred tree.

    Returns
    -------
    dendropy.Tree
        UPGMA-inferred tree.
    np.ndarray
        Pairwise distance matrix.
    """
    sequences = sequences_from_fasta(fasta_filepath)
    labels = sorted(sequences.keys())
    dist_matrix = hamming_distance_from_sequences(sequences)
    tree = get_tree_from_dissimilarities(dist_matrix, labels)
    save_trees([tree], output_filepath)
    return tree, dist_matrix


def metal_bootstrap_estimators(fasta_filepath, n_bootstraps, output_filepath="./boot_trees_metal.tre"):
    """
    Generates bootstrap replicate trees via Hamming distance resampling.

    Parameters
    ----------
    fasta_filepath : str
        Path to aligned FASTA sequences.
    n_bootstraps : int
        Number of replicates.
    output_filepath : str
        Output path for trees.

    Returns
    -------
    list of dendropy.Tree
        List of bootstrap trees.
    np.ndarray
        Array of distance matrices (n_bootstraps, N, N).
    """
    sequences = sequences_from_fasta(fasta_filepath)
    labels = sorted(sequences.keys())
    dist_matrices = bootstrap_hamming_distances(sequences, n_bootstraps)
    trees = [get_tree_from_dissimilarities(mat, labels) for mat in dist_matrices]
    save_trees(trees, output_filepath)
    return trees, dist_matrices


def multivariate_normal_bootstrap_estimators(fasta_filepath, n_bootstraps, output_filepath="./boot_trees_mvt.tre", sites_per_gene = 100, **kwargs):
    """
    Generates trees by sampling distance matrices from a multivariate normal distribution.

    Parameters
    ----------
    fasta_filepath : str
        Path to aligned FASTA file.
    n_bootstraps : int
        Number of bootstrap replicates.
    output_filepath : str
        Output path for trees.
    kwargs : dict
        Additional arguments passed to `compute_covariance_matrix`.

    Returns
    -------
    list of dendropy.Tree
        Bootstrap trees generated from MVN samples.
    np.ndarray
        Sampled distance matrices of shape (n_bootstraps, N, N).
    """
    tmp_file = tempfile.NamedTemporaryFile(delete=False)
    tmp_file.close()
    _, dist_matrix = metal_estimator(fasta_filepath, output_filepath=tmp_file.name)
    metal_tree = Phylo.read(tmp_file.name, "newick")
    os.remove(tmp_file.name)
    labels = [term.name for term in metal_tree.get_terminals()]
    labels.sort()

    N = dist_matrix.shape[0]
    sigma, _, _ = compute_covariance_matrix(metal_tree, sites_per_gene = sites_per_gene, **kwargs)
    n_bases = get_sequence_length(fasta_filepath)
    sigma *= sites_per_gene / n_bases
    mu = dist_matrix[np.triu_indices(N, k=1)]
    samples = np.random.multivariate_normal(mu, sigma, size=n_bootstraps)

    def vector_to_matrix(v):
        mat = np.zeros((N, N))
        i, j = np.triu_indices(N, k=1)
        mat[i, j] = mat[j, i] = v
        return mat

    trees = [get_tree_from_dissimilarities(vector_to_matrix(sample), labels) for sample in samples]
    save_trees(trees, output_filepath)
    return trees, samples
