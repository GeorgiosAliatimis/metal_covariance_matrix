import dendropy
import numpy as np
from io import StringIO
from Bio import Phylo
from collections import defaultdict

def tree_from_distance_matrix(D, labels = None):
    """
    Constructs a phylogenetic tree from a pairwise distance matrix using the UPGMA algorithm.

    Parameters
    ----------
    D : np.ndarray
        A 2D NumPy array of shape (N, N) representing the symmetric pairwise distance matrix
        between N taxa.
    labels : list of str, optional
        A list of N labels corresponding to the taxa. If not provided, taxa will be labeled
        as stringified indices (e.g., "0", "1", ..., "N-1").

    Returns
    -------
    tree : dendropy.Tree
        A rooted phylogenetic tree inferred using the UPGMA method, with branch lengths
        corresponding to the input distances.
    """
    if labels is None:
        labels = [chr(ord('a') + i) for i in range(N)]
    labels.sort()
    T = dendropy.TaxonNamespace(labels)
    distances = {
        T[i]: {
            T[j]: D[i, j]
            for j in range(len(T))
        }
        for i in range(len(T))
    }
    pdm = dendropy.PhylogeneticDistanceMatrix()
    pdm.compile_from_dict(distances=distances, taxon_namespace=T)
    return pdm.upgma_tree()

def distance_matrix_from_tree(tree):
    """
    Computes the pairwise distance matrix between all leaf nodes in the given tree.

    Parameters
    ----------
    tree : dendropy.Tree
        A rooted or unrooted phylogenetic tree with branch lengths defined.

    Returns
    -------
    dist_matrix : np.ndarray
        A 2D NumPy array of shape (N, N), where N is the number of leaf nodes (taxa) in the tree.
        Each entry [i, j] represents the path length (tree distance) between the i-th and j-th
        leaf, based on sorted leaf labels.

    Notes
    -----
    The order of rows and columns corresponds to the lexicographically sorted list of leaf labels.
    This ensures consistent indexing across trees.
    """
    pdm = tree.phylogenetic_distance_matrix()
    taxa = sorted(tree.taxon_namespace, key=lambda x: x.label)
    N = len(taxa)
    dist_matrix = np.zeros((N,N))
    for i in range(N):
        taxon1 = taxa[i]
        for j in range(i+1,N):
            taxon2 = taxa[j]
            dist_matrix[i,j] = pdm(taxon1, taxon2)
            dist_matrix[j,i] = dist_matrix[i,j]
    return dist_matrix

def transform_hamming_to_coalescent_distances(hd_mat, mutation_rate):
    """
    Transforms a matrix of normalized Hamming distances into coalescent distances,
    given a mutation rate.

    Parameters
    ----------
    hd_mat : np.ndarray
        A square matrix of shape (N, N) representing pairwise normalized Hamming distances
        between N sequences or taxa. Should be symmetric with zeros on the diagonal.

    mutation_rate : float
        The mutation rate per unit time used to convert Hamming distances into
        coalescent distances.

    Returns
    -------
    np.ndarray
        A square matrix of the same shape as `hd_mat`, where each element is the
        corresponding coalescent distance calculated from the input distances and
        mutation rate.
    """
    if not np.all((hd_mat >= 0) & (hd_mat <= 1)):
        raise ValueError("All elements of hd_mat must be within the range [0, 1].")

    coal_mat = np.zeros_like(hd_mat)

    coal_mat[hd_mat >= 0.75] = float("inf")
    
    mask = hd_mat < 0.75
    coal_mat[mask] = -(3/4 / mutation_rate) * (
        np.log(1 + (8/3)*mutation_rate) + np.log(1 - (4/3)*hd_mat[mask])
    )

    # Ensure no negative values due to numerical imprecision
    coal_mat[coal_mat < 0] = 0

    return coal_mat

def save_trees(trees, output_path, schema):
    tree_list = dendropy.TreeList(taxon_namespace=trees[0].taxon_namespace)
    tree_list.extend(trees)
    tree_list.write(path=output_path, schema=schema, suppress_rooting=True)

def rescale_tree_to_diameter(tree, target_diameter = 1):
    """Scales tree so its diameter (=2 * root-to-tip) equals target_diameter."""
    tree_diameter = 2 * tree.max_distance_from_root()
    scale_tree(tree, 1/tree_diameter)

def scale_tree(tree, factor):
    """Multiplies all branch lengths by a constant factor."""
    for node in tree.preorder_node_iter():
        if node.edge_length is not None:
            node.edge_length *= factor

def normalize_terminal_branch_lengths(tree):
    """Subtracts the minimum terminal branch length from all terminal branches."""
    min_terminal_edge_lengths = min(
        node.edge.length
        for node in tree.leaf_node_iter()
        if node.edge.length is not None
    )
    for node in tree.leaf_node_iter():
        node.edge_length -= min_terminal_edge_lengths


def convert_to_phylo_tree(tree):
    module = tree.__class__.__module__
    if module.startswith("Bio.Phylo"):
        # Already a Phylo tree
        pass
    elif module.startswith("dendropy"):
        # Dendropy tree converted to Phylo tree
        newick_tree = tree.as_string("newick")
        handle = StringIO(newick_tree)
        tree = Phylo.read(handle, "newick")
    else:
        raise TypeError(f"Unknown tree type: {type(tree)} from module {module}")
    
    return tree

def convert_to_dendropy_tree(tree):
    module = tree.__class__.__module__
    if module.startswith("Bio.Phylo"):
        #Convert Phylo tree to Dendropy
        newick_str = StringIO()
        Phylo.write(tree, newick_str, "newick")
        newick_str.seek(0)
        tree = dendropy.Tree.get(file=newick_str, schema="newick")
    elif module.startswith("dendropy"):
        # Already dendropy
        pass
    else:
        raise TypeError(f"Unknown tree type: {type(tree)} from module {module}")
    
    return tree

def get_bipartitions(tree):
    """
    Given a DendroPy Tree, return the set of bipartitions defined by its internal nodes.
    Each bipartition is represented as a frozenset of taxon labels on one side of the cut.
    """
    biparts = set()
    for node in tree.internal_nodes():
        # skip the root if you don't want the trivial bipartition of all taxa
        if node == tree.seed_node:
            continue
        # collect labels in the clade below this node
        labels = {leaf.taxon.label for leaf in node.leaf_nodes()}
        biparts.add(frozenset(labels))
    return biparts


def compute_split_frequencies(
    species_tree: dendropy.Tree,
    bootstrap_trees: dendropy.TreeList,
) -> dendropy.Tree:
    """
    Parameters
    ----------
    species_tree : dendropy.Tree
        A rooted, ultrametric species tree (no support values yet).
    bootstrap_trees : dendropy.TreeList
        A list of bootstrap replicate trees.

    Returns
    -------
    dendropy.Tree
        The same species_tree object, with each internal node.label set to
        its bootstrap support frequency [0,1].
    """

    # Ensure that species_tree is a deep copy of original species_tree
    species_tree = species_tree.clone(depth=2)
    species_tree.encode_bipartitions()  # Ensure bipartitions are tracked

    # 1) Compute species bipartitions
    species_bips = get_bipartitions(species_tree)

    # 2) Count occurrences
    counts = defaultdict(int)
    for bt in bootstrap_trees:
        bt_bips = get_bipartitions(bt)
        for bip in species_bips:
            if bip in bt_bips:
                counts[bip] += 1

    total = len(bootstrap_trees)
    freqs = {bip: counts[bip] / total for bip in species_bips}

    # 3) Annotate species_tree internal nodes
    for node in species_tree.internal_nodes():
        if node == species_tree.seed_node:
            continue
        labels = frozenset(leaf.taxon.label for leaf in node.leaf_nodes())
        node.label = f"{freqs.get(labels, 0.0):.3f}"

    return species_tree
