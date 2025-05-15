from collections import defaultdict
import dendropy


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
