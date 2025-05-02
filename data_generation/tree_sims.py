"""
Tree Simulation Utilities

This module provides tools to simulate species trees under the Yule process and
generate gene trees from those species trees under the multispecies coalescent model.

Key functionalities:
- Generate species trees with a specified number of taxa and normalized depth.
- Generate gene trees based on a species tree and the coalescent process.
- Normalize trees so that the depth from root to leaves matches a specified value.

Dependencies:
- DendroPy
- NumPy
"""

import dendropy
from dendropy.simulate import treesim
import numpy as np


def scale_tree(tree, tree_depth=1):
    """
    Scales a phylogenetic tree so that the distance from the root to all leaves equals `tree_depth`.

    Parameters:
        tree (dendropy.Tree): The input tree to be scaled.
        tree_depth (float): Desired root-to-leaf distance after scaling.

    Returns:
        dendropy.Tree: A new UPGMA tree with distances scaled to `tree_depth`.
    """
    T = tree.taxon_namespace
    ntax = len(T)

    # Compute the original pairwise distance matrix
    D = tree.phylogenetic_distance_matrix()
    D = np.asarray([[D(i, j) for j in T] for i in T])

    # Normalize distances to [0, tree_depth]
    D /= D.max()
    D *= tree_depth

    # Create a distance dictionary for DendroPy
    distances = {T[i]: {T[j]: D[i, j] for j in range(ntax)} for i in range(ntax)}

    # Build a new tree using UPGMA from the scaled distances
    pdm = dendropy.PhylogeneticDistanceMatrix()
    pdm.compile_from_dict(distances=distances, taxon_namespace=T)
    return pdm.upgma_tree()


def generate_species_tree(ntax=10, normalize=True, tree_depth=1, rng=None):
    """
    Generates a species tree under the Yule (pure-birth) process.

    Parameters:
        ntax (int): Number of taxa (species) to simulate.
        normalize (bool): Whether to scale the tree to a fixed depth.
        tree_depth (float): Desired depth of the tree if `normalize` is True.
        rng (random.Random): Optional random number generator for reproducibility.

    Returns:
        dendropy.Tree: A species tree with `ntax` tips.
    """
    labels = [chr(ord('a') + i) for i in range(num_taxa)]
    taxa = dendropy.TaxonNamespace(labels)

    # Simulate a Yule tree
    sp_tree = treesim.birth_death_tree(
        birth_rate=1,
        death_rate=0,
        num_extant_tips=ntax,
        taxon_namespace=taxa,
        rng=rng
    )

    return scale_tree(sp_tree, tree_depth) if normalize else sp_tree


def generate_gene_tree_from_species_tree(sp_tree, normalize=True, rng=None):
    """
    Simulates a gene tree within a given species tree under the coalescent process.

    Parameters:
        sp_tree (dendropy.Tree): The species tree to simulate within.
        normalize (bool): Whether to scale the gene tree to a fixed depth.
        rng (random.Random): Optional random number generator for reproducibility.

    Returns:
        dendropy.Tree: A gene tree simulated from the given species tree.
    """
    # Map each species taxon to a unique gene taxon
    gene_to_species_map = dendropy.TaxonNamespaceMapping.create_contained_taxon_mapping(
        containing_taxon_namespace=sp_tree.taxon_namespace,
        num_contained=1,
        contained_taxon_label_fn=lambda x, _: str(x.label)
    )

    # Simulate the gene tree
    gene_tree = treesim.contained_coalescent_tree(
        containing_tree=sp_tree,
        gene_to_containing_taxon_map=gene_to_species_map,
        rng=rng,
        default_pop_size=1
    )

    return scale_tree(gene_tree) if normalize else gene_tree