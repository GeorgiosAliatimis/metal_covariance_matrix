from Bio import Phylo
from collections import defaultdict
from io import StringIO
from plot_tree import *

def get_bipartitions(tree):
    """
    Given a tree in Newick format, extract its bipartitions.
    A bipartition is a set of taxa that are split by an internal branch.
    """
    bipartitions = set()
    
    # Iterate through the branches of the tree and compute splits
    for clade in tree.find_clades():
        if clade.is_terminal():  # Skip leaf nodes
            continue
        
        # Get the bipartition of the tree rooted at this node (clade)
        partition = sorted([leaf.name for leaf in clade.get_terminals()])
        bipartitions.add(tuple(partition))  # Store the partition as a sorted tuple
    
    return bipartitions

def compute_split_frequencies(species_tree_filepath, bootstrap_trees_filepath, output_filepath = ""):
    # Parse the species tree
    species_tree_obj = Phylo.read(species_tree_filepath, "newick")

    # Load the bootstrap trees from the file
    bootstrap_trees = []
    with open(bootstrap_trees_filepath, "r") as f:  # Path to your bootstrap trees file
        bootstrap_trees = f.read().strip().split(';')  # Split trees by semicolon (assuming one tree per line)

    # Remove empty strings from the bootstrap trees list (if any)
    bootstrap_trees = [tree.strip() + ";" for tree in bootstrap_trees if tree.strip()]
    
    # Get the bipartitions from the species tree
    species_bipartitions = get_bipartitions(species_tree_obj)
    
    # Initialize a dictionary to store frequencies for each bipartition in the species tree
    bipartition_counts = defaultdict(int)
    
    # Iterate over bootstrap trees to compare their bipartitions to the species tree
    for bt in bootstrap_trees:
        # Use StringIO to treat the bootstrap tree string as a file-like object
        bt_tree_obj = Phylo.read(StringIO(bt), "newick")
        bt_bipartitions = get_bipartitions(bt_tree_obj)
        
        # Count how many bipartitions from the species tree appear in this bootstrap tree
        for bipartition in species_bipartitions:
            if bipartition in bt_bipartitions:
                bipartition_counts[bipartition] += 1
    
    # Normalize the counts to frequencies (divide by the number of bootstrap trees)
    total_trees = len(bootstrap_trees)
    frequencies = {bipartition: count / total_trees for bipartition, count in bipartition_counts.items()}
    
    for k, v in frequencies.items():
        print(f"{k}: {v}")
    # Now assign these frequencies as support values to the branches of the species tree
    for clade in species_tree_obj.find_clades():
        if clade.is_terminal():
            continue
        
        # Get the bipartition of the current internal clade
        partition = sorted([leaf.name for leaf in clade.get_terminals()])
        bipartition = tuple(partition)
        
        if bipartition in frequencies:
            clade.confidence = frequencies[bipartition]  # Store the frequency as the support value
    
    if output_filepath:
        Phylo.write(species_tree_obj, output_filepath, "newick")
    return species_tree_obj, frequencies
