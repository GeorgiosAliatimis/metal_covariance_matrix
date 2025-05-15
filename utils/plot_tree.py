from Bio import Phylo 
import matplotlib.pyplot as plt
import dendropy
from .treetools import convert_to_dendropy_tree, convert_to_phylo_tree, normalize_terminal_branch_lengths

def plot_tree(tree, ax = None, normalize = True):
    if normalize:
        tree = convert_to_dendropy_tree(tree)
        normalize_terminal_branch_lengths(tree)
    tree = convert_to_phylo_tree(tree)
    do_show = ax is None
    for clade in tree.find_clades():
        if not clade.is_terminal() and clade.confidence is not None:
            clade.name = f"{100 * clade.confidence:.0f}%"        
    Phylo.draw(tree, show_confidence = False, do_show = do_show, axes = ax)
    if ax:
        ax.set_ylabel("")             
        ax.set_yticks([]) 
        ax.spines['left'].set_visible(False)
        ax.yaxis.set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)   

def plot_tree_from_file(path, schema, **kwargs):
    tree = Phylo.read(path, schema)
    plot_tree(tree, **kwargs)

def print_tree(tree, show_support=True, show_lengths=True):
    tree = convert_to_dendropy_tree(tree)
    print_tree_helper(tree.seed_node, show_support = show_support, show_lengths=show_lengths)
    
def print_tree_helper(node, indent="", is_last=True, show_support=True, show_lengths=True):
    """
    Recursively prints a DendroPy tree in ASCII format.

    Parameters:
        node (dendropy.Node): The current node.
        indent (str): String used for indenting the current level.
        is_last (bool): Whether this node is the last child.
        show_support (bool): Whether to show support values on internal nodes.
        show_lengths (bool): Whether to show edge (branch) lengths.
    """
    connector = "└── " if is_last else "├── "

    # Build label
    if node.is_leaf():
        label = node.taxon.label
    else:
        if show_support and node.label:
            label = f"[{node.label}]"
        else:
            label = "*"

    # Add branch length if enabled
    if show_lengths and node.edge_length is not None:
        label += f" :{node.edge_length:.5f}"
        
    print(indent + connector + label)

    # Prepare next indentation
    indent += "    " if is_last else "│   "

    # Recurse on children
    children = node.child_nodes()
    for i, child in enumerate(children):
        print_tree_helper(child, indent, is_last=(i == len(children) - 1),
                       show_support=show_support, show_lengths=show_lengths)
