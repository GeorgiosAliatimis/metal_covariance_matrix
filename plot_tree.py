import io 
from Bio import Phylo 
import matplotlib.pyplot as plt
import dendropy

def plot_dendropy_tree(tree, **kwargs):
    newick_tree = tree.as_string("newick")
    handle = io.StringIO(newick_tree)
    tree = Phylo.read(handle, "newick")
    Phylo.draw(tree, **kwargs)

def plot_tree_file(tree_filepath, schema, **kwargs):
    tree = dendropy.Tree.get(path=tree_filepath,schema=schema)
    plot_dendropy_tree(tree, **kwargs)


def plot_tree_with_support(tree_filepath, axes = None):
    tree = Phylo.read(tree_filepath, "newick")
    if axes is None:
        fig, axes = plt.subplots(figsize=(10, 10))
    
    # Plot the tree without showing it yet
    Phylo.draw(tree, do_show=False, axes=axes)

    depths = tree.depths()
    
    # Annotate the branches with support values (assuming support is stored in .confidence)
    for clade in tree.find_clades():
        if hasattr(clade, 'confidence') and clade.confidence is not None:  # Check if there's a confidence score
            # In Phylo.draw(), the horizontal position is given by the branch length (x)
            # We can calculate the y-coordinate manually if needed, or use a fixed value for simplicity
            if clade.branch_length is not None:
                x = clade.branch_length
            else:
                x = 0  # If no branch length is available, use a default value (0 or other)
            
            # Use tree.depths() to calculate the vertical position (y) of the clade
            y = depths.get(clade,0)  # Call the depths method to get the vertical position
            
            # Annotate with branch length and confidence value
            axes.annotate(f'{clade.confidence:.2f}', xy=(x, y), 
                        xytext=(x, y),  # Adjust the position of the text annotation
                        arrowprops=dict(facecolor='black', arrowstyle="->"))
        else:
            # Optionally, annotate with a default value if confidence is None
            axes.annotate('No support', xy=(x, y), 
                        xytext=(x, y), 
                        arrowprops=dict(facecolor='black', arrowstyle="->"))
    
