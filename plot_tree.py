import io 
from Bio import Phylo 
import matplotlib.pyplot as plt

def plot_dendropy_tree(tree):
    newick_tree = tree.as_string("newick")
    handle = io.StringIO(newick_tree)
    tree = Phylo.read(handle, "newick")
    Phylo.draw(tree)


def plot_tree_with_support(tree_filepath):
    tree = Phylo.read(tree_filepath, "newick")
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot the tree without showing it yet
    Phylo.draw(tree, do_show=False, axes=ax)
    
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
            y = tree.depths(clade)  # Call the depths method to get the vertical position
            
            # Annotate with branch length and confidence value
            ax.annotate(f'{clade.confidence:.2f}', xy=(x, y), 
                        xytext=(x + 0.1, y),  # Adjust the position of the text annotation
                        arrowprops=dict(facecolor='black', arrowstyle="->"))
        else:
            # Optionally, annotate with a default value if confidence is None
            ax.annotate('No support', xy=(x, y), 
                        xytext=(x + 0.1, y), 
                        arrowprops=dict(facecolor='black', arrowstyle="->"))
    
    # Show the plot with the annotations
    plt.show()
