import dendropy
from dendropy.simulate import treesim
import numpy as np
from utils.treetools import rescale_tree_to_diameter, excel_style_labels

class TreeSimulator:
    """
    Simulates species and gene trees under the Yule process and multispecies coalescent model.

    Attributes:
        ntax (int): Number of taxa for the species tree.
        tree_diameter (float): Diameter to which species tree is scaled.
        rng (random.Random): Random number generator for reproducibility.
    """

    def __init__(self, ntax=10, tree_diameter=1.0, rng=None):
        self.ntax = ntax
        self.tree_diameter = tree_diameter
        self.rng = rng or np.random.default_rng()
        self.species_tree = None

    def generate_species_tree(self):
        """
        Generates a species tree under the Yule process.

        Returns:
            dendropy.Tree: A species tree with `ntax` tips.
        """
        labels = excel_style_labels(self.ntax)
        taxa = dendropy.TaxonNamespace(labels)
        self.species_tree = treesim.birth_death_tree(
            birth_rate=1,
            death_rate=0,
            num_extant_tips=self.ntax,
            taxon_namespace=taxa,
            rng=self.rng
        )
        rescale_tree_to_diameter(self.species_tree, self.tree_diameter)
        print("Species tree generated.")
        return self.species_tree

    def generate_gene_tree(self):
        """
        Simulates a gene tree from the species tree under the coalescent model.

        Returns:
            dendropy.Tree: A simulated gene tree.
        """
        if self.species_tree is None:
            print("No species tree available. Generating species tree...")
            self.generate_species_tree()

        gene_to_species = dendropy.TaxonNamespaceMapping.create_contained_taxon_mapping(
            containing_taxon_namespace=self.species_tree.taxon_namespace,
            num_contained=1,
            contained_taxon_label_fn=lambda x, _: str(x.label)
        )

        gene_tree = treesim.contained_coalescent_tree(
            containing_tree=self.species_tree,
            gene_to_containing_taxon_map=gene_to_species,
            rng=self.rng,
            default_pop_size=1
        )

        return gene_tree

    def generate_multiple_gene_trees(self, n_trees):
        """
        Generates multiple gene trees from the species tree.

        Args:
            n_trees (int): Number of gene trees to generate.

        Returns:
            dendropy.TreeList: A list of gene trees.
        """
        gene_trees = dendropy.TreeList([
            self.generate_gene_tree()
            for _ in range(n_trees)
        ])
        print("Gene trees generated.")
        return gene_trees

    def reset(self):
        """
        Resets the internal state of the simulator.
        Clears the stored species tree.
        """
        self.species_tree = None


    def get_species_tree(self):
        """
        Returns the internally stored species tree.

        Returns:
            dendropy.Tree or None: The current species tree, or None if not set.
        """
        return self.species_tree
