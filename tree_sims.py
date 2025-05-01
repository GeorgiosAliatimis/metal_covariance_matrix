import dendropy
from dendropy.simulate import treesim
import string
import numpy as np


def scale_tree(tree, tree_depth=1):
    # Scales tree so that the distance from root to leaves is tree_depth.
    T = tree.taxon_namespace
    ntax = len(T)
    D = tree.phylogenetic_distance_matrix()
    D = np.asarray([[D(i,j)for j in T] for i in T ])
    D /= D.max() 
    D *= tree_depth
    distances = {T[i]:{T[j]: D[i,j] for j in range(ntax)} for i in range(ntax)}
    pdm = dendropy.PhylogeneticDistanceMatrix()
    pdm.compile_from_dict(distances=distances,taxon_namespace=T)
    return pdm.upgma_tree()


def generate_species_tree(ntax = 10,normalize= True, tree_depth=1,rng=None):
    # Generate species tree with specified tree_depth under Yule process.
    labels = string.ascii_lowercase[:ntax]
    taxa = dendropy.TaxonNamespace(labels)
    sp_tree = treesim.birth_death_tree(birth_rate=1, 
                                death_rate=0, 
                                num_extant_tips=ntax,
                                taxon_namespace=taxa,
                                rng = rng)
    return scale_tree(sp_tree,tree_depth) if normalize else sp_tree

def generate_gene_tree_from_species_tree(sp_tree, normalize = True, rng=None):
    gene_to_species_map = dendropy.TaxonNamespaceMapping.create_contained_taxon_mapping(
            containing_taxon_namespace=sp_tree.taxon_namespace,
            num_contained=1, contained_taxon_label_fn = lambda x,_: str(x.label) )
    gene_tree = treesim.contained_coalescent_tree(containing_tree=sp_tree,
                                            gene_to_containing_taxon_map=gene_to_species_map, 
                                            rng=rng, 
                                            default_pop_size = 1)
    return scale_tree(gene_tree) if normalize else gene_tree
