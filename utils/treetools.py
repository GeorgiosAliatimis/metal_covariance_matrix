import dendropy

def tree_from_distance_matrix(D, labels = None):
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

def save_trees(trees, output_path, schema):
    tree_list = dendropy.TreeList(taxon_namespace=trees[0].taxon_namespace)
    tree_list.extend(trees)
    tree_list.write(path=output_path, schema=schema, suppress_rooting=True)

