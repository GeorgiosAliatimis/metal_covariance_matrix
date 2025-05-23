from tree_estimators import Metal, Glass 
from data_generation import TreeSimulator, SequenceSimulator
import random
from utils.treetools import rf_distance
from utils.matrix_metrics import matrices_comparison
from tree_estimators.covariance_matrix import compute_covariance_matrix_from_tree
from utils.experiment_decorators import experiment_runner, run_parameter_sweep

@experiment_runner(
    num_runs = 100,
    columns = [
        "rf_metal",
        "rf_glass", 
        'trace_ratio', 
        'logdet_ratio', 
        'frobenius_norm_ratio', 
        'spectral_norm_ratio'
    ]
)
def estimation(num_gene_trees: int = 100,
    ntax: int = 20,
    num_sites_per_gene: int = 100,
    mutation_rate: float = 1.0,
    species_tree_diameter: float = 1.0,
    seed: int = 0
) -> tuple[float,float,float,float,float,float]:
    rng = random.Random(seed)

    tree_sim = TreeSimulator(ntax=ntax, tree_diameter=species_tree_diameter, rng=rng)

    # Generate species and gene trees
    tree_sim.generate_species_tree()
    gene_trees = tree_sim.generate_multiple_gene_trees(num_gene_trees)

    #Generate a fasta file of the concatenated sequences of gene trees

    seq_sim = SequenceSimulator(seq_length=num_sites_per_gene, mutation_rate=mutation_rate, seed=seed)

    sequences = seq_sim.simulate_concatenated(gene_trees)

    metal = Metal(sequences = sequences)
    metal.estimate_tree()
    rf_metal = rf_distance(metal.tree, tree_sim.species_tree, normalize= True)
    

    glass = Glass(sequences = sequences, sites_per_gene= num_sites_per_gene)
    glass.estimate_tree()
    rf_glass = rf_distance(glass.tree, tree_sim.species_tree, normalize= True)

    sigma_coal = compute_covariance_matrix_from_tree(
        tree = tree_sim.species_tree,
        sites_per_gene=num_sites_per_gene, 
        mutation_rate = mutation_rate,
        mode = "coal"
    )
    sigma_total = compute_covariance_matrix_from_tree(
        tree = tree_sim.species_tree,
        sites_per_gene=num_sites_per_gene, 
        mutation_rate = mutation_rate,
        mode = "total"
    )
    coal_uncertainty_ratio = matrices_comparison(sigma_coal, sigma_total)
    coal_uncertainty_ratio = list(coal_uncertainty_ratio.values())
    return rf_metal,rf_glass, *coal_uncertainty_ratio
    


if __name__ == "__main__":
    default_params = {
        "num_gene_trees": 100,
        "ntax": 10,
        "num_sites_per_gene": 100,
        "mutation_rate": 1.0,
        "species_tree_diameter": 1.0,
    }
    
    parameter_sweep = {
        "num_gene_trees": [10,20,50,100,200],
        "ntax": [5,10,20,40],
        "num_sites_per_gene": [20,50,100,200,500,1000],
        "mutation_rate": [.1,.5,1.0,2.0],
        "species_tree_diameter": [.2,.5,1.0,2.0],
    }
    log_filename = "logs.log"
    output_dir="estimation"

    run_parameter_sweep(
        func = estimation,
        sweep = parameter_sweep,
        default_params = default_params,
        output_dir=output_dir,
        log_filename = log_filename
    )