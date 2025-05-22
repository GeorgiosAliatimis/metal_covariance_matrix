from metal import Metal 
from utils.treetools import compute_split_frequencies, get_bipartitions
from data_generation import TreeSimulator, SequenceSimulator
import random
import os 
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import numpy as np
from utils.experiment_decorators import experiment_runner, run_parameter_sweep


@experiment_runner(
    num_runs=30,
    columns=["bootstrapping", "gaussian"]
)
def auc_comparison(num_gene_trees: int = 100,
    ntax: int = 20,
    num_sites_per_gene: int = 100,
    mutation_rate: float = 1.0,
    species_tree_diameter: float = 1.0,
    seed: int = 0
) -> tuple[float,float]:
    rng = random.Random(seed)

    tree_sim = TreeSimulator(ntax=ntax, tree_diameter=species_tree_diameter, rng=rng)

    # Generate species and gene trees
    tree_sim.generate_species_tree()
    gene_trees = tree_sim.generate_multiple_gene_trees(num_gene_trees)

    #Generate a fasta file of the concatenated sequences of gene trees

    seq_sim = SequenceSimulator(seq_length=num_sites_per_gene, mutation_rate=mutation_rate, seed=seed)

    sequences = seq_sim.simulate_concatenated(gene_trees)

    est = Metal(sequences = sequences, seed = seed)
    est.estimate_tree()

    boot_trees = est.bootstrap_hamming(n_bootstraps = 100)
    gauss_trees = est.gaussian_sampling(n_bootstraps = 100, mutation_rate=mutation_rate, sites_per_gene=num_sites_per_gene)

    boot_freqs = compute_split_frequencies(est.metal_tree, boot_trees)
    gauss_freqs = compute_split_frequencies(est.metal_tree, gauss_trees)

    true_splits = get_bipartitions(tree_sim.species_tree)

    def auc(freqs):
        splits = list(freqs.keys())
        y_true = [split in true_splits for split in splits]
        y_scores = [freqs[split] for split in splits]
        return roc_auc_score(y_true, y_scores)
    
    boot_score = auc(boot_freqs)
    gauss_score = auc(gauss_freqs)

    print(f"Auc for boot is {boot_score:.2f}")
    print(f"Auc for gauss is {gauss_score:.2f}")
    return boot_score, gauss_score

if __name__ == "__main__":
    default_params = {
        "num_gene_trees": 100,
        "ntax": 40,
        "num_sites_per_gene": 100,
        "mutation_rate": 1.0,
        "species_tree_diameter": 1.0,
    }
    
    parameter_sweep = {
        "num_gene_trees": [20,50,100,200,500],
        "ntax": [10,20,40,80],
        "num_sites_per_gene": [20,50,100,200,500],
        "mutation_rate": [.1,.5,1.0,2.0],
        "species_tree_diameter": [.2,.5,1.0,2.0],
    }
    log_filename = "logs.log"
    output_dir="aucs"

    run_parameter_sweep(
        func = auc_comparison,
        sweep = parameter_sweep,
        default_params = default_params,
        output_dir=output_dir,
        log_filename = log_filename
    )
    
    symbols = {
        "num_gene_trees": r"$m$",
        "ntax": r"$n$",
        "num_sites_per_gene": r"$K$",
        "mutation_rate": r"$\mu$",
        "species_tree_diameter": r"$\Delta$",
    }
    for param_name, param_values in parameter_sweep.items():
        x = np.arange(len(param_values))  
        width = 0.3  # spacing between bootstrapping and gaussian plots
        fig, ax = plt.subplots(figsize=(10, 6))
        symbol = symbols[param_name]
        for i, param_value in enumerate(param_values):
            if default_params[param_name] == param_value:
                filename = f"default.csv"
            else:
                filename = f"{param_name}_{param_value}.csv"
            data = np.loadtxt(os.path.join(output_dir,filename), delimiter= ",",skiprows=1)
            data = data[~np.isnan(data).any(axis=1)] #cleanup
            ax.boxplot(data[:,0], positions=[x[i] - width/2], widths=0.2,
               patch_artist=True, boxprops=dict(facecolor='lightblue'),
               medianprops=dict(color='blue'))
    
            ax.boxplot(data[:,1], positions=[x[i] + width/2], widths=0.2,
                    patch_artist=True, boxprops=dict(facecolor='lightgreen'),
                    medianprops=dict(color='green'))
        ax.set_xticks(x)
        ax.set_xticklabels(param_values)
        ax.set_xlabel(symbol)
        ax.set_ylabel("AUC")
        ax.set_title(f"AUC Boxplots against {symbol}")

        # Add legend
        ax.plot([], c='lightblue', label='Bootstrapping', linewidth=10)
        ax.plot([], c='lightgreen', label='Gaussian', linewidth=10)
        ax.legend()

        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir,f"{param_name}.png") )
