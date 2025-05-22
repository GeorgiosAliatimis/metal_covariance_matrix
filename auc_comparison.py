from metal import Metal 
from utils.treetools import save_trees, compute_split_frequencies, get_bipartitions
from data_generation import TreeSimulator, SequenceSimulator
import random
import os 
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import numpy as np

import sys
import logging
from tqdm import tqdm
import csv
from contextlib import contextmanager

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

def run_experiments(
    params: dict[str, int | float | str],
    num_runs: int = 100,
    output_dir: str = ".",
    filename: str = "aucs.csv"
) -> None:

    @contextmanager
    def suppress_output():
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = devnull
            sys.stderr = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

    def count_existing_rows(filepath: str) -> int:
        if not os.path.exists(filepath):
            return 0
        with open(filepath, "r", newline="") as f:
            return max(0, sum(1 for _ in f) - 1)  # exclude header

    os.makedirs(output_dir, exist_ok=True)

    filepath = os.path.join(output_dir, filename)
    file_exists = os.path.exists(filepath)
    start_seed = count_existing_rows(filepath)

    if start_seed >= num_runs:
        logging.info(f"Skipping {filepath} — already has {start_seed} runs.")
        return

    n_exceptions = 0

    for seed in tqdm(range(start_seed, num_runs), desc = f"Running {num_runs - start_seed } experiments: "):
        with open(filepath, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(["bootstrapping", "gaussian"])  # header
                file_exists= True

            params["seed"] = seed
            try:
                with suppress_output():
                    result = auc_comparison(**params)
            except Exception as e:
                n_exceptions += 1
                result = ( float("nan"), float("nan") )
                logging.error(f"Error for {params}, seed={seed}: {e}", exc_info=True)
            writer.writerow(list(result))

    logging.info(f"Completed experiment with {n_exceptions} exceptions.")

if __name__ == "__main__":
    default_vals = {
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
    n_runs = 30
    output_dir = "aucs"

    logging.basicConfig(
        filename=os.path.join(output_dir, "experiments.log"),   
        level=logging.INFO,          
        format="%(asctime)s %(levelname)s: %(message)s"
    )

    run_experiments(
        params=default_vals,
        num_runs=n_runs,
        output_dir = output_dir,
        filename = "default.csv"
    )

    for param_name, param_values in parameter_sweep.items():
        for param_value in param_values:
            if default_vals[param_name] == param_value:
                continue
            params = default_vals.copy()
            params[param_name] = param_value

            filename = f"{param_name}_{param_value}.csv"

            logging.info(f"Starting experiment for {param_name}={param_value}.")
            run_experiments(
                params=params,
                num_runs=n_runs,
                output_dir = output_dir,
                filename = filename
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
            if default_vals[param_name] == param_value:
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


            
            
            