# 🧬 Metal Covariance Matrix

A Python toolkit for simulating species and gene trees, generating DNA sequences, and estimating covariance matrices.

---

## 📦 Installation

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install the package in editable mode
pip install -e .
```

## ⚙️ Usage

### 1. Generate Simulated Data

If you don't have your own input data, you can generate species trees, gene trees, and DNA sequences using:

```bash
python3 data_generation.py
```

This will create a directory called `gene_data/` containing:
* `species_tree.nex`
* `gene_trees.nex`
* `concatenated_seq_alignment.fasta`

Otherwise, you need to have **concatenated** gene alignments as a fasta file.

### 2. Run Analysis
Once you have a concatenated alignment file (either generated or your own), run:
```bash
python3 run_analysis.py path_to_fasta_file
```
Replace path_to_fasta_file with the path to your concatenated sequence alignment file, for example:
```bash
python3 run_analysis.py gene_data/
```
This will produce ... 
