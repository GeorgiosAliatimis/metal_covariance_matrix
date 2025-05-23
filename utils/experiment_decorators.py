import os 
import sys
import logging
from tqdm import tqdm
import csv
from contextlib import contextmanager
from functools import wraps
from typing import Callable


def experiment_runner(
    num_runs: int = 100,
    columns: list[str] = None,
) -> Callable:

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> None:
            @contextmanager
            def suppress_output():
                with open(os.devnull, "w") as devnull:
                    old_stdout, old_stderr = sys.stdout, sys.stderr
                    sys.stdout = sys.stderr = devnull
                    try:
                        yield
                    finally:
                        sys.stdout, sys.stderr = old_stdout, old_stderr

            filename = kwargs.pop("filename", None)
            if not filename:
                raise ValueError("Missing required 'filename' argument when calling the function.")
            
            output_dir = kwargs.pop("output_dir", ".")
            print(output_dir)
             
            filepath = os.path.join(output_dir, filename)
            os.makedirs(output_dir, exist_ok=True)

            def count_existing_rows(filepath: str) -> int:
                if not os.path.exists(filepath):
                    return 0
                with open(filepath, "r", newline="") as f:
                    return max(0, sum(1 for _ in f) - 1)  # exclude header

            file_exists = os.path.exists(filepath)
            start_seed = count_existing_rows(filepath)

            if start_seed >= num_runs:
                logging.info(f"Skipping {filepath} — already has {start_seed} runs.")
                return

            n_exceptions = 0

            for seed in tqdm(range(start_seed, num_runs), desc=f"Running {num_runs - start_seed} experiments"):
                with open(filepath, "a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    if not file_exists:
                        if columns:
                            writer.writerow(columns)
                        else:
                            raise ValueError("CSV header 'columns' must be specified.")
                        file_exists = True

                    kwargs["seed"] = seed
                    try:
                        with suppress_output():
                            result = func(*args, **kwargs)
                    except Exception as e:
                        n_exceptions += 1
                        result = tuple(float("nan") for _ in columns)
                        logging.error(f"Error for {kwargs}, seed={seed}: {e}", exc_info=True)

                    writer.writerow(list(result))

            logging.info(f"Completed experiment with {n_exceptions} exceptions.")

        return wrapper
    return decorator

def run_parameter_sweep(
    func: Callable,
    sweep: dict[str, list],
    default_params: dict[str, int | float | str],
    output_dir: str = None,
    log_filename: str = None
) -> None:
    """
        Changes one parameter at a time while keeping all other
        experimental parameters as default_params.
    """
    os.makedirs(output_dir, exist_ok=True)

    if log_filename is not None:
        logging.basicConfig(
            filename=os.path.join(output_dir, log_filename),   
            level=logging.INFO,          
            format="%(asctime)s %(levelname)s: %(message)s"
        )

    func(filename="default.csv",output_dir = output_dir, **default_params)
    for param_name, param_values in sweep.items():
        for param_value in param_values:
            if default_params[param_name] == param_value:
                continue
            params = default_params.copy()
            params[param_name] = param_value

            filename = f"{param_name}_{param_value}.csv"
            logging.info(f"Starting experiment for {param_name}={param_value}.")

            func(filename=filename,output_dir = output_dir, **params)

