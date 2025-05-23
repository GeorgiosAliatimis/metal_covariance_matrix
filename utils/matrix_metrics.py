import numpy as np
import logging

def matrices_comparison(mat1, mat2):
    """
    Compare two matrices using various metrics.

    Parameters:
        mat1 (np.ndarray): First square matrix.
        mat2 (np.ndarray): Second square matrix.

    Returns:
        dict: Dictionary with trace, logdet, Frobenius norm, and spectral norm ratios.
    """
    def logdet(matrix):
        sign, ld = np.linalg.slogdet(matrix)
        if sign <= 0:
            logging.warning("Matrix must be positive-definite for log-determinant.")
            return float("nan")
        return ld
    
    def frobenius_norm(matrix):
        return np.linalg.norm(matrix, ord='fro')
    
    def spectral_norm(matrix):
        return np.linalg.norm(matrix, ord=2)
    
    if mat1.shape != mat2.shape:
        raise ValueError("Both matrices must have the same shape.")
    if mat1.shape[0] != mat1.shape[1]:
        raise ValueError("Both matrices must be square.")

    metrics = {
        "trace_ratio": mat1.trace() / mat2.trace(),
        "logdet_ratio": logdet(mat1) / logdet(mat2),
        "frobenius_norm_ratio": frobenius_norm(mat1) / frobenius_norm(mat2),
        "spectral_norm_ratio": spectral_norm(mat1) / spectral_norm(mat2)
    }

    return metrics
