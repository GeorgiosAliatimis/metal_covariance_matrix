import itertools
import numpy as np
from utils.treetools import distance_matrix_from_tree

def order_triplet(triplet, tree_distance, tol=1e-8):
    """
    Given `triplet` = [a, b, c] (leaf names) and a Biopython `tree`,
    return a permutation [x, y, z] such that:
        dist(x,y) <= dist(x,z) == dist(y,z)
    within numerical tolerance `tol`.
    """
    a, b, c = triplet
    d_ab = tree_distance(a, b)
    d_ac = tree_distance(a, c)
    d_bc = tree_distance(b, c)

    if abs(d_ab - d_ac) < tol and d_bc <= d_ab:
        return [b, c, a]
    if abs(d_ab - d_bc) < tol and d_ac <= d_ab:
        return [a, c, b]
    if abs(d_ac - d_bc) < tol and d_ab <= d_ac:
        return [a, b, c]

    raise ValueError(f"Triplet {triplet!r} does not satisfy dist(x,y) ≤ dist(x,z)==dist(y,z)")

def classify_and_order_quartet(quartet, tree_distance, tol=1e-8):
    """
    Classify a quartet of leaf names as either 'comb' or 'cherry',
    returning the classification and ordered list [x, y, z, w].
    """
    pairs = list(itertools.combinations(quartet, 2))
    x, y = min(pairs, key=lambda p: tree_distance(*p))
    z, w = [leaf for leaf in quartet if leaf not in (x, y)]

    distance_diff = tree_distance(x, w) - tree_distance(x, z)

    if abs(distance_diff) < tol:
        return "cherry", [x, y, z, w]
    elif distance_diff > 0:
        return "comb", [x, y, z, w]
    else:
        return "comb", [x, y, w, z]

def compute_covariance_matrix_from_tree(tree,**kwargs):
    dist_matrix= distance_matrix_from_tree(tree)
    return compute_covariance_matrix(dist_matrix = dist_matrix, **kwargs)

def compute_covariance_matrix(dist_matrix, mutation_rate=1, sites_per_gene=100, mode = "total"):
    """
    Compute a covariance matrix for a phylogenetic tree under a mutation model.

    Parameters
    ----------
    dist_matrix : np.ndarray
        A square matrix of shape (N, N) representing pairwise distances
        between N sequences or taxa based on the METAL estimate.
    mutation_rate : float
        Substitution rate per unit branch length.
    sites_per_gene : int
        Number of sites per gene considered for the covariance.

    Returns
    -------
    sigma : np.ndarray
        Covariance matrix of shape (N, N) for N = C(m, 2).
    """
    mu = mutation_rate
    N = dist_matrix.shape[0]
    leaves = list(range(N))
    
    leaf_pairs = list(itertools.combinations(leaves, 2))
    pair_index = { (x, y): i for i, (x, y) in enumerate(leaf_pairs) }
    pair_index.update({ (y, x): i for (x, y), i in pair_index.items() })

    sigma = np.zeros((len(leaf_pairs), len(leaf_pairs)))

    def insert_covariance_components(combinations):
        for (p1, p2, value) in combinations:
            idx1 = pair_index[p1]
            idx2 = pair_index[p2]
            sigma[idx1, idx2] = value
            sigma[idx2, idx1] = value
        
    def tree_distance(*p):
        return dist_matrix[p]
    
    def cov_value(coal, sub, correction):
        # correction is product of expectations under coal model
        # coal is covariance under coal model
        # sub is e^{-\mu(g_{ab} + g_{cd} - \delta_{ab,cd})}
        # cov_sub propto sub - expectation of product
        # expectation of product = covariance + product of expecations
        # ie coal + correction.
        cov_coal = 9/16 * coal
        cov_sub = 9/16 / sites_per_gene * (sub - coal - correction)
        if mode == "total":
            return cov_coal + cov_sub
        elif mode == "coal":
            return cov_coal
        elif mode == "sub":
            return cov_sub

    def two_leaves(pair):
        S_ab = tree_distance(*pair)

        variance = (4 * mu**2 * np.exp(-2 * mu * S_ab)) / ((1 + 4 * mu) * (1 + 2 * mu)**2)
        e_ab_ab = np.exp(-mu * S_ab) / (1 + 2 * mu)
        correction = (np.exp(-2 * mu * S_ab)) / (1 + 2 * mu)**2
        value = cov_value(variance, e_ab_ab, correction)

        insert_covariance_components([(pair, pair, value)])

    def three_leaves(triplet):
        a, b, c = triplet
        S_ab = tree_distance(a, b)
        S_ac = tree_distance(a, c)
        S_bc = tree_distance(b, c)

        tau = (S_ac - S_ab) / 2
        factor1 = (4 * mu**2) / ((1 + 2*mu)**2 * (1 + 4*mu) * (3 + 4*mu))
        factor2 = (4 * mu**2) / ((1 + 2*mu)**2 * (1 + 4*mu))
        factor3 = (8 * mu**2) / ((1 + 2*mu) * (1 + 4*mu) * (3 + 4*mu))

        cov_ab_ac = np.exp(-tau + 2 * (-mu) * S_ac) * factor1
        cov_ac_bc = np.exp(-mu * (S_ac + S_bc)) * (factor2 - np.exp(-tau) * factor3)

        ex = np.exp(-mu * (S_ac - S_ab/2)) / ((1 + mu) * (1 + 2 * mu))
        def correction(S1, S2):
            return np.exp(-mu * (S1 + S2)) / (1 + 2 * mu)**2

        def compute_value(cov, S1, S2):
            return cov_value(cov, ex, correction(S1, S2))

        value_ab_ac = compute_value(cov_ab_ac, S_ab, S_ac)
        value_ab_bc = value_ab_ac
        value_ac_bc = compute_value(cov_ac_bc, S_ac, S_bc)

        insert_covariance_components([
            ((a, b), (a, c), value_ab_ac),
            ((a, b), (b, c), value_ab_bc),
            ((a, c), (b, c), value_ac_bc)
        ])

    def four_leaves(quartet, kind):
        a, b, c, d = quartet

        S_ab = tree_distance(a, b)
        S_ac = tree_distance(a, c)
        S_ad = tree_distance(a, d)
        S_bc = tree_distance(b, c)
        S_bd = tree_distance(b, d)
        S_cd = tree_distance(c, d)
        Delta = S_ad

        def correction(S1, S2):
            return np.exp(-mu * (S1 + S2)) / (1 + 2*mu)**2

        if kind == "cherry":
            tau1 = (Delta - S_ab)/2
            tau2 = (Delta - S_cd)/2
            factor_cov = (8 * mu**2) / ((1 + 2*mu)**2 * (1 + 4*mu) * (3 + 4*mu) * (3 + 2*mu))
            cov_ab_cd = np.exp(-mu * (S_ab + S_cd) - (tau1 + tau2)) * factor_cov
            cov_ac_bd = 1 - 2*(1+2*mu)/(3+4*mu)*(np.exp(-tau1)+np.exp(-tau2)) + \
                        (5+4*mu)*(1+2*mu)/(3+4*mu)/(3+2*mu) * np.exp(-(tau1+tau2))
            cov_ac_bd *= (4 * mu**2 / ((1 + 4*mu)*(1 + 2*mu)**2)) * np.exp(-2*mu * Delta)
        else:  # comb
            tau1 = (S_ac - S_ab) / 2
            tau2 = (Delta - S_ac) / 2
            cov_ab_cd = (np.exp(-(2*mu + 1)*(tau1 + tau2)) / (3 + 4*mu)) * \
                        (1 - ((1 + 2*mu)/(3 + 2*mu)) * np.exp(-2*tau2))
            cov_ab_cd *= (4 * mu**2 / ((1 + 4*mu)*(1 + 2*mu)**2)) * np.exp(-mu * (Delta + S_ab))
            cov_ac_bd = (np.exp(-(2*mu + 1)*tau2) / (3 + 4*mu)) * \
                        (1 - ((1 + 2*mu)/(3 + 2*mu)) * np.exp(-2*tau2 - tau1))
            cov_ac_bd *= (4 * mu**2 / ((1 + 4*mu)*(1 + 2*mu)**2)) * np.exp(-mu * (Delta + S_ac))

        if kind == "cherry":
            ex_ab_cd = (np.exp(-mu * (S_ab + S_cd)) / (1 + 2*mu)**2) + \
                       np.exp(-(2*mu+1)*Delta + (S_ab + S_cd)/2) * (2*mu) / \
                       ((mu+1)*(2*mu+1)**2*(2*mu+3))
            ex_ac_bd = (np.exp(-mu * (Delta + (S_ab + S_cd)/2)) / ((1 + 2*mu)*(1 + mu)**2)) + \
                       np.exp(-(2*mu+1)*Delta + (S_ab + S_cd)/2) * (mu*(mu+2)) / \
                       ((mu+1)**2*(2*mu+1)*(2*mu+3))
        else:
            ex_ab_cd = (np.exp(-mu * (S_ab + S_cd)) / (1 + 2*mu)**2) + \
                       np.exp(-mu*(Delta + S_ac) + (Delta - S_ab)/2) * (2*mu) / \
                       ((mu+1)*(2*mu+1)**2*(2*mu+3))
            ex_ac_bd = (np.exp(-mu * (Delta + (S_ab + S_cd)/2)) / ((1 + 2*mu)*(1 + mu)**2)) + \
                       np.exp(-mu*(Delta + S_ac) + (Delta - S_ab)/2) * (mu*(mu+2)) / \
                       ((mu+1)**2*(2*mu+1)*(2*mu+3))

        value_ab_cd = cov_value(cov_ab_cd, ex_ab_cd,correction(S_ab, S_cd))
        value_ac_bd = cov_value(cov_ac_bd, ex_ac_bd,correction(S_ac, S_bd))
        value_ad_bc = value_ac_bd

        insert_covariance_components([
            ((a, b), (c, d), value_ab_cd),
            ((a, c), (b, d), value_ac_bd),
            ((a, d), (b, c), value_ad_bc)
        ])

    for p in leaf_pairs:
        two_leaves(p)

    for triplet in itertools.combinations(leaves, 3):
        ordered = order_triplet(triplet, tree_distance)
        three_leaves(ordered)

    for quartet in itertools.combinations(leaves, 4):
        kind, ordered = classify_and_order_quartet(quartet, tree_distance)
        four_leaves(ordered, kind)

    return sigma
