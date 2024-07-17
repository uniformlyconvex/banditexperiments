import functools
import numpy as np
import scipy.optimize as opt
import typing as t

from dataclasses import dataclass

from lsb.constraints import ConstrainedSubset, FiniteSubset, _Singleton, _extend_constraint, _bounds_to_linear


@dataclass
class OptimalEstimate:
    """Holds the optimal estimate of the inner product between two sets."""
    x: np.ndarray
    theta: np.ndarray
    inner_product: float


@functools.lru_cache  # Use an LRU cache since often the decision set is the same and the bandit computes the optimal action repeatedly
def maximise_inner_product(set_A: ConstrainedSubset | FiniteSubset, set_B: ConstrainedSubset | FiniteSubset) -> OptimalEstimate:
    """Maximises the inner product between two arbitrary sets specified by their constraints."""
    if (dim := set_A.dimension) != set_B.dimension:
        raise ValueError("Dimensions of the two sets must match.")
    
    # When both A, B are singletons it's trivial, else we loop over the elements of the finite subsets.
    # If both are finite subsets, this amounts to a brute-force search.
    if isinstance(set_A, _Singleton) and isinstance(set_B, _Singleton):
        x = set_A.point
        theta = set_B.point
        inner_product = np.dot(x, theta)
        return OptimalEstimate(x, theta, inner_product)
    
    if isinstance(set_A, FiniteSubset):
        return max(
            (maximise_inner_product(point, set_B) for point in set_A.as_constrained_subsets()),
            key = lambda estimate: estimate.inner_product
        )
    
    if isinstance(set_B, FiniteSubset):
        return max(
            (maximise_inner_product(set_A, point) for point in set_B.as_constrained_subsets()),
            key = lambda estimate: estimate.inner_product
        )
    
    def objective(args: np.ndarray) -> float:
        # Scipy minimises functions of one variable, so we unpack the arguments and negate the inner product
        x = args[:dim]
        theta = args[dim:]
        return -np.dot(x, theta)
    
    # Extend the constraints to allow fo the inclusion of a second variable
    set_A_constraints = [
        _extend_constraint(constraint, dim, extend_end=True)
        for constraint in set_A.constraints
    ]
    set_B_constraints = [
        _extend_constraint(constraint, dim, extend_end=False)
        for constraint in set_B.constraints
    ]
    constraints = set_A_constraints + set_B_constraints

    # Bounds objects are apparently not supported by trust-constr, so we convert them to LinearConstraints
    constraints = [
        constraint if not isinstance(constraint, opt.Bounds) else _bounds_to_linear(constraint)
        for constraint in constraints
    ]

    res = opt.minimize(
        objective,
        np.ones(2 * dim),
        method='trust-constr',
        constraints=constraints,
    )

    x = res.x[:dim]
    theta = res.x[dim:]
    inner_product = -res.fun

    return OptimalEstimate(x, theta, inner_product)


def regularised_lst_sqrs(X_t: np.ndarray, y_t: np.ndarray, lmbda: float) -> np.ndarray:
    """Solves the regularised least squares problem."""
    return np.linalg.solve(
        X_t.T @ X_t + lmbda * np.eye(X_t.shape[1]),
        X_t.T @ y_t
    ).reshape(-1)
