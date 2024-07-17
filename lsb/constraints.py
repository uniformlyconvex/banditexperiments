import functools
import numpy as np
import scipy.optimize as opt
import typing as t

Con = t.TypeVar('Con', opt.NonlinearConstraint, opt.LinearConstraint, opt.Bounds)
Constraints: t.TypeAlias = t.Iterable[opt.NonlinearConstraint | opt.LinearConstraint | opt.Bounds]

def does_constraint_hold(constraint: Con, item: np.ndarray) -> bool:
    """Check if a constraint holds for a given item."""
    if isinstance(constraint, opt.NonlinearConstraint):
        # lb <= f(item) <= ub ?
        res = constraint.fun(item)
        return np.all(constraint.lb <= res) and np.all(res <= constraint.ub)
    
    if isinstance(constraint, opt.LinearConstraint):
        # lb <= A @ item <= ub ?
        prod = constraint.A @ item
        return np.all(constraint.lb <= prod) and np.all(prod <= constraint.ub)
    
    if isinstance(constraint, opt.Bounds):
        # lb <= item <= ub ?
        return np.all(constraint.lb <= item) and np.all(item <= constraint.ub)

def _extend_bounds(lb: np.ndarray, ub: np.ndarray, dimension: int, extend_end: bool=True) -> t.Tuple[np.ndarray, np.ndarray]:
    """Extend the bounds to be twice their initial dimension, to allow for the inclusion of another variable."""
    # This is messy due to needing to handle scalars
    if isinstance(lb, (float, int)):
        lb *= np.ones((dimension,))
    if isinstance(ub, (float, int)):
        ub *= np.ones((dimension,))

    old_lb = lb.reshape(1,-1)
    old_ub = ub.reshape(1,-1)
    lb_extension = -np.inf * np.ones((1, dimension))
    ub_extension = np.inf * np.ones((1, dimension))

    lb = np.hstack((old_lb, lb_extension)) if extend_end else np.hstack((lb_extension, old_lb))
    ub = np.hstack((old_ub, ub_extension)) if extend_end else np.hstack((ub_extension, old_ub))

    return lb.reshape(-1), ub.reshape(-1)


def _extend_constraint(constraint: Con, dimension: int, extend_end: bool=True) -> Con:
    """Extend a constraint to be twice its initial dimension, to allow for the inclusion of another variable."""
    if isinstance(constraint, opt.NonlinearConstraint):
        # A nonlinear constraint is of the form lb <= f(x) <= ub.
        # So we can just wrap the function to ignore the new dimension.#
        @functools.wraps(constraint.fun)
        def fun(x: np.ndarray) -> np.ndarray:
            true_param = x[:dimension] if extend_end else x[dimension:]
            return constraint.fun(true_param)
        
        return opt.NonlinearConstraint(fun, constraint.lb, constraint.ub)
    
    if isinstance(constraint, opt.LinearConstraint):
        # A linear constraint is of form lb < A @ x < ub.
        # So we can just extend the matrix to include the new dimension, which will be multiplied by 0.
        A_extension = np.zeros((constraint.A.shape[0], dimension))
        A = np.hstack((constraint.A, A_extension)) if extend_end else np.hstack((A_extension, constraint.A))
        return opt.LinearConstraint(A, constraint.lb, constraint.ub)

    if isinstance(constraint, opt.Bounds):
        # A bounds constraint is of form lb < x < ub.
        # So we can just extend the bounds to include the new dimension, which will be between -infty and infty.
        lb, ub = _extend_bounds(constraint.lb, constraint.ub, dimension, extend_end)
        return opt.Bounds(lb, ub)
    

def _bounds_to_linear(bounds: opt.Bounds) -> opt.LinearConstraint:
    """
    Converts a bounds object to a linear constraint.
    Necessary since some optimisation functions support LinearConstraints but not Bounds.
    """
    A = np.eye(bounds.lb.shape[0])
    return opt.LinearConstraint(A, bounds.lb, bounds.ub)


class ConstrainedSubset:
    """ A subset of R^d specified by some constraints."""
    def __init__(self, dimension: int, constraints: Constraints) -> None:
        self._dimension = dimension
        self._constraints = constraints

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def constraints(self) -> Constraints:
        return self._constraints
    
    def __contains__(self, vector: np.ndarray) -> bool:
        return all(
            does_constraint_hold(constraint, vector)
            for constraint in self.constraints
        )
    

class UnitSphere(ConstrainedSubset):
    def __init__(self, dimension: int) -> None:
        constraint = opt.NonlinearConstraint(
            fun=np.linalg.norm,
            lb=0.0,
            ub=1.0
        )
        super().__init__(dimension, [constraint])


class _Singleton(ConstrainedSubset):
    def __init__(self, point: np.ndarray) -> None:
        constraint = opt.Bounds(point, point)
        super().__init__(point.shape[0], [constraint])

        self._point = point

    @property
    def point(self) -> np.ndarray:
        return self._point
    

class FiniteSubset:
    # FiniteSubset can't inherit ConstrainedSubset because we can't define a finite subset purely by constraints.
    # Instead we just abstract away some nonsense
    def __init__(self, points: list[np.ndarray]) -> None:
        if not all(point.shape == points[0].shape for point in points):
            raise ValueError("All points must have the same dimension.")
        self._points = points
        self._dimension = points[0].shape[0]

    @property
    def points(self) -> list[np.ndarray]:
        return self._points
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    def __contains__(self, vector: np.ndarray) -> bool:
        return any(np.all(vector == point) for point in self.points)
    
    def __iter__(self) -> t.Iterator[np.ndarray]:
        return iter(self.points)

    def __len__(self) -> int:
        return len(self.points)
    
    def as_constrained_subsets(self) -> t.Iterator[ConstrainedSubset]:
        return iter(_Singleton(point) for point in self.points)