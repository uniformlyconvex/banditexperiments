import numpy as np
import scipy.optimize as opt
import typing as t

from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays, floating_dtypes

from lsb.utils import _extend_constraint, Con

@st.composite
def lb_and_ub(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray]:
    dimension = draw(st.integers(min_value=1, max_value=5))
    reasonable_vectors = arrays(
        dtype=floating_dtypes(),
        shape=(1,dimension),
    )
    lb = draw(reasonable_vectors)
    ub = draw(reasonable_vectors)
    return (lb, ub)


def bounds(draw: st.DrawFn) -> opt.Bounds:
    lb, ub = draw(lb_and_ub())
    return opt.Bounds(lb, ub)


@st.composite
def nonlinear_constraint(draw: st.DrawFn) -> opt.NonlinearConstraint:
    def generate_random_polynomial() -> t.Callable[[np.ndarray], np.ndarray]:
        degree = np.random.randint(1, 5)
        coefficients = np.random.uniform(-1, 1, size=degree + 1)
        return lambda x: np.polyval(coefficients, x)
    
    lb, ub = draw(lb_and_ub())
    fun = generate_random_polynomial()
    return opt.NonlinearConstraint(fun, lb, ub)


@st.composite
def linear_constraint(draw: st.DrawFn) -> opt.LinearConstraint:
    dimension = draw(st.integers(min_value=1, max_value=5))
    no_constraints = draw(st.integers(min_value=1, max_value=5))
    reasonable_matrices = arrays(
        dtype=floating_dtypes(),
        shape=(no_constraints, dimension)
    )
    lb, ub = draw(lb_and_ub())
    matrix = draw(reasonable_matrices)

    return opt.LinearConstraint(matrix, lb, ub)


@given(bounds())
def test_extend_bounds_end(bounds: opt.Bounds) -> None:
    dimension = len(bounds.lb)
    expected_lb = np.hstack((bounds.lb, -np.inf * np.ones((1,dimension))))
    expected_ub = np.hstack((bounds.ub, np.inf * np.ones((1,dimension))))
    expected = opt.Bounds(expected_lb, expected_ub)

    result = _extend_constraint(bounds, dimension)
    message = f"Expected: {expected}, got: {result}"
    assert np.array_equal(result.lb, expected.lb, equal_nan=True), message
    assert np.array_equal(result.ub, expected.ub, equal_nan=True), message

@given(bounds())
def test_extend_bounds_beginning(bounds: opt.Bounds) -> None:
    dimension = len(bounds.lb)
    expected_lb = np.hstack((-np.inf * np.ones((1,dimension)), bounds.lb))
    expected_ub = np.hstack((np.inf * np.ones((1,dimension)), bounds.ub))
    expected = opt.Bounds(expected_lb, expected_ub)

    result = _extend_constraint(bounds, dimension, extend_end=False)
    message = f"Expected: {expected}, got: {result}"
    assert np.array_equal(result.lb, expected.lb, equal_nan=True), message
    assert np.array_equal(result.ub, expected.ub, equal_nan=True), message

@given(nonlinear_constraint())
def test_extend_nonlinear_constraint_end(constraint: opt.NonlinearConstraint) -> None:
    dimension = len(constraint.lb)
    expected_lb = np.hstack((constraint.lb, -np.inf * np.ones((1,dimension))))
    expected_ub = np.hstack((constraint.ub, np.inf * np.ones((1,dimension))))

    result = _extend_constraint(constraint, dimension)
    # Check the result's lower and upper bounds
    message = f"Expected lower bound: {expected_lb}, got: {result.lb}"
    assert np.array_equal(result.lb, expected_lb, equal_nan=True), message
    message = f"Expected upper bound: {expected_ub}, got: {result.ub}"
    assert np.array_equal(result.ub, expected_ub, equal_nan=True), message

    # Check that the function right-pads the result with zeros
    x = np.random.normal(size=dimension)
    result_value = result.fun(x)
    expected_value = np.hstack((constraint.fun(x).reshape(1,-1), np.zeros((1,dimension)))).reshape(-1)
    message = f"Expected value: {expected_value}, got: {result_value}"
    assert np.array_equal(result_value, expected_value, equal_nan=True), message

@given(nonlinear_constraint())
def test_extend_nonlinear_constraint_beginning(constraint: opt.NonlinearConstraint) -> None:
    dimension = len(constraint.lb)
    expected_lb = np.hstack((-np.inf * np.ones((1,dimension)), constraint.lb))
    expected_ub = np.hstack((np.inf * np.ones((1,dimension)), constraint.ub))

    result = _extend_constraint(constraint, dimension, extend_end=False)
    # Check the result's lower and upper bounds
    message = f"Expected lower bound: {expected_lb}, got: {result.lb}"
    assert np.array_equal(result.lb, expected_lb, equal_nan=True), message
    message = f"Expected upper bound: {expected_ub}, got: {result.ub}"
    assert np.array_equal(result.ub, expected_ub, equal_nan=True), message

    # Check that the function left-pads the result with zeros
    x = np.random.normal(size=dimension)
    result_value = result.fun(x)
    expected_value = np.hstack((np.zeros((1,dimension)), constraint.fun(x).reshape(1,-1))).reshape(-1)
    message = f"Expected value: {expected_value}, got: {result_value}"
    assert np.array_equal(result_value, expected_value, equal_nan=True), message


@given(linear_constraint())
def test_extend_linear_constraint_end(constraint: opt.LinearConstraint):
    dimension = len(constraint.lb)
    expected_lb = np.hstack((constraint.lb, -np.inf * np.ones((1,dimension))))
    expected_ub = np.hstack((constraint.ub, np.inf * np.ones((1,dimension))))

    result = _extend_constraint(constraint, dimension)
    message = f"Expected lower bound: {expected_lb}, got: {result.lb}"
    assert np.array_equal(result.lb, expected_lb, equal_nan=True), message
    message = f"Expected upper bound: {expected_ub}, got: {result.ub}"
    assert np.array_equal(result.ub, expected_ub, equal_nan=True), message

    expected_matrix = np.hstack((constraint.A, np.zeros((constraint.A.shape[0], dimension))))
    message = f"Expected matrix: {expected_matrix}, got: {result.A}"
    assert np.array_equal(result.A, expected_matrix, equal_nan=True), message


@given(linear_constraint())
def test_extend_linear_constraint_beginning(constraint: opt.LinearConstraint) -> None:
    dimension = len(constraint.lb)
    expected_lb = np.hstack((-np.inf * np.ones((1,dimension)), constraint.lb))
    expected_ub = np.hstack((np.inf * np.ones((1,dimension)), constraint.ub))

    result = _extend_constraint(constraint, dimension, extend_end=False)
    message = f"Expected lower bound: {expected_lb}, got: {result.lb}"
    assert np.array_equal(result.lb, expected_lb, equal_nan=True), message
    message = f"Expected upper bound: {expected_ub}, got: {result.ub}"
    assert np.array_equal(result.ub, expected_ub, equal_nan=True), message

    expected_matrix = np.hstack((np.zeros((constraint.A.shape[0], dimension)), constraint.A))
    message = f"Expected matrix: {expected_matrix}, got: {result.A}"
    assert np.array_equal(result.A, expected_matrix, equal_nan=True), message
