import numpy as np
import scipy.optimize as opt

from dataclasses import dataclass

from lsb.algorithm import Algorithm, AlgorithmHyperparams
from lsb.bandit import ActionResult, DecisionSetBandit
from lsb.constraints import ConstrainedSubset, UnitSphere, FiniteSubset
from lsb.utils import maximise_inner_product, regularised_lst_sqrs


class OFULHyperparams(AlgorithmHyperparams):
    """Hyperparameters for the OFUL algorithm."""
    reg_param: float = 0.01  # A regularisation parameter used for regularised least-squares
    delta: float = 0.01  # The confidence level of the confidence set


class OFULBandit(DecisionSetBandit):
    """
    A bandit that provides a decision set that is a constrained subset of R^d.
    Noise is Gaussian with given subgaussianness.
    """
    def sample_noise(self) -> float:
        return self.rng.normal(loc=0.0, scale=self.problem_hyperparams.subgaussianness)
    
    def get_decision_set(self, time: int) -> ConstrainedSubset | FiniteSubset:
        """Returns the decision set at time t."""
        # Something random for now
        constraint = opt.NonlinearConstraint(
            fun=np.linalg.norm,
            lb=0.0,
            ub=1.0
        )
        return ConstrainedSubset(
            dimension=self.problem_hyperparams.dimension,
            constraints=[constraint]
        )


class OFUL(Algorithm):
    """LinUCB (OFUL) algorithm"""
    def __post_init__(self):
        self._action_history = np.array([])
        self._reward_history = np.array([])
        self._t = 1

        self.confidence_set = UnitSphere(self.problem_hyperparams.dimension)
        self.theta_estimate = np.zeros(self.problem_hyperparams.dimension)
    
    @property
    def algo_hyperparams(self) -> OFULHyperparams:
        """We have to override this property to update the type hint."""
        return self._algo_hyperparams

    @property
    def t(self) -> int:
        """The current time step."""
        return self._t
        
    def get_action(self, decision_set: ConstrainedSubset | FiniteSubset) -> np.ndarray:
        """Returns the best action to take, based on the decision set and the current confidence set."""
        optimisation_result = maximise_inner_product(decision_set, self.confidence_set)
        best_action = optimisation_result.x
        return best_action
    
    def update(self, result: ActionResult) -> None:
        """Update action history/reward history and confidence set."""
        self._action_history = (
            np.vstack((self._action_history, result.action)) if self._action_history.size
            else result.action.reshape(1, -1)
        )
        self._reward_history = np.append(self._reward_history, result.reward)
        self._t += 1
                                         
        self.theta_estimate, self.confidence_set = OFUL.compute_confidence_set(
            self._action_history,
            self._reward_history,
            self.algo_hyperparams.delta,
            self.algo_hyperparams.reg_param,
            R=self.problem_hyperparams.subgaussianness,
            S=self.problem_hyperparams.parameter_norm_bound
        )

    @staticmethod
    def compute_V_t(X_t: np.ndarray, lmbda: float) -> np.ndarray:
        """Initially implemented separately for testing."""
        complicated_part = X_t.T @ X_t
        V = lmbda * np.eye(complicated_part.shape[0])
        return V + complicated_part

    @staticmethod
    def compute_confidence_set(
        action_history: np.ndarray,
        reward_history: np.ndarray,
        delta: float,
        reg_param: float,
        R: float,
        S: float
    ) -> tuple[np.ndarray, ConstrainedSubset]:
        """Computes the confidence set based on the action and reward history."""
        V_t = OFUL.compute_V_t(X_t=action_history, lmbda=reg_param)
        lmbda_I = reg_param * np.eye(V_t.shape[0])

        norm_bound = R * np.sqrt(
            2 * np.log(
                (np.linalg.det(V_t) ** 0.5 * np.linalg.det(lmbda_I) ** -0.5) /
                delta
            )
        ) + np.sqrt(reg_param) * S

        theta_estimate = regularised_lst_sqrs(X_t=action_history, y_t=reward_history, lmbda=reg_param)
        
        def offset_norm(theta: np.ndarray) -> float:
            """Given theta, returns ||theta_estimate - theta||_V_t."""
            arg = theta_estimate - theta
            return np.sqrt(arg.T @ V_t @ arg)

        constraint = opt.NonlinearConstraint(
            fun=offset_norm,
            lb=0.0,
            ub=norm_bound
        )
        confidence_set = ConstrainedSubset(dimension=action_history.shape[1], constraints=[constraint])
        
        return (theta_estimate, confidence_set)
