import numpy as np

from lsb.algorithm import Algorithm, AlgorithmHyperparams
from lsb.bandit import ActionResult, DecisionSetBandit
from lsb.constraints import FiniteSubset

class TSHyperparams(AlgorithmHyperparams):
    """Thompson sampling hyperparameters."""
    delta: float = 0.01


class TSBandit(DecisionSetBandit):  
    """
    A bandit that chooses finitely-many arms on the unit sphere.
    Reward noise is Gaussian with given subgaussianness.
    """
    def sample_noise(self) -> float:
        return self.rng.normal(loc=0.0, scale=self.problem_hyperparams.subgaussianness)
    
    def get_decision_set(self, time: int) -> FiniteSubset:
        """Returns the decision set at time t."""
        # Generate random points on the unit sphere
        random_points = self.rng.normal(
            size=(self.problem_hyperparams.no_arms, self.problem_hyperparams.dimension)
        )
        return FiniteSubset([
            point / np.linalg.norm(point)
            for point in random_points
        ])
    
 
class TS(Algorithm):
    """Thompson sampling algorithm."""
    def __post_init__(self):
        self._action_history = np.array([])
        self._reward_history = np.array([])
        self._t = 1

        self._posterior_mean = np.zeros(self.problem_hyperparams.dimension)
        self._posterior_covariance = np.eye(self.problem_hyperparams.dimension)

    @property
    def algo_hyperparams(self) -> TSHyperparams:
        """We have to override this property to update the type hint."""
        return self._algo_hyperparams
    
    @property
    def t(self) -> int:
        """The current time step."""
        return self._t
    
    @property
    def theta_estimate(self) -> np.ndarray:
        """The current estimate of the hidden parameter."""
        return self._posterior_mean

    def get_action(self, decision_set: FiniteSubset) -> np.ndarray:
        """
        Choose an action from the decision set by drawing from the posterior
        and maximising the inner product.
        """
        sample = self.rng.multivariate_normal(self._posterior_mean, self._posterior_covariance)
        action = max(decision_set.points, key=lambda point: np.dot(point, sample))
        return action

    def update(self, result: ActionResult) -> None:
        """Update action history/reward history and posterior."""
        self._action_history = np.vstack((self._action_history, result.action)) if self._action_history.size else result.action.reshape(1, -1)
        self._reward_history = np.append(self._reward_history, result.reward)

        self._posterior_mean, self._posterior_covariance = self.compute_posterior_mean_covariance(
            self._action_history,
            self._reward_history,
            self.problem_hyperparams.dimension,
            self.algo_hyperparams.delta,
            self.t,
            self.problem_hyperparams.subgaussianness
        )

    @staticmethod
    def compute_posterior_mean_covariance(
        action_history: np.ndarray,
        reward_history: np.ndarray,
        dimension: int,
        delta: float,
        timestep: int,
        R: float
    ) -> tuple[np.ndarray, np.ndarray]:
        v = R * np.sqrt(
            9 * dimension * np.log(
                timestep / delta
            )
        )
        B_t = np.eye(dimension) + action_history.T @ action_history
        mean = np.linalg.inv(B_t) @ action_history.T @ reward_history
        covariance = v**2 * np.linalg.inv(B_t)

        return mean, covariance    
