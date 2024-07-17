import math
import numpy as np
import typing as t

from abc import ABC, abstractmethod
from dataclasses import dataclass

from lsb.constraints import ConstrainedSubset, FiniteSubset
from lsb.utils import maximise_inner_product

@dataclass
class ActionResult:
    """
    The results of taking an action in a bandit environment.
    For ease we return the action together with the (noisy) reward, the regret,
    and the pseudoregret.
    """
    action: np.ndarray
    decision_set: ConstrainedSubset | FiniteSubset
    reward: float
    denoised_reward: float
    regret: float
    pseudoregret: float

    def __repr__(self) -> str:
        return format(self, ".3f")
    
    def __format__(self, format_spec: str) -> str:
        string = (
            f"Reward {self.reward:{format_spec}}, "
            f"Regret {self.regret:{format_spec}}, "
            f"Pseudoregret {self.pseudoregret:{format_spec}}"
        )
        return string


@dataclass
class ProblemHyperparams:
    """Hyperparameters of the problem itself, not the algorithm."""
    dimension: int
    no_arms: int = math.inf  # This type hint is wrong but it's really hard to type hint integer or infinity correctly
    subgaussianness: float = 1.0
    parameter_norm_bound: float = 1.0


class DecisionSetBandit(ABC):
    """
    A 'Decision set' bandit. Each time step, the bandit provides a decision set
    which is either a finite set of arms or a constrained subset of R^d.
    """
    def __init__(self, rng: np.random.Generator, problem_hyperparams: ProblemHyperparams) -> None:
        self._rng = rng
        self._problem_hyperparams = problem_hyperparams

        self._hidden_param = self.generate_hidden_param()
    
    @property
    def problem_hyperparams(self) -> ProblemHyperparams:
        return self._problem_hyperparams
    
    @property
    def rng(self) -> np.random.Generator:
        return self._rng
    
    @property
    def hidden_param(self) -> np.ndarray:
        return self._hidden_param
    
    def generate_hidden_param(self) -> np.ndarray:
        param = self.rng.normal(size=self.problem_hyperparams.dimension)
        normalised_param = param / np.linalg.norm(param)
        return normalised_param
    
    @abstractmethod
    def get_decision_set(self, time: int) -> ConstrainedSubset | FiniteSubset:
        pass

    @abstractmethod
    def sample_noise(self) -> float:
        pass

    def get_results(
        self,
        action: np.ndarray,
        decision_set: ConstrainedSubset | FiniteSubset
    ) -> ActionResult:
        denoised_reward = np.dot(self.hidden_param, action)

        noise = self.sample_noise()
        reward = denoised_reward + noise

        best_action_result = maximise_inner_product(
            decision_set, FiniteSubset([self.hidden_param])
        )
        optimal_reward = np.dot(self.hidden_param, best_action_result.x)

        regret = optimal_reward - reward
        pseudoregret = optimal_reward - denoised_reward

        return ActionResult(action, decision_set, reward, denoised_reward, regret, pseudoregret)
