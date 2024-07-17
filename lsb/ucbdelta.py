import numpy as np

from dataclasses import dataclass

from lsb.algorithm import Algorithm, AlgorithmHyperparams
from lsb.bandit import ActionResult, DecisionSetBandit
from lsb.constraints import FiniteSubset

@dataclass
class UCBHyperparams(AlgorithmHyperparams):
    delta: float = 0.01


class UCBBandit(DecisionSetBandit):
    def __post_init__(self):
        self._arms = self._generate_arms(self.problem_hyperparams.no_arms)
    
    @property
    def arms(self) -> FiniteSubset:
        return self._arms
    
    def sample_noise(self) -> float:
        return self.rng.normal(loc=0.0, scale=self.problem_hyperparams.subgaussianness)
    
    def _generate_arms(self, no_arms: int) -> FiniteSubset:
        arms = self.rng.normal(size=(no_arms, self.problem_hyperparams.dimension))
        return FiniteSubset([
            arm / np.linalg.norm(arm)
            for arm in arms
        ])
    
    def get_decision_set(self, time: int) -> FiniteSubset:
        """Returns the decision set at time t."""
        # This time we keep the same arms every time.
        return self.arms


class UCBDelta(Algorithm):
    def __post_init__(self):
        self._action_counts = np.zeros(self.problem_hyperparams.no_arms)
        self._average_rewards = np.zeros(self.problem_hyperparams.no_arms)

    @property
    def algo_hyperparams(self) -> UCBHyperparams:
        """We have to override this property to update the type hint."""
        return self._algo_hyperparams
    
    def get_action(self, decision_set: FiniteSubset) -> np.ndarray:
        """Returns the action to take at time t."""
        N = self._action_counts
        d = self.problem_hyperparams.no_arms
        c = np.sqrt(
            ((1 + N) / N ** 2) *
            (1 + 2 * np.log(
                (d * (1 + N) ** 1/2) / self.algo_hyperparams.delta
            ))
        )
        UCB = self._average_rewards + c
        return decision_set.points[np.argmax(UCB)]
    
    def update(self, results: ActionResult) -> None:
        decision_set = results.decision_set
        matches = np.all(np.array(decision_set.points) == results.action, axis=1)
        index = np.where(matches)[0][0]
        
        previous_total_reward = self._average_rewards[index] * self._action_counts[index]
        new_total_reward = previous_total_reward + results.reward
        new_avg_reward = new_total_reward / (self._action_counts[index] + 1)

        self._action_counts[index] += 1
        self._average_rewards[index] = new_avg_reward