import numpy as np
import typing as t

from abc import ABC
from dataclasses import dataclass

from lsb.bandit import ProblemHyperparams

@dataclass
class AlgorithmHyperparams:
    pass

class Algorithm(ABC):
    def __init__(
        self,
        rng: np.random.Generator,
        problem_hyperparams: t.Type[ProblemHyperparams],
        algo_hyperparams: t.Type[AlgorithmHyperparams]
    ) -> None:
        self._rng = rng
        self._problem_hyperparams = problem_hyperparams
        self._algo_hyperparams = algo_hyperparams

        # Allow algorithms to do some post-init stuff
        try:
            self.__post_init__()
        except AttributeError:
            pass
    
    @property
    def rng(self) -> np.random.Generator:
        return self._rng
    
    @property
    def problem_hyperparams(self) -> t.Type[ProblemHyperparams]:
        return self._problem_hyperparams
    
    @property
    def algo_hyperparams(self) -> t.Type[AlgorithmHyperparams]:
        return self._algo_hyperparams