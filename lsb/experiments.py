import math
import numpy as np
import warnings

from lsb.bandit import ProblemHyperparams
from lsb.oful import OFULBandit, OFULHyperparams, OFUL
from lsb.tracking import Tracker
from lsb.ts import TSBandit, TSHyperparams, TS
from lsb.ucbdelta import UCBBandit, UCBHyperparams, UCBDelta

def oful():
    # LinUCB example
    DIM = 3
    SUBGAUSSIAN_R = 0.1
    NO_TIME_STEPS = 100
    SEED = 42

    rng = np.random.default_rng(SEED)

    problem_hyperparams = ProblemHyperparams(dimension=DIM, no_arms=math.inf, subgaussianness=SUBGAUSSIAN_R)
    bandit = OFULBandit(rng=rng, problem_hyperparams=problem_hyperparams)
    algo_hyperparams = OFULHyperparams(reg_param=0.01, delta=0.01)
    algorithm = OFUL(rng=rng, problem_hyperparams=problem_hyperparams, algo_hyperparams=algo_hyperparams)

    tracker = Tracker()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for t in range(NO_TIME_STEPS):
            decision_set = bandit.get_decision_set(time=t)
            action = algorithm.get_action(decision_set)
            results = bandit.get_results(action, decision_set)
            algorithm.update(results)

            log_distance_to_truth = np.log(
                np.linalg.norm(bandit.hidden_param - algorithm.theta_estimate)
            )

            tracker.track_scalars({
                "Reward": results.reward,
                "Denoised Reward": results.denoised_reward,
                "Pseudoregret": results.pseudoregret,
                "Log Distance to Truth": log_distance_to_truth
            })

            if t % (NO_TIME_STEPS // 100) == 0:
                print(f'Time step {t} / {NO_TIME_STEPS} complete')

    tracker.plot_all()

def ucbdelta_dimensionality():
    MAX_DIM = 20
    STEP_SIZE = 5

    NO_TIME_STEPS = 10000
    NO_ARMS = 10
    NO_RUNS = 20
    SEED = 42

    rng = np.random.default_rng(SEED)

    tracker = Tracker(max_runs=NO_RUNS, max_time_steps=NO_TIME_STEPS)
    dimensions = range(STEP_SIZE, MAX_DIM+1, STEP_SIZE)

    for dim in dimensions:
        print("Running dimension", dim)
        
        for run in range(NO_RUNS):
            print(f"\t Run {run + 1} / {NO_RUNS}")
            problem_hyperparams = ProblemHyperparams(dimension=dim, no_arms=NO_ARMS, subgaussianness=1.0)
            bandit = UCBBandit(rng=rng, problem_hyperparams=problem_hyperparams)
            algo_hyperparams = UCBHyperparams(delta=0.001)
            algorithm = UCBDelta(rng=rng, problem_hyperparams=problem_hyperparams, algo_hyperparams=algo_hyperparams)
            tracker.new_run()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                for t in range(NO_TIME_STEPS):
                    decision_set = bandit.get_decision_set(time=t)
                    action = algorithm.get_action(decision_set)
                    results = bandit.get_results(action, decision_set)
                    algorithm.update(results)

                    tracker.track_scalars(
                        {
                            f"Pseudoregret (dimension {dim})": results.pseudoregret
                        },
                        cumulative=True,
                        average_over_runs=True
                    )

                    if t % (NO_TIME_STEPS // 10) == 0:
                        print(f'\t\tTime step {t} / {NO_TIME_STEPS} complete')

        tracker.finished_runs()

    tracker.plot_together(
        [
            [f"Pseudoregret (dimension {dim})" for dim in dimensions]
        ]
    )

ucbdelta_dimensionality()