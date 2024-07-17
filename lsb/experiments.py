import numpy as np
import warnings

from lsb.oful import OFULBandit, OFULHyperparams, OFUL
from lsb.tracking import Tracker
from lsb.ts import TSBandit, TSHyperparams, TS
from lsb.ucbdelta import UCBBandit, UCBHyperparams, UCBDelta

def decision_set_bandit_experiment():
    DIM = 2

    rng = np.random.default_rng(42)

    bandit = OFULBandit(dimension=DIM, rng=rng)
    hyperparams = OFULHyperparams(reg_param=0.01, subgaussian_R=1.0, param_bound_S=1.0)
    algorithm = OFUL(dimension=DIM, delta=0.01, rng=rng, hyperparams=hyperparams)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for t in range(1000):
            decision_set = bandit.get_decision_set(time=t)
            action = algorithm.get_action(decision_set)
            results = bandit.get_results(action, decision_set)
            algorithm.update(results)

            in_confidence_set = bandit.hidden_param in algorithm.confidence_set
            distance_to_truth = np.linalg.norm(bandit.hidden_param - algorithm.theta_estimate)

            print(f"Time step {t}: {results:4f}")
            print(f"Parameter in confidence set {in_confidence_set}; distance to truth {distance_to_truth:.4f}")

def other():
    DIM = 3
    rng = np.random.default_rng(42)

    bandit = TSBandit(dimension=DIM, rng=rng, no_arms=10)
    hyperparams = TSHyperparams(subgaussian_R=1.0)
    algorithm = TS(dimension=DIM, delta=0.01, rng=rng, hyperparams=hyperparams)
    tracker = Tracker()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for t in range(1000):
            decision_set = bandit.get_decision_set(time=t)
            action = algorithm.get_action(decision_set)
            results = bandit.get_results(action, decision_set)
            algorithm.update(results)

            log_distance_to_truth = np.log(np.linalg.norm(bandit.hidden_param - algorithm.theta_estimate))


def compare():
    DIM = 5
    NO_ARMS = 10
    SUBGAUSSIAN_R = 0.1
    rng = np.random.default_rng(42)


    oful_hyperparams = OFULHyperparams(reg_param=0.01, subgaussian_R=SUBGAUSSIAN_R, param_bound_S=1.0)
    ts_hyperparams = TSHyperparams(subgaussian_R=SUBGAUSSIAN_R)
    bandit = TSBandit(dimension=DIM, rng=rng, no_arms=NO_ARMS, hyperparams=ts_hyperparams)
    oful = OFUL(dimension=DIM, delta=0.01, rng=rng, hyperparams=oful_hyperparams)
    ts = TS(dimension=DIM, delta=0.01, rng=rng, hyperparams=ts_hyperparams)

    tracker = Tracker()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for t in range(1000):
            decision_set = bandit.get_decision_set(time=t)
            oful_action = oful.get_action(decision_set)
            ts_action = ts.get_action(decision_set)

            oful_results = bandit.get_results(oful_action, decision_set)
            ts_results = bandit.get_results(ts_action, decision_set)

            oful.update(oful_results)
            ts.update(ts_results)

            oful_log_distance_to_truth = np.log(np.linalg.norm(bandit.hidden_param - oful.theta_estimate))
            ts_log_distance_to_truth = np.log(np.linalg.norm(bandit.hidden_param - ts.theta_estimate))

            tracker.track_scalars({
                "OFUL Reward": oful_results.reward,
                "OFUL Denoised Reward": oful_results.denoised_reward,
                "OFUL Pseudoregret": oful_results.pseudoregret,
                "OFUL Log Distance to Truth": oful_log_distance_to_truth,
                "TS Reward": ts_results.reward,
                "TS Denoised Reward": ts_results.denoised_reward,
                "TS Pseudoregret": ts_results.pseudoregret,
                "TS Log Distance to Truth": ts_log_distance_to_truth
            })
            tracker.track_scalars({
                "OFUL Cumulative Pseudoregret": oful_results.pseudoregret,
                "TS Cumulative Pseudoregret": ts_results.pseudoregret
                },
                cumulative=True
            )

            if t % 50 == 0:
                print(f"Time step {t}: OFUL {oful_results:4f}, TS {ts_results:4f}")

    tracker.plot_together(
        ["OFUL Reward", "TS Reward"],
        ["OFUL Denoised Reward", "TS Denoised Reward"],
        ["OFUL Pseudoregret", "TS Pseudoregret"],
        ["OFUL Log Distance to Truth", "TS Log Distance to Truth"],
        ["OFUL Cumulative Pseudoregret", "TS Cumulative Pseudoregret"]
    )


def ucb_delta():
    DIM = 5
    NO_ARMS = 100
    SUBGAUSSIAN_R = 0.1
    rng = np.random.default_rng(42)

    ucb_hyperparams = UCBHyperparams(subgaussian_R=SUBGAUSSIAN_R)
    bandit = UCBBandit(dimension=DIM, rng=rng, no_arms=NO_ARMS, hyperparams=ucb_hyperparams)
    ucb = UCBDelta(dimension=DIM, delta=0.01, rng=rng, no_arms=NO_ARMS, hyperparams=ucb_hyperparams)

    tracker = Tracker()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for t in range(10000):
            decision_set = bandit.get_decision_set(time=t)
            ucb_action = ucb.get_action(decision_set)

            ucb_results = bandit.get_results(ucb_action, decision_set)

            ucb.update(ucb_results)

            tracker.track_scalars({
                "UCB Reward": ucb_results.reward,
                "UCB Denoised Reward": ucb_results.denoised_reward,
                "UCB Pseudoregret": ucb_results.pseudoregret,
            })
            tracker.track_scalars({
                "UCB Cumulative Pseudoregret": ucb_results.pseudoregret
                },
                cumulative=True
            )

            if t % 50 == 0:
                print(f"Time step {t}: UCB {ucb_results:4f}")

    tracker.plot_all() 

ucb_delta()