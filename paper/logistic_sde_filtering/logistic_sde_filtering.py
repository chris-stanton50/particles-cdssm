"""Compare bootstrap and guided particle filters on the logistic model.

For each fixed horizon ``T``, the particle count is set to ``N = 0.64 * T``.
One latent path and observation sequence is simulated per horizon and reused
for every repetition.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from particles_cdssm.cdssm_lib import CDSSM_LIB
from particles_cdssm.tools import build_cdssm, isremote
import particles_cdssm.feynman_kac as sfk
from particles_cdssm import CDSSM_SMC
from particles_cdssm.auxiliary_bridges import LocalLinearOUProp


T_VALUES = (100, 200, 400, 800, 1600)
PARTICLE_SCALING = 0.64
DEFAULT_REPETITIONS = 1


def _run_filter(fk, n_particles):
    """Run one filter and return its ESS trace and elapsed CPU time."""
    smc = CDSSM_SMC(
        fk=fk, N=n_particles, resampling="systematic", ESSrmin=1.0,
        store_history=False, verbose=False, collect=None, num=50,
    )
    smc.run()
    return {
        "ess": np.asarray(smc.summaries.ESSs),
        "cpu_time": float(smc.cpu_time),
        "LogLt": float(smc.logLt),
    }


def generate_results(repetitions=DEFAULT_REPETITIONS):
    """Run both filters with a particle count that scales linearly with T.

    For every hard-coded horizon, ``N = 0.64 * T``. One latent path and its
    observations are simulated, then reused by bootstrap and guided filters
    across every repetition at that horizon.
    """
    repetitions = int(repetitions)
    if repetitions < 1:
        raise ValueError("repetitions must be positive")

    cdssm = build_cdssm(CDSSM_LIB["LAMPERTI_LOGISTIC_NB_LOW_CDT"])

    results = {}
    for horizon in T_VALUES:
        particle_count = PARTICLE_SCALING * horizon
        if not float(particle_count).is_integer():
            raise ValueError("each T must make 0.64 * T an integer particle count")
        n_particles = int(particle_count)
        configuration = {
            "N": n_particles,
            "T": horizon,
            "K": PARTICLE_SCALING,
            "bootstrap": {"ess": [], "cpu_time": [], "LogLt": []},
            "guided": {"ess": [], "cpu_time": [], "LogLt": []},
        }

        # One data set per T; repetitions differ only in filter randomness.
        _, observations = cdssm.simulate(horizon)
        for _ in range(repetitions):
            for name in ("bootstrap", "guided"):
                # Use a fresh FK object for every SMC run.
                if name == "bootstrap":
                    fk = sfk.BootstrapDA(cdssm=cdssm, data=observations)
                else:
                    fk = sfk.ForwardGuidedDA(
                        cdssm=cdssm,
                        data=observations,
                        proposal_sde_cls=LocalLinearOUProp,
                    )
                run = _run_filter(fk, n_particles)
                configuration[name]["ess"].append(run["ess"])
                configuration[name]["cpu_time"].append(run["cpu_time"])
                configuration[name]["LogLt"].append(run["LogLt"])

        for name in ("bootstrap", "guided"):
            configuration[name] = {
                "ess": np.stack(configuration[name]["ess"]),
                "cpu_time": np.asarray(configuration[name]["cpu_time"]),
                "LogLt": np.asarray(configuration[name]["LogLt"]),
            }
        results[(n_particles, horizon)] = configuration
    return results


def results_to_dataframe(results):
    """Flatten results into one row per Feynman--Kac model and repetition.

    ``K`` is the particle-scaling factor, ``repetition`` identifies repeated
    runs, and ``AvgESS`` is the mean ESS over every observation time.
    """
    rows = []

    for result in results.values():
        for fk in ("bootstrap", "guided"):
            ess_runs = result[fk]["ess"]
            cpu_times = result[fk]["cpu_time"]
            log_lts = result[fk]["LogLt"]
            for repetition, (ess_values, cpu_time, log_lt) in enumerate(
                zip(ess_runs, cpu_times, log_lts), start=1
            ):
                rows.append(
                    {
                        "N": result["N"],
                        "T": result["T"],
                        "K": result["K"],
                        "repetition": repetition,
                        "CPU": cpu_time,
                        "fk": fk,
                        "AvgESS": float(np.mean(ess_values)),
                        "LogLt": log_lt,
                    }
                )

    return pd.DataFrame(
        rows,
        columns=["N", "T", "K", "repetition", "CPU", "fk", "AvgESS", "LogLt"],
    )


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", dest="run_id", required=True, help="identifier for this experiment run")
    parser.add_argument("--repetitions", type=int, required=True, help="independent SMC runs per filter")
    return parser.parse_args()


def main():
    args = _parse_args()
    results = generate_results(args.repetitions)
    dataframe = results_to_dataframe(results)
    local_or_remote = "remote" if isremote() else "local"
    results_dir = Path("results") / local_or_remote
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"r_id_{args.run_id}_rep_{args.repetitions}.csv"
    dataframe.to_csv(output_path, index=False)
    print(f"Saved {len(dataframe)} rows to {output_path}")
    return dataframe


if __name__ == "__main__":
    main()
