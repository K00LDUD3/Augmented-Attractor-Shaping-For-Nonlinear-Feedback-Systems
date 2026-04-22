"""
Pillar 6 — Standalone PID Baseline Stress Test & IC Generator
================================================================
Generates a fixed set of 250 LHS initial conditions on the SURFACE of the
6D hypercube [-40, 40]^6, then runs the pure PID baseline (zero RL action)
over all of them with multiprocessing.

The ICs and baseline results are saved once and reused by train_sac.py.

Batch folder: IC_basePID_stresstest_batch/

Usage:
    python run_baseline_stress_test.py
    python run_baseline_stress_test.py --n-ics 250 --seed 42
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch  # ALWAYS first (DLL conflict)
import numpy as np
import json
import time
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

from ExperimentTracker import ExperimentTracker
from Logger import Logger

from lorenz_env import CoupledLorenzHybridEnv


# ---------------------------------------------------------------------------
# JSON Encoder
# ---------------------------------------------------------------------------
class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalar types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PILLAR_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.dirname(PILLAR_DIR)

SURROGATE_PATH = os.path.join(
    PROJECT_ROOT, "Pillar 5", "experiments",
    "2026-04-10_06-03-45_29331e02", "data", "checkpoints", "ckpt_best_model.pth"
)
PID_GAINS_PATH = os.path.join(
    PROJECT_ROOT, "Pillar 4", "ideal_pid_gains_kp30.json"
)

BATCH_DIR = os.path.join(PILLAR_DIR, "IC_basePID_stresstest_batch")


# ---------------------------------------------------------------------------
# Environment Parameters (matches train_sac.py DEFAULT_PARAMS)
# ---------------------------------------------------------------------------
ENV_PARAMS = {
    "variant": "6c",
    "dt": 0.005,
    "episode_length": 4000,
    "rl_lambda": 25.0,
    "u_max": 250.0,
    "freq_ratio": 10,
    "state_bound": 40.0,
    "init_bound": 40.0,
    "reward_distance_coeff": 5.0,
    "reward_effort_coeff": 20.0,
    "reward_gali_coeff": 0.1,
    "reward_convergence_bonus": 1.0,
    "reward_sparsity_coeff": 0.1,
    "reward_action_deriv_coeff": 1.0,
    "convergence_threshold": 0.5,
}

VARIANT_CONFIG = {"include_gali_obs": True, "reward_gali_coeff": 0.05, "label": "GALI Both (Obs+Reward)"}


# ---------------------------------------------------------------------------
# IC Generation
# ---------------------------------------------------------------------------
def generate_lhs_ics(n_samples, dim=6, bounds=(-40.0, 40.0), seed=42):
    """
    Generate a fixed set of initial conditions on the SURFACE of the 6D hypercube
    using Latin Hypercube Sampling for the interior coordinates.
    """
    from scipy.stats import qmc

    sampler = qmc.LatinHypercube(d=dim, seed=seed)
    unit_samples = sampler.random(n=n_samples)  # [0, 1]^d

    # Snap each sample to a random face (surface of the hypercube)
    np.random.seed(seed)
    for i in range(n_samples):
        face_dim = np.random.randint(0, dim)
        face_side = np.random.randint(0, 2)
        unit_samples[i, face_dim] = float(face_side)

    lo, hi = bounds
    ics = lo + unit_samples * (hi - lo)  # scale to [lo, hi]^d
    return ics


# ---------------------------------------------------------------------------
# Environment Factory
# ---------------------------------------------------------------------------
def create_env(params, variant_config):
    """Instantiate the hybrid environment with variant-specific config."""
    gali_coeff = variant_config["reward_gali_coeff"]

    env = CoupledLorenzHybridEnv(
        surrogate_path=SURROGATE_PATH,
        pid_gains_path=PID_GAINS_PATH,
        dt=params["dt"],
        episode_length=params["episode_length"],
        rl_lambda=params["rl_lambda"],
        u_max=params["u_max"],
        freq_ratio=params["freq_ratio"],
        reward_distance_coeff=params["reward_distance_coeff"],
        reward_effort_coeff=params["reward_effort_coeff"],
        reward_gali_coeff=gali_coeff,
        reward_convergence_bonus=params["reward_convergence_bonus"],
        convergence_threshold=params["convergence_threshold"],
        state_bound=params["state_bound"],
        init_bound=params["init_bound"],
    )

    env._include_gali_obs = variant_config["include_gali_obs"]

    if not variant_config["include_gali_obs"]:
        original_build = env._build_obs
        def masked_build(state, u_pid, log_gali_norm):
            obs = original_build(state, u_pid, log_gali_norm)
            obs[6] = 0.0
            return obs
        env._build_obs = masked_build

    return env


# ---------------------------------------------------------------------------
# Worker (top-level for pickling)
# ---------------------------------------------------------------------------
def _baseline_worker(args):
    """
    Worker function for parallel PID baseline.
    Must be top-level (not nested) so it's picklable by multiprocessing.
    Each worker creates its own env instance with its own surrogate copy.
    """
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["FORCE_CPU_SURROGATE"] = "1"  # Bypass torch CUDA init entirely

    ic_batch, params, variant_config = args

    # Create a fresh env in this process (loads its own surrogate)
    env = create_env(params, variant_config)
    zero_action = np.zeros(env.action_space.shape[0])
    results = []

    for idx, ic in ic_batch:
        obs, info = env.reset(options={"initial_state": ic})
        done = False
        ep_reward = 0.0

        while not done:
            obs, reward, terminated, truncated, step_info = env.step(zero_action)
            ep_reward += reward
            done = terminated or truncated

        m = step_info.get("episode_metrics", {})
        result = {
            "ic_index": idx,
            "initial_state": ic.tolist(),
            "reward": float(ep_reward),
            "total_effort": float(m.get("total_effort", 0)),
            "final_error": float(m.get("final_error", 0)),
            "itwae": float(m.get("itwae", 0)),
            "convergence_step": m.get("convergence_step", None),
            "max_overshoot": float(m.get("max_overshoot", 0)),
            "diverged": m.get("diverged", False),
        }
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Standalone PID Baseline Stress Test & IC Generator")
    parser.add_argument("--n-ics", type=int, default=250, help="Number of ICs to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for IC generation")
    parser.add_argument("--max-workers", type=int, default=8, help="Max parallel workers")
    args = parser.parse_args()

    # --- Experiment Tracker ---
    tracker = ExperimentTracker(BATCH_DIR)

    run_params = {
        **ENV_PARAMS,
        "n_ics": args.n_ics,
        "seed": args.seed,
        "max_workers": args.max_workers,
        "ic_strategy": "LHS_surface_snap",
    }

    run = tracker.create_run(
        params=run_params,
        notes=f"Standalone PID Baseline Stress Test | {args.n_ics} surface ICs (seed={args.seed})"
    )

    # --- Logger ---
    logger = Logger(
        f=run.get_path("logs/run.log"),
        sesh_file=run.get_path("logs/.sesh_num")
    )

    # Copy this script for reproducibility
    run.copy_file(os.path.abspath(__file__), "configs/")

    logger.start_session()
    try:
        logger.log(f"Run UID: {run.uid}", tag="BEG", level="INFO")
        logger.log(f"Generating {args.n_ics} LHS ICs on the surface of [-40, 40]^6 (seed={args.seed})...",
                    tag="DEF", level="INFO")

        # --- Generate ICs ---
        master_ics = generate_lhs_ics(
            n_samples=args.n_ics,
            bounds=(-ENV_PARAMS["init_bound"], ENV_PARAMS["init_bound"]),
            seed=args.seed,
        )

        # Verify surface constraint
        lo, hi = -ENV_PARAMS["init_bound"], ENV_PARAMS["init_bound"]
        n_on_surface = sum(
            1 for ic in master_ics
            if any(np.isclose(ic[d], lo) or np.isclose(ic[d], hi) for d in range(6))
        )
        logger.log(f"  Surface verification: {n_on_surface}/{args.n_ics} ICs on boundary",
                    tag="DEF", level="INFO")

        # Save master ICs
        with open(run.get_path("logs/master_fixed_ics.json"), "w") as f:
            json.dump({"master_ics": master_ics.tolist(), "seed": args.seed,
                        "n_ics": args.n_ics, "strategy": "LHS_surface_snap"}, f, indent=2)
        logger.log("Master ICs saved to logs/master_fixed_ics.json", tag="PAT", level="INFO")

        # --- PID Baseline Stress Test ---
        n_episodes = len(master_ics)
        n_workers = min(multiprocessing.cpu_count(), args.max_workers)

        logger.log(f"=" * 60, tag="DEF", level="INFO")
        logger.log(f"PID BASELINE STRESS TEST ({n_episodes} ICs, {n_workers} workers)", tag="DEF", level="INFO")
        logger.log(f"=" * 60, tag="DEF", level="INFO")

        start_t = time.time()

        # Split (index, ic) pairs across workers
        indexed_ics = list(enumerate(master_ics))
        chunk_size = max(1, len(indexed_ics) // n_workers)
        chunks = [indexed_ics[i:i + chunk_size] for i in range(0, len(indexed_ics), chunk_size)]

        worker_args = [(chunk, ENV_PARAMS, VARIANT_CONFIG) for chunk in chunks]

        # --- Run in parallel ---
        results = []
        try:
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                for batch_results in pool.map(_baseline_worker, worker_args):
                    results.extend(batch_results)
        except Exception as e:
            logger.log(f"Parallel baseline failed ({e}), falling back to sequential", tag="DEF", level="WARNING")
            results = _baseline_worker((indexed_ics, ENV_PARAMS, VARIANT_CONFIG))

        # Sort by index so results are deterministic in ordering
        results.sort(key=lambda r: r["ic_index"])

        elapsed = time.time() - start_t

        # --- Aggregate stats ---
        efforts = [r["total_effort"] for r in results]
        errors  = [r["final_error"] for r in results]
        itwae   = [r["itwae"] for r in results]
        rewards = [r["reward"] for r in results]
        conv_steps = [r["convergence_step"] for r in results if r["convergence_step"] is not None]
        n_diverged = sum(1 for r in results if r["diverged"])

        baseline_summary = {
            "n_episodes": n_episodes,
            "n_workers": n_workers,
            "wall_time_sec": round(elapsed, 1),
            "n_diverged": n_diverged,
            "effort":  {"mean": float(np.mean(efforts)), "std": float(np.std(efforts)),
                        "min": float(np.min(efforts)),   "max": float(np.max(efforts)),
                        "median": float(np.median(efforts))},
            "error":   {"mean": float(np.mean(errors)),  "std": float(np.std(errors)),
                        "min": float(np.min(errors)),    "max": float(np.max(errors)),
                        "median": float(np.median(errors))},
            "itwae":   {"mean": float(np.mean(itwae)),   "std": float(np.std(itwae)),
                        "min": float(np.min(itwae)),     "max": float(np.max(itwae)),
                        "median": float(np.median(itwae))},
            "reward":  {"mean": float(np.mean(rewards)),  "median": float(np.median(rewards))},
            "convergence_step": {"mean": float(np.mean(conv_steps)) if conv_steps else None,
                                 "median": float(np.median(conv_steps)) if conv_steps else None},
        }

        # --- Log ---
        logger.log(f"  Completed in {elapsed:.1f}s ({n_workers} workers)", tag="DEF", level="INFO")
        logger.log(f"  Diverged:      {n_diverged}/{n_episodes}", tag="DEF", level="INFO")
        logger.log(f"  Effort:        mean={baseline_summary['effort']['mean']:.1f}  "
                   f"std={baseline_summary['effort']['std']:.1f}  "
                   f"median={baseline_summary['effort']['median']:.1f}  "
                   f"[{baseline_summary['effort']['min']:.1f}, {baseline_summary['effort']['max']:.1f}]",
                   tag="DEF", level="INFO")
        logger.log(f"  Final Error:   mean={baseline_summary['error']['mean']:.6f}  "
                   f"std={baseline_summary['error']['std']:.6f}  "
                   f"median={baseline_summary['error']['median']:.6f}",
                   tag="DEF", level="INFO")
        logger.log(f"  ITWAE:         mean={baseline_summary['itwae']['mean']:.2f}  "
                   f"std={baseline_summary['itwae']['std']:.2f}  "
                   f"median={baseline_summary['itwae']['median']:.2f}",
                   tag="DEF", level="INFO")
        logger.log(f"  Reward:        mean={baseline_summary['reward']['mean']:.1f}  "
                   f"median={baseline_summary['reward']['median']:.1f}",
                   tag="DEF", level="INFO")
        if conv_steps:
            logger.log(f"  Conv Step:     mean={baseline_summary['convergence_step']['mean']:.0f}  "
                       f"median={baseline_summary['convergence_step']['median']:.0f}",
                       tag="DEF", level="INFO")
        logger.log(f"=" * 60, tag="DEF", level="INFO")

        # --- Save ---
        with open(run.get_path("logs/pid_baseline.json"), "w") as f:
            json.dump({"summary": baseline_summary, "episodes": results}, f, indent=2, cls=NumpyEncoder)
        logger.log("PID baseline results saved to logs/pid_baseline.json", tag="PAT", level="INFO")

        run.add_notes(
            f"Completed {n_episodes} IC baseline. "
            f"Effort: mean={baseline_summary['effort']['mean']:.1f}, "
            f"Error: mean={baseline_summary['error']['mean']:.6f}. "
            f"Time={elapsed:.0f}s."
        )

        logger.log(f"Done. Results are in: {BATCH_DIR}", tag="FIN", level="INFO")

    except Exception as e:
        logger.log(f"Stress test failed: {e}", tag="DEF", level="ERROR")
        run.add_notes(f"FAILED: {e}")
        raise
    finally:
        logger.end_session()


if __name__ == "__main__":
    main()
