import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from experiments.drone_search_env import DroneSearchEnv
from src.environment.grid import CellType
from src.metrics.collector import MetricsCollector
from src.model.disaster_model import DisasterModel

RL_MODEL_DIR = ROOT / "results" / "rl_model"
RL_MODEL_PATH = RL_MODEL_DIR / "ppo_drone_search"
EVAL_SEEDS = list(range(30))
N_TRAIN_STEPS = 100_000
HAZARD_RATE = "medium"


def train() -> PPO:
    """
    Trains a PPO agent and saves it to results/rl_model/.

    Returns:
        - The trained PPO model.
    """
    RL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    env = DroneSearchEnv(hazard_rate=HAZARD_RATE)
    env.reset(seed=0)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=42,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        ent_coef=0.01,
    )
    model.learn(total_timesteps=N_TRAIN_STEPS)
    model.save(str(RL_MODEL_PATH))
    print(f"\nModel saved to {RL_MODEL_PATH}.zip")
    return model


def evaluate_rl(model: PPO, n_eval: int = 30) -> dict[str, float]:
    """
    Evaluates the trained policy using SB3's evaluate_policy.

    Args:
        - model: The trained PPO model.
        - n_eval: Number of evaluation episodes.

    Returns:
        - Dict with 'mean_reward' and 'std_reward'.
    """
    env = DroneSearchEnv(hazard_rate=HAZARD_RATE)
    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=n_eval, deterministic=True
    )
    print(f"\n=== RL policy evaluation ({n_eval} episodes) ===")
    print(f"  Mean reward : {mean_reward:.3f}")
    print(f"  Std  reward : {std_reward:.3f}")
    return {"mean_reward": float(mean_reward), "std_reward": float(std_reward)}


def _coverage_pct(sim: DisasterModel) -> float:
    grid_state = sim.disaster_grid.grid_state
    passable = int((grid_state != CellType.OBSTACLE).sum())
    if passable == 0:
        return 0.0
    return float((sim.coverage_grid > 0).sum() / passable * 100)


def run_baseline(strategy: str, seed: int) -> dict:
    """
    Runs one DisasterModel episode at swarm_size=1 and returns metrics.

    Args:
        - strategy: One of 'random', 'astar', 'pheromone'.
        - seed: Random seed for this run.

    Returns:
        - Dict with strategy, seed, survivors_found, agents_lost,
          timesteps_run, coverage_pct.
    """
    sim = DisasterModel(
        strategy=strategy,
        swarm_size=1,
        hazard_rate=HAZARD_RATE,
        seed=seed,
    )
    while not sim.is_done:
        sim.step()
    return {
        "strategy": strategy,
        "seed": seed,
        "survivors_found": sim.survivors_found_count,
        "agents_lost": sim.agents_lost,
        "timesteps_run": sim.timestep,
        "coverage_pct": round(_coverage_pct(sim), 2),
    }


def run_rl_episode(model: PPO, seed: int) -> dict:
    """
    Runs one RL episode deterministically and returns game metrics.

    Args:
        - model: The trained PPO model.
        - seed: Episode seed.

    Returns:
        - Dict with strategy='rl', seed, survivors_found, agents_lost,
          timesteps_run, coverage_pct, total_reward.
    """
    env = DroneSearchEnv(hazard_rate=HAZARD_RATE)
    obs, _ = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(int(action))
        total_reward += float(reward)
        done = terminated or truncated
    sim = env.model
    return {
        "strategy": "rl",
        "seed": seed,
        "survivors_found": sim.survivors_found_count,
        "agents_lost": sim.agents_lost,
        "timesteps_run": sim.timestep,
        "coverage_pct": round(_coverage_pct(sim), 2),
        "total_reward": round(total_reward, 4),
    }


def compare(model: PPO) -> None:
    """
    Compares RL against baseline strategies at swarm_size=1 over 30 seeds.

    Args:
        - model: The trained PPO model.
    """
    rows: list[dict] = []

    for strategy in ("random", "astar", "pheromone"):
        print(f"Running {strategy} x {len(EVAL_SEEDS)} seeds ...", flush=True)
        for seed in EVAL_SEEDS:
            rows.append(run_baseline(strategy, seed))

    print(f"Running rl x {len(EVAL_SEEDS)} seeds ...", flush=True)
    for seed in EVAL_SEEDS:
        rows.append(run_rl_episode(model, seed))

    df = pd.DataFrame(rows)
    out_path = ROOT / "results" / "rl_evaluation.csv"
    df.to_csv(out_path, index=False)
    print(f"\nEvaluation results written to {out_path}")

    print("\n=== swarm_size=1 comparison (mean +/- std, 30 seeds each) ===")
    metrics = ["survivors_found", "agents_lost", "coverage_pct"]
    summary = df.groupby("strategy")[metrics].agg(["mean", "std"]).round(3)
    print(summary.to_string())


def main() -> None:
    print(f"Training PPO for {N_TRAIN_STEPS:,} timesteps ...")
    model = train()

    stats = evaluate_rl(model)
    if stats["mean_reward"] <= 0.0:
        print(
            "\nWARNING: mean reward is not positive - "
            "consider increasing N_TRAIN_STEPS."
        )

    compare(model)


if __name__ == "__main__":
    main()
