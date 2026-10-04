"""Play an exported Orbit Wars agent against random and baseline opponents.

This is a CPU match evaluator. It does not need a training dataset. Offline
token metrics still live in ``python -m orbit_board_bc_train.cli eval``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable


def _rank_reward(reward: float | None) -> float:
    """Missing rewards sort below every numeric reward."""
    if reward is None:
        return float("-inf")
    return float(reward)


def competition_rank(rewards: list[float | None], seat: int) -> int:
    """1-based rank. Tied rewards share the best rank in that tie."""
    mine = _rank_reward(rewards[seat])
    return 1 + sum(1 for reward in rewards if _rank_reward(reward) > mine)


def classify_result(rewards: list[float | None], seat: int) -> str:
    scored = [_rank_reward(reward) for reward in rewards]
    best = max(scored)
    if scored[seat] < best:
        return "loss"
    tied = sum(1 for reward in scored if reward == best)
    return "win" if tied == 1 else "draw"


def summarize_games(games: list[dict[str, Any]]) -> dict[str, Any]:
    if not games:
        return {
            "games": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "win_rate": None,
            "draw_rate": None,
            "loss_rate": None,
            "average_rank": None,
            "average_reward": None,
            "average_steps": None,
        }
    wins = draws = losses = 0
    rank_total = 0
    reward_total = 0.0
    step_total = 0
    for game in games:
        rewards = list(game["rewards"])
        seat = int(game["seat"])
        outcome = classify_result(rewards, seat)
        if outcome == "win":
            wins += 1
        elif outcome == "draw":
            draws += 1
        else:
            losses += 1
        rank_total += competition_rank(rewards, seat)
        agent_reward = rewards[seat]
        if agent_reward is None:
            reward_total = None
        elif reward_total is not None:
            reward_total += float(agent_reward)
        step_total += int(game["steps"])
    count = len(games)
    return {
        "games": count,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": wins / count,
        "draw_rate": draws / count,
        "loss_rate": losses / count,
        "average_rank": rank_total / count,
        "average_reward": None if reward_total is None else reward_total / count,
        "average_steps": step_total / count,
    }


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load agent module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "agent"):
        raise AttributeError(f"{path} does not define agent(obs)")
    return module


def _agent_identity(module) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "model_loaded": getattr(module, "_MODEL", None) is not None,
        "load_error": getattr(module, "_LOAD_ERROR", None),
    }
    checkpoint = getattr(module, "_CKPT", None)
    model = getattr(module, "_MODEL", None)
    if isinstance(checkpoint, dict) and "config" in checkpoint:
        identity["config"] = checkpoint["config"]
    if model is not None:
        identity["parameter_count"] = int(sum(parameter.numel() for parameter in model.parameters()))
    return identity


def _play_game(agents: list[Any], seed: int, episode_steps: int | None) -> tuple[list[float | None], list[str], int]:
    from kaggle_environments import make

    # The built-in random agent draws from the process-global random module.
    # The board itself uses random.Random(configuration seed), so this does not
    # change planet placement. Reseeding per game keeps later games independent.
    random.seed(int(seed))
    configuration: dict[str, Any] = {"seed": int(seed)}
    if episode_steps is not None:
        configuration["episodeSteps"] = int(episode_steps)
    env = make("orbit_wars", configuration=configuration, debug=False)
    env.run(agents)
    final = env.steps[-1]
    rewards: list[float | None] = []
    for state in final:
        reward = state.reward
        rewards.append(None if reward is None else float(reward))
    statuses = [str(state.status) for state in final]
    return rewards, statuses, len(env.steps)


def _opponent(name: str, baseline_agent: Callable) -> Any:
    if name == "random":
        return "random"
    if name == "baseline":
        return baseline_agent
    raise ValueError(f"Unknown opponent {name}")


def run_matchup(
    our_agent: Callable,
    opponent_name: str,
    baseline_agent: Callable,
    games: int,
    seed: int,
    players: int,
    episode_steps: int | None,
) -> list[dict[str, Any]]:
    if players < 2:
        raise ValueError("players must be at least 2")
    if games < 1:
        return []
    records = []
    for index in range(games):
        seat = index % players
        game_seed = seed + index
        agents: list[Any] = [_opponent(opponent_name, baseline_agent) for _ in range(players)]
        agents[seat] = our_agent
        started = time.perf_counter()
        error = None
        try:
            rewards, statuses, steps = _play_game(agents, game_seed, episode_steps)
        except Exception as exc:
            # A numeric opponent reward keeps a failed game from looking like a draw.
            rewards = [0.0] * players
            rewards[seat] = None
            statuses = ["ERROR"] * players
            steps = 0
            error = f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - started
        outcome = classify_result(rewards, seat)
        record = {
            "opponent": opponent_name,
            "players": players,
            "seed": game_seed,
            "seat": seat,
            "rewards": rewards,
            "statuses": statuses,
            "steps": steps,
            "elapsed_sec": round(elapsed, 3),
            "outcome": outcome,
            "rank": competition_rank(rewards, seat),
            "error": error,
        }
        records.append(record)
        print(
            f"{opponent_name} {players}p seed={game_seed} seat={seat} "
            f"{outcome} rank={record['rank']} rewards={rewards} steps={steps} "
            f"({elapsed:.1f}s)",
            flush=True,
        )
    return records


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    agent_path = Path(args.agent)
    if not agent_path.is_absolute():
        agent_path = (repo_root / agent_path).resolve()
    baseline_path = Path(args.baseline)
    if not baseline_path.is_absolute():
        baseline_path = (repo_root / baseline_path).resolve()

    # Importing the package root lets exported agents resolve orbit_board_bc_*.
    root_str = str(repo_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    our_module = _load_module(agent_path, "orbit_wars_eval_agent")
    baseline_module = _load_module(baseline_path, "orbit_wars_eval_baseline")
    our_agent = our_module.agent
    baseline_agent = baseline_module.agent

    groups: dict[str, list[dict[str, Any]]] = {}
    if args.games > 0:
        groups["two_player_vs_random"] = run_matchup(
            our_agent, "random", baseline_agent, args.games, args.seed, 2, args.episode_steps
        )
        groups["two_player_vs_baseline"] = run_matchup(
            our_agent,
            "baseline",
            baseline_agent,
            args.games,
            args.seed + 1_000_000,
            2,
            args.episode_steps,
        )
    if args.four_player_games > 0:
        groups["four_player_vs_random"] = run_matchup(
            our_agent,
            "random",
            baseline_agent,
            args.four_player_games,
            args.seed + 2_000_000,
            4,
            args.episode_steps,
        )

    # The embedded checkpoint loads on the first action.
    identity = _agent_identity(our_module)
    digest = hashlib.sha256(agent_path.read_bytes()).hexdigest()
    report = {
        "environment": "kaggle_environments.orbit_wars",
        "agent_path": str(agent_path),
        "agent_sha256": digest,
        "agent_bytes": agent_path.stat().st_size,
        "baseline_path": str(baseline_path),
        "baseline_description": "Nearest-planet sniper from competitiondata/main.py",
        "episode_steps": args.episode_steps,
        "seed_start": args.seed,
        "seat_rotation": "seat = game_index % players",
        "win_definition": "unique highest environment reward",
        "rank_definition": "1 + number of players with a strictly higher reward",
        "agent": identity,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "matchups": {name: {"summary": summarize_games(games), "games": games} for name, games in groups.items()},
    }
    try:
        import torch

        report["runtime"]["torch"] = torch.__version__
    except Exception:
        report["runtime"]["torch"] = None
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m orbit_board_bc_train.local_match")
    parser.add_argument("--agent", default="submission/main.py", help="Exported agent file with agent(obs)")
    parser.add_argument(
        "--baseline",
        default="competitiondata/main.py",
        help="Heuristic opponent used for the baseline matchup",
    )
    parser.add_argument("--games", type=int, default=4, help="2-player games per opponent. Seats alternate.")
    parser.add_argument("--four-player-games", type=int, default=4, help="4-player games versus three random agents")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--episode-steps",
        type=int,
        default=None,
        help="Override the environment episode length. Default is the Orbit Wars 500-turn game.",
    )
    parser.add_argument("--out", default="eval_results/local_matches.json")
    args = parser.parse_args(argv)
    report = evaluate(args)
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parents[1] / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: group["summary"] for name, group in report["matchups"].items()}, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
