"""
Frozen, greedy evaluation of a trained residual Q-table.

Runs N headless episodes with the Q-table corrections enabled and N with
them disabled, against two opponents (the current scripted opponent, and
a perturbed "held-out" opponent). Reports win/loss/draw rates for all
four combinations and appends a summary line to eval_results.jsonl.

This script is READ-ONLY on qtable.npy — it never writes back to the table.

Usage:
    python evaluate.py
    python evaluate.py --episodes 1000
    python evaluate.py --qtable qtable_good.npy
    python evaluate.py --perturbation 0.0    # skip held-out opponent
"""

import argparse
import hashlib
import json
import os
import time

import numpy as np

from physics_env import (
    AirHockeyWorld, TABLE_WIDTH, TABLE_HEIGHT,
    ROBOT_MIN_X, ROBOT_MAX_X, ROBOT_MIN_Y, ROBOT_MAX_Y,
    PUCK_RADIUS, PADDLE_RADIUS,
)
from sim_opponent import ScriptedOpponent
from strategy import _classical_plan
import qpolicy


# ---------- config ----------

DEFAULT_EPISODES = 500
DEFAULT_PERTURBATION = 0.25        # 25% stats perturbation for held-out opponent
RESULTS_LOG = "eval_results.jsonl"


# ---------- helpers ----------

class _StateView:
    """Minimal shim that _classical_plan expects."""
    pass


def _make_state_view(world):
    sv = _StateView()
    sv.puck_pos = tuple(world.puck_body.position)
    sv.puck_vel = tuple(world.puck_body.velocity)
    sv.robot_pos = tuple(world.robot_body.position)
    return sv


def _state_to_tablestate(world):
    """
    Build a TableState-like object with the fields qpolicy._discretize reads.
    """
    class _TS:
        pass
    ts = _TS()
    ts.puck_pos = tuple(world.puck_body.position)
    ts.puck_vel = tuple(world.puck_body.velocity)
    ts.robot_pos = tuple(world.robot_body.position)
    return ts


def _qtable_hash(path):
    """Short hash of qtable.npy for run identification."""
    if not os.path.exists(path):
        return "none"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


# ---------- episode ----------

def run_episode(world, opponent, qtable, use_qpolicy,
                max_steps=1200, dt=1 / 60):
    """
    One headless episode. If use_qpolicy is True and qtable is not None,
    corrections are applied. Otherwise, pure rule-based.

    Returns 'win' (robot scored), 'loss' (player scored), or 'draw'.
    """
    world.serve(toward='robot')

    for _ in range(max_steps):
        sv = _make_state_view(world)
        base_x, base_y = _classical_plan(sv)

        if use_qpolicy and qtable is not None:
            ts = _state_to_tablestate(world)
            s = qpolicy._state_index(qpolicy._discretize(ts))
            action_idx = int(np.argmax(qtable[s]))
            dx, dy = qpolicy.ACTIONS[action_idx]
            rx = base_x + dx * qpolicy.RESIDUAL_STEP
            ry = base_y + dy * qpolicy.RESIDUAL_STEP
        else:
            rx, ry = base_x, base_y

        rx = max(ROBOT_MIN_X, min(rx, ROBOT_MAX_X))
        ry = max(ROBOT_MIN_Y, min(ry, ROBOT_MAX_Y))
        world.set_robot_target(rx, ry, dt)

        # opponent
        class _S:
            pass
        s_opp = _S()
        s_opp.puck_pos = tuple(world.puck_body.position)
        s_opp.puck_vel = tuple(world.puck_body.velocity)
        raw_target = opponent.act(s_opp, dt)
        ox, oy = world.player_body.position
        tx = ox + (raw_target[0] - ox)
        ty = oy + (raw_target[1] - oy)
        world.set_player_target(tx, ty, dt)

        world.step(dt)

        goal = world.check_goal()
        if goal == 'robot':
            return 'win'
        if goal == 'player':
            return 'loss'

    return 'draw'


# ---------- batches ----------

def evaluate_batch(world, opponent, qtable, use_qpolicy, episodes, label):
    """Run `episodes` episodes and return win/loss/draw rates + avg length."""
    wins = losses = draws = 0
    total_steps = 0

    for _ in range(episodes):
        result = run_episode(world, opponent, qtable, use_qpolicy)
        if result == 'win':
            wins += 1
        elif result == 'loss':
            losses += 1
        else:
            draws += 1

    total = wins + losses + draws
    return {
        "label": label,
        "episodes": total,
        "win_rate": wins / total,
        "loss_rate": losses / total,
        "draw_rate": draws / total,
    }


def perturb_opponent(opponent, amount):
    """Return a copy of the opponent with stats jittered by ±amount fraction."""
    new = ScriptedOpponent.__new__(ScriptedOpponent)
    new.model = opponent.model
    new.home = opponent.home
    new.avg_speed = opponent.avg_speed * (1.0 + np.random.uniform(-amount, amount))
    new.attack_bias = float(np.clip(
        opponent.attack_bias * (1.0 + np.random.uniform(-amount, amount)),
        0.0, 1.0))
    new.corner_bias = float(np.clip(
        opponent.corner_bias * (1.0 + np.random.uniform(-amount, amount)),
        0.0, 1.0))
    return new


# ---------- reporting ----------

def print_summary(results):
    print()
    print(f"{'opponent':<14}{'policy':<22}{'win':>8}{'loss':>8}{'draw':>8}")
    print("-" * 60)
    for r in results:
        print(f"{r['opponent']:<14}{r['policy']:<22}"
              f"{r['win_rate']:>8.3f}{r['loss_rate']:>8.3f}{r['draw_rate']:>8.3f}")
    print()


def compute_delta(results):
    """Return the win-rate delta of (learned - baseline) per opponent."""
    deltas = {}
    for r in results:
        key = r["opponent"]
        deltas.setdefault(key, {})[r["policy"]] = r["win_rate"]
    out = {}
    for opponent, rates in deltas.items():
        if "baseline" in rates and "learned" in rates:
            out[opponent] = rates["learned"] - rates["baseline"]
    return out


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--qtable", type=str, default="qtable.npy")
    parser.add_argument("--player-model", type=str, default="player_model.json")
    parser.add_argument("--perturbation", type=float,
                        default=DEFAULT_PERTURBATION,
                        help="fraction to jitter held-out opponent stats; 0 disables")
    args = parser.parse_args()

    # load qtable (read-only)
    qtable = None
    if os.path.exists(args.qtable):
        try:
            qtable = np.load(args.qtable)
            if qtable.shape != (qpolicy.N_STATES, qpolicy.N_ACTIONS):
                print(f"warning: qtable shape {qtable.shape} does not match "
                      f"({qpolicy.N_STATES}, {qpolicy.N_ACTIONS}), ignoring")
                qtable = None
        except (OSError, ValueError) as e:
            print(f"could not load qtable: {e}")
            qtable = None
    else:
        print(f"warning: {args.qtable} not found — evaluating baseline only")

    # load player model into ScriptedOpponent
    from player_model import load_model
    model = load_model(args.player_model)
    if model is None:
        print(f"warning: {args.player_model} not found — using neutral opponent")
    train_opponent = ScriptedOpponent(model)

    # held-out opponent
    heldout_opponent = None
    if args.perturbation > 0:
        np.random.seed(int(time.time()))
        heldout_opponent = perturb_opponent(train_opponent, args.perturbation)

    world = AirHockeyWorld()
    results = []

    # --- training opponent ---
    print(f"evaluating {args.episodes} episodes vs training opponent…")
    r = evaluate_batch(world, train_opponent, None, False,
                       args.episodes, "baseline")
    r["opponent"] = "trained"
    r["policy"] = "baseline"
    results.append(r)

    if qtable is not None:
        r = evaluate_batch(world, train_opponent, qtable, True,
                           args.episodes, "learned")
        r["opponent"] = "trained"
        r["policy"] = "learned"
        results.append(r)

    # --- held-out opponent ---
    if heldout_opponent is not None:
        print(f"evaluating {args.episodes} episodes vs held-out opponent "
              f"(perturbation={args.perturbation:.2f})…")
        r = evaluate_batch(world, heldout_opponent, None, False,
                           args.episodes, "baseline")
        r["opponent"] = "held-out"
        r["policy"] = "baseline"
        results.append(r)

        if qtable is not None:
            r = evaluate_batch(world, heldout_opponent, qtable, True,
                               args.episodes, "learned")
            r["opponent"] = "held-out"
            r["policy"] = "learned"
            results.append(r)

    # --- summary ---
    print_summary(results)
    deltas = compute_delta(results)
    for opp, d in deltas.items():
        sign = "+" if d >= 0 else ""
        print(f"learned - baseline win rate ({opp}): {sign}{d:.3f}")
    print()

    # --- append to log ---
    record = {
        "timestamp": time.time(),
        "qtable_path": args.qtable,
        "qtable_hash": _qtable_hash(args.qtable),
        "episodes": args.episodes,
        "perturbation": args.perturbation,
        "results": results,
        "deltas": deltas,
    }
    with open(RESULTS_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"appended to {RESULTS_LOG}")


if __name__ == "__main__":
    main()