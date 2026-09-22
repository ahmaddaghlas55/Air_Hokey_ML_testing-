# Autonomous Air Hockey

A simulated air hockey table where a robot plays against a human on
opposite halves. The robot uses a hand-written attack-and-defend
policy, refined by residual tabular Q-learning, and adapts to the
specific player over time via an online learning loop.

The physics and strategy are developed entirely in software against a
`pygame` + `pymunk` simulation. The same strategy layer is intended to
be swapped onto real hardware (Raspberry Pi + two geared rotational
arms) behind a single interface, without changing any of the code
above the interface.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [What You Need](#what-you-need)
3. [How the Game Works](#how-the-game-works)
4. [Project Layout](#project-layout)
5. [The Learning Pipeline](#the-learning-pipeline)
6. [Command Reference](#command-reference)
7. [Understanding the Output](#understanding-the-output)
8. [Tuning and Debugging](#tuning-and-debugging)
9. [Design Notes](#design-notes)
10. [Known Issues](#known-issues)

---

## Quick Start

```bash
# 1. clone / enter the project
cd AirHockeyEmulator

# 2. create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # on Windows: venv\Scripts\activate

# 3. install dependencies
pip install pygame pymunk numpy

# 4. play the game
python main.py
```

Move your mouse to control the orange paddle on the left. First to 7
goals wins. Press **R** to restart, **ESC** or close the window to quit.

To run the full learning cycle (train and evaluate), see
[The Learning Pipeline](#the-learning-pipeline).

---

## What You Need

- **Python 3.10+** (tested on 3.12)
- **pygame** — rendering and input
- **pymunk** — 2D physics
- **numpy** — Q-table storage and math

Install everything with:

```bash
pip install pygame pymunk numpy
```

Tested under WSL2 + WSLg, and natively on Windows and Linux. macOS is
untested but should work — the dependencies are cross-platform.

### WSL Notes

- The pygame window should appear under WSLg without extra setup.
- USB gamepads are **not** passed through to WSL by default. If you
  want to control the paddle with an Xbox controller rather than the
  mouse, either run the project natively on Windows, or attach the
  controller to WSL with `usbipd-win`.

---

## How the Game Works

### The Table

- **1000 × 500 pixels**, horizontal orientation
- **Left half** (`x ∈ [0, 500]`) — the **player**, controlled by mouse
- **Right half** (`x ∈ [500, 1000]`) — the **robot**, controlled by code
- A **goal gap** at the vertical middle of each side wall, 160 px tall
- Puck radius 15 px, paddle radius 20 px

**Scoring:** if the puck enters the left goal, the **robot** scores.
If it enters the right goal, the **player** scores. First to 7 wins.

### The Robot's Policy

Every frame, the robot decides where to move its paddle in three
layers:

1. **Rule-based target** (`strategy._classical_plan`)
   - If the puck is on the robot's side: **attack** — stand behind
     the puck and drive it toward the player's goal.
   - If the puck is on the player's side: **defend** — predict where
     the puck will cross the robot's defensive line and intercept it.
2. **Residual Q-learning correction** (optional, if `qtable.npy` exists)
   - Chooses one of 9 small shifts (up to 15 px) to apply to the
     rule-based target.
3. **Difficulty noise** — small random jitter, controlled per level.

If `qtable.npy` doesn't exist, layers 2 is skipped and the robot plays
pure rule-based. That's the baseline.

### The Player Model

Every paddle-puck contact is logged to `opponent_hits.jsonl`. The
`player_model.py` script summarizes the last 500 hits into a statistical
fingerprint of how the player plays:

- Average shot speed
- Attack bias (fraction of shots directed at the robot)
- Positional zone histograms
- Average reaction gap between hits

This model is used in two places:

1. To build a **scripted opponent** (`sim_opponent.py`) that mimics the
   player, used for training the Q-table.
2. To set **difficulty parameters** (`difficulty.py`) — reaction time,
   paddle speed, aim noise — at runtime.

---

## Project Layout

```
AirHockeyEmulator/
├── main.py                Entry point — runs the game loop
├── sim_backend.py         pygame rendering, mouse input, HUD, scoring
├── physics_env.py         pymunk world: puck, paddles, walls, goals
├── interface.py           Abstract TableBackend, TableState
├── predictor.py           Trajectory prediction with one wall bounce
├── strategy.py            Rule-based attack/defend + residual Q hook
├── qpolicy.py             Runtime lookup for the residual Q-table
├── difficulty.py          Player model → difficulty parameters
├── player_model.py        Summarizes opponent_hits.jsonl
├── sim_opponent.py        Scripted opponent built from player_model.json
├── train.py               Headless residual RL trainer (warm-starts)
├── evaluate.py            Frozen, greedy evaluation of a qtable
├── retrain.sh             Full pipeline: model → train → evaluate
├── PROJECT_STATUS.md      Deeper architecture and history document
├── README.md              This file
│
├── opponent_hits.jsonl    (generated) append-only log of player hits
├── player_model.json      (generated) player fingerprint
├── qtable.npy             (generated) trained residual Q-table
└── eval_results.jsonl     (generated) accumulated evaluation records
```

---

## The Learning Pipeline

The project has two loops: a **short loop** (play, retrain, play) for
online learning, and a **long loop** (train from scratch) for initial
setup.

### Initial Setup (once)

```bash
# 1. play a few matches so the log has data
python main.py

# 2. build the player model from the hits you just generated
python player_model.py

# 3. train the Q-table from scratch
python train.py
```

After step 3, `qtable.npy` exists. The next run of `main.py` will use
it automatically.

### Everyday Use (each session)

```bash
# 1. play a few matches
python main.py

# 2. run the full pipeline: rebuild model, continue training, evaluate
./retrain.sh
```

`retrain.sh` does three things:

1. **`python player_model.py`** — rebuilds the player model from the
   **last 500 hits** (sliding window, so it follows your evolving style).
2. **`python train.py`** — warm-starts from the existing `qtable.npy`
   and runs 5000 more episodes. The Q-table accumulates experience
   across sessions.
3. **`python evaluate.py`** — runs 500 greedy episodes with and without
   the Q-table, against both the current opponent and a held-out
   perturbed opponent, and appends the results to `eval_results.jsonl`.

Every cycle prints a comparison table so you can see whether the latest
training run helped, hurt, or didn't move the needle.

### Why Residual RL

Earlier attempts used direct tabular Q-learning from scratch and failed
repeatedly: the policy never improved on random. Residual RL — where
the Q-table only learns small corrections to a working rule-based
policy — was the first approach that worked. The rule-based policy
prevents the Q-table from being stuck in states with no positive reward
signal, and the bounded corrections make the worst-case behavior safe.

See `PROJECT_STATUS.md` for the full history of what was tried and what
failed.

---

## Command Reference

### Playing

| Command | What it does |
|---|---|
| `python main.py` | Play the game with the current policy |

### Learning

| Command | What it does |
|---|---|
| `python player_model.py` | Rebuild `player_model.json` from recent hits |
| `python train.py` | Train the Q-table (warm-starts from `qtable.npy`) |
| `python evaluate.py` | Measure current Q-table vs rule-based baseline |
| `./retrain.sh` | Run all three in order |

### Evaluation Options

```bash
python evaluate.py --episodes 1000          # more samples per condition
python evaluate.py --qtable qtable_good.npy # evaluate a backup table
python evaluate.py --perturbation 0.0       # skip held-out opponent
```

### Backups and Resets

```bash
cp qtable.npy qtable_good.npy    # back up a good Q-table
cp qtable_good.npy qtable.npy    # restore it later

rm qtable.npy                    # force pure rule-based play
rm opponent_hits.jsonl           # reset the player log
rm player_model.json             # reset the player model
```

---

## Understanding the Output

### `player_model.py`

Prints your fingerprint:

```
model built from 189 hits (last 500 window)
  avg speed:    1013 px/s
  avg angle:    44.7 deg
  attack bias:  0.20  (1.0 = always toward robot)
  corner bias:  0.11  (1.0 = always from a corner)
  x zones:      [13, 29, 28, 67, 52]
  y zones:      [9, 41, 46, 59, 34]
  avg gap:      2463 ms
```

- `attack bias` near 0.5 means the player is balanced; near 1.0 means
  they always aim at the robot; near 0 means they mostly deflect.
- `x zones` and `y zones` are 5-bucket histograms. High values mean the
  player hits from that zone often.

### `train.py`

Prints a line every 500 episodes:

```
ep    500  win_rate=0.32  loss_rate=0.14  timeout_rate=0.54  eps=0.29
ep   2000  win_rate=0.44  loss_rate=0.11  timeout_rate=0.46  eps=0.27
...
```

- `win_rate` — fraction of episodes where the robot scored
- `loss_rate` — fraction where the player scored
- `timeout_rate` — fraction that ended without a goal
- `eps` — exploration rate (1.0 = all random, 0.05 = almost all greedy)

**What good looks like:** `win_rate` starts around 0.30 and climbs.
`loss_rate` stays below 0.20. `timeout_rate` shrinks as the robot
learns to score.

At the end, it prints the action distribution:

```
action distribution in final policy:
  (+0,+0) : 214 states       ← trust the rule-based policy
  (-1,+0) :  86 states       ← shift 15 px left
  ...
action 0 (no correction) chosen in 33.4% of states
```

`action 0` being chosen in 30–70% of states is healthy — the Q-table
mostly trusts the rule-based policy but corrects it sometimes. If it's
near 100%, the Q-table isn't learning anything. If it's near 0%, the
Q-table is overriding the rule-based policy everywhere, which is risky.

### `evaluate.py`

Prints a comparison:

```
opponent      policy                     win    loss    draw
------------------------------------------------------------
trained       baseline                 0.320   0.100   0.580
trained       learned                  0.430   0.095   0.475
held-out      baseline                 0.295   0.115   0.590
held-out      learned                  0.360   0.120   0.520

learned - baseline win rate (trained): +0.110
learned - baseline win rate (held-out): +0.065
```

**What good looks like:**

- `learned - baseline` is positive for both opponents.
- The **trained-opponent delta is larger than the held-out delta** — this
  shows the learned policy has specialized to the training opponent
  (which is fine and expected; the ideal is not zero overfitting).

If the trained and held-out deltas are identical, something is wrong —
either the perturbation is too small, or the baseline is broken (see
[Known Issues](#known-issues)).

### `eval_results.jsonl`

One JSON object per `evaluate.py` run. Accumulate these over weeks and
plot the deltas to see whether the online learning loop is trending
upward over time.

---

## Tuning and Debugging

### The Game Feels Too Fast / Too Slow

Edit `main.py`:

```python
TIME_SCALE = 0.6    # 0.5 = half speed, 1.0 = real, 2.0 = double
```

### The Robot Is Too Easy / Too Hard

Edit `physics_env.py`:

```python
ROBOT_MAX_SPEED = 1200.0    # px/s — how fast the robot paddle can move
```

Or edit `difficulty.py` for per-difficulty-level presets.

### Training Is Stuck

Look at the first 2000 episodes. If `win_rate` is flat:

1. Check that the rule-based policy works. Delete `qtable.npy` and
   play `python main.py`. The robot should score sometimes.
2. If the robot never scores even in attack mode, check
   `ATTACK_STANDOFF` in `strategy.py`. It must be **smaller than**
   `PUCK_RADIUS + PADDLE_RADIUS` (i.e. less than 35) for the paddle to
   actually make contact.

### Training Is Making Things Worse

RL is noisy. Back up good tables:

```bash
cp qtable.npy qtable_good.npy
```

Restore a known-good version:

```bash
cp qtable_good.npy qtable.npy
```

### Evaluate Reports Identical Deltas for Trained and Held-Out

Increase the perturbation:

```bash
python evaluate.py --perturbation 0.5
```

If deltas are still identical, the baseline is likely broken — see
[Known Issues](#known-issues).

---

## Design Notes

### Why a Simulation-First Pipeline

The ML strategy piece needs a lot of practice to develop. Doing that
against half-built hardware is expensive and slow. Instead, the whole
system was developed against a physics simulation, with the strategy
layer talking to one interface (`TableBackend`). Swapping in the real
hardware later means implementing a new backend, not changing any of
the strategy, model, or training code.

### Why Paddles Are Kinematic Bodies

Paddles are `pymunk.Body.KINEMATIC`, which means their position is
driven by setting velocity and letting pymunk integrate it during
`space.step(dt)`. This is important: setting both `position` and
`velocity` on a kinematic body causes pymunk to double-integrate, and
the paddle shoots off-screen. Any changes to `_drive_kinematic` must
preserve the invariant that `space.step(dt)` moves the paddle by at
most `velocity * dt`.

### Why Residual RL Instead of Direct RL

Direct tabular Q-learning (state → action → target position) failed
repeatedly in this project. Sparse terminal reward and coarse state
discretization meant the Q-table never learned useful behavior over
20,000 episodes. Residual RL — where the Q-table learns only small
corrections to a working rule-based policy — succeeded immediately.
The rule-based baseline prevents the Q-table from being stuck with no
signal, and the bounded correction size makes learning safer.

See `PROJECT_STATUS.md` for the full history.

---

## Known Issues

### Baseline Attack Policy Might Not Make Contact

The attack standoff (`ATTACK_STANDOFF` in `strategy.py`) determines
how far behind the puck the robot stands before pushing it. If this
value exceeds `PUCK_RADIUS + PADDLE_RADIUS` (35 px), the paddle never
makes contact, and the rule-based policy barely scores.

**Symptom:** The `baseline` row in `evaluate.py` shows win rate under
0.10 and draw rate above 0.70.

**Fix:** Set `ATTACK_STANDOFF` below 35. A value like 15 works well.

### Evaluation Can Look Better Than It Is

If the baseline policy is broken (see above), any working policy —
including the Q-table corrections — will look enormously better by
comparison. Always check that the baseline is a competent player
before trusting the delta in `evaluate.py`.

### Scripted Opponent Is Not the Real Player

The scripted opponent is built from the last 500 hits of your play. It
approximates you, but has no strategic intent — it just chases the puck
at your average speed and shoots from your average zone. It will miss
things a real human wouldn't, and vice versa.

This means: **training against the scripted opponent and playing
against it in evaluation are not the same as playing against you.** The
held-out opponent condition in `evaluate.py` partially addresses this
by perturbing the opponent's stats, but it's a rough proxy.

Real evaluation happens when you play `python main.py` yourself.

---

## Further Reading

- **`PROJECT_STATUS.md`** — deeper architecture document, full history
  of what was tried, what failed, and why
- **`strategy.py`** — the rule-based policy, well-commented
- **`train.py`** — the residual RL training loop, well-commented
- **`evaluate.py`** — how the trained policy is measured

---

## License

[MIT or your choice — fill in here]