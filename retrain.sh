#!/bin/bash
#
# Full learning pipeline:
#   1. Rebuild player model from the most recent MAX_HITS hits
#   2. Continue training the Q-table for 5000 more episodes
#   3. Evaluate the updated Q-table vs rule-based baseline
#
# Usage:
#   ./retrain.sh
#
# Run after each play session to adapt to the player's evolving style.

set -e

cd "$(dirname "$0")"

echo "=========================================="
echo " step 1/3  rebuilding player model"
echo "=========================================="
python player_model.py

echo ""
echo "=========================================="
echo " step 2/3  continuing Q-table training"
echo "=========================================="
python train.py

echo ""
echo "=========================================="
echo " step 3/3  evaluating Q-table"
echo "=========================================="
python evaluate.py

echo ""
echo "=========================================="
echo " done. run 'python main.py' to play"
echo "=========================================="