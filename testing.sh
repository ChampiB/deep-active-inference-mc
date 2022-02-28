#!/bin/bash
#SBATCH --mail-type=END
source /cluster/home/cug/tmac3/.venv/Deep_AI_MCTS/bin/activate
python3 testing.py -n /cluster/home/cug/tmac3/deep-active-inference-mc/figs_final_model_0.01_30_1.0_50_10_5/checkpoints/
