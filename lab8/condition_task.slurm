#!/usr/bin/env zsh
#SBATCH --partition=instruction
#SBATCH --time=00:03:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=condition_task.output

cd $SLURM_SUBMIT_DIR

# 1. FIX: Use the specific version found.
module load gcc/7.5.0 

# 2. FIX: Corrected compilation command:
#    - Added -O3 optimization.
#    - Fixed the misplaced backslash (\) and spacing in the path.
#    - Added -pthread for Taskflow's multi-threading support.
g++ -std=c++17 -O3 -I/srv/home/eang2/ECE455lab/lab8/taskflow condition_task.cpp -o condition_task -pthread

./condition_task