#!/bin/bash

# Command below is used to redirect the output to the log file and not to the console:
# "stdbuf -oL -eL"

# for i in {1..4}; do
#     echo "Training student $i..."
#     stdbuf -oL -eL uv run run_experiment.py train_students_$i > train_students_$i.log 2>&1
# done

# for i in {0..4}; do
#     echo "Generating flips $i..."
#     stdbuf -oL -eL uv run run_experiment.py generate_flips_$i > generate_flips_$i.log 2>&1
# done

echo "Training users 0 and 1..."
stdbuf -oL -eL uv run run_experiment.py train_users_0 > train_users_0_300.log 2>&1 &
stdbuf -oL -eL uv run run_experiment.py train_users_1 > train_users_0_500.log 2>&1 &

wait

stdbuf -oL -eL uv run run_experiment.py train_users_2 > train_users_0_1000.log 2>&1 &
stdbuf -oL -eL uv run run_experiment.py train_users_3 > train_users_0_1500.log 2>&1 &
