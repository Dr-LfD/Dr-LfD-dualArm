#!/bin/bash

# Function to handle SIGINT (Ctrl+C)
handle_sigint() {
    echo "Script terminated by user (Ctrl+C)."
    exit 1
}

# Trap SIGINT signal
trap handle_sigint SIGINT

declare -a results
declare -a output_files

for i in {1..10}
do
    echo "Running interleaved_dmg - Iteration $i"
    tmpfile=$(mktemp)
    output_files[$i]=$tmpfile

    python3 interleaved_dmg_plugin.py --task_name two_arm_threading  2>&1 | tee "$tmpfile"
    exit_status=${PIPESTATUS[0]}

    if [ $exit_status -eq 130 ]; then
        results[$i]="Iteration $i: Interrupted by user (Ctrl+C)"
        echo "${results[$i]}"
        break
    elif [ $exit_status -eq 0 ]; then
        results[$i]="Iteration $i: Success"
    else
        results[$i]="Iteration $i: Failed (exit code $exit_status)"
    fi
    echo "${results[$i]}"
done

echo
echo "===== Summary Report ====="
for i in "${!results[@]}"; do
    echo "${results[$i]}"
    echo "Output:"
    cat "${output_files[$i]}"
    echo "-------------------------"
    rm -f "${output_files[$i]}"
done