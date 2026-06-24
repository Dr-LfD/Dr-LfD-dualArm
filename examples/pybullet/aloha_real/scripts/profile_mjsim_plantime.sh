#!/bin/bash

# Function to handle SIGINT (Ctrl+C)
handle_sigint() {
    echo "Script terminated by user (Ctrl+C)."
    exit 1
}

# Trap SIGINT signal
trap handle_sigint SIGINT

for i in {1..5}
do
    echo "Running interleaved_sim.py - Iteration $i"
    timeout 180s python3 interleaved_sim.py
    exit_status=$?
    if [ $exit_status -eq 124 ]; then
        echo "Iteration $i was terminated because it exceeded 2 minutes."
    elif [ $exit_status -eq 130 ]; then
        echo "Iteration $i was interrupted by user (Ctrl+C)."
        exit 1
    else
        echo "Iteration $i completed successfully."
    fi
done