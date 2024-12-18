#!/bin/bash

# Define the base path for the checkpoints
base_dir="baselines/ckpt/books"

# Define the output file
output_file="output.csv"

# Only two GPUs available: 0 and 1
cuda_devices=(1)

# device_in_use stores how many jobs are running on each GPU
device_in_use=(0)

# Arrays to keep track of job PIDs and their associated GPU indices
declare -A pids

# Maximum concurrent jobs across all GPUs
max_concurrent_jobs=4

# Maximum jobs per GPU
max_jobs_per_gpu=2

# ------------------------------------------------------------
# Functions
# ------------------------------------------------------------

# Function to wait for any job to finish and free its device
wait_for_process_and_free_device() {
    wait -n  # Wait for any job to finish
    
    # Find which job finished and free its GPU slot
    for idx in "${!pids[@]}"; do
        pid="${pids[$idx]}"
        if ! kill -0 "$pid" 2>/dev/null; then
            # This job has finished
            device_in_use[$idx]=$((device_in_use[$idx]-1))
            unset pids[$idx]  # Remove PID from array
            echo "A job on GPU ${cuda_devices[$idx]} has completed. GPU load is now ${device_in_use[$idx]}."
            break
        fi
    done
}

# Function to get an available GPU index for a new job
get_available_device() {
    while true; do
        # Check if we can start a new job without exceeding concurrency
        current_jobs=0
        for usage in "${device_in_use[@]}"; do
            current_jobs=$((current_jobs+usage))
        done
        if [ "$current_jobs" -ge "$max_concurrent_jobs" ]; then
            # No slot for a new job, wait for one to complete
            wait_for_process_and_free_device
        fi

        # Check for available GPU
        for idx in "${!cuda_devices[@]}"; do
            if [ "${device_in_use[$idx]}" -lt "$max_jobs_per_gpu" ]; then
                return $idx
            fi
        done

        # If we get here, no GPU is free, but we have job slots. We must wait for a job on a busy GPU to complete.
        wait_for_process_and_free_device
    done
}

# ------------------------------------------------------------
# Prepare the task list
# ------------------------------------------------------------
tasks=()
# for method_path in "$base_dir"/*; do
#     if [ -d "$method_path" ]; then
#         tasks+=("$method_path")
#     fi
# done
for method_path in '/egr/research-optml/jiajingh/Unlearn-Simple/MUSE/baselines/ckpt/books/FT1e-4_combined2048_ft'; do
    tasks+=("$method_path")
done
# ------------------------------------------------------------
# Function to run a job for a given method_path
# ------------------------------------------------------------
run_job() {
    local method_path="$1"
    local method=$(basename "$method_path")
    local model_dirs=""
    local names=""
    
    # Sort potential checkpoints numerically
    checkpoints=($(ls "$method_path" 2>/dev/null | sort -t '-' -k 2 -n))

    # Filter out only directories
    dirs=()
    for item in "${checkpoints[@]}"; do
        if [ -d "$method_path/$item" ]; then
            dirs+=("$item")
        fi
    done

    # If we have checkpoint directories, use them; otherwise treat the method_path itself as the model directory
    if [ ${#dirs[@]} -gt 0 ]; then
        for checkpoint in "${dirs[@]}"; do
            model_dirs="$model_dirs \"baselines/ckpt/books/$method/$checkpoint\""
            names="$names \"$method$checkpoint\""
        done
    else
        # No checkpoint directories found; treat method_path itself as the model directory
        model_dirs="\"baselines/ckpt/books/$method\""
        names="\"$method\""
    fi

    # Find an available GPU and start a job
    get_available_device
    device_idx=$?
    cuda_device=${cuda_devices[$device_idx]}

    command="CUDA_VISIBLE_DEVICES=$cuda_device python eval.py --model_dirs $model_dirs --names $names --corpus books --out_file baselines/ckpt/books/$method/$output_file"
    
    echo "Starting job on GPU $cuda_device: $command"
    eval $command &
    job_pid=$!

    # Increment usage count for this GPU
    device_in_use[$device_idx]=$((device_in_use[$device_idx]+1))
    pids[$device_idx]=$job_pid
}

# ------------------------------------------------------------
# Main execution loop
# ------------------------------------------------------------
# We will continuously try to keep 4 jobs running until all tasks are done.

while true; do
    # Count how many jobs are currently running
    current_jobs=0
    for usage in "${device_in_use[@]}"; do
        current_jobs=$((current_jobs+usage))
    done

    if [ ${#tasks[@]} -eq 0 ] && [ $current_jobs -eq 0 ]; then
        # No tasks left and no jobs running
        break
    fi

    if [ $current_jobs -lt $max_concurrent_jobs ] && [ ${#tasks[@]} -gt 0 ]; then
        # We can start a new job
        task="${tasks[0]}"
        tasks=("${tasks[@]:1}")  # Remove the task from the list
        run_job "$task"
    else
        # Either we have no tasks left or we are at max capacity
        # Wait for at least one job to finish to free a slot if tasks remain
        if [ $current_jobs -ge $max_concurrent_jobs ]; then
            wait_for_process_and_free_device
        else
            # No current jobs, no tasks, must be done
            break
        fi
    fi
done

echo "All jobs have completed."
