#!/bin/bash

# Define the base path for the checkpoints
base_dir="baselines/ckpt/books/"

# Define the output file
output_file="output.csv"

# Array of available CUDA devices
cuda_devices=(0 1 2 3 4 5 6 7)

# Initialize device usage: 0 means available
device_in_use=(0 0 0 0 0 0 0 0)

# Array to keep track of job PIDs
declare -A pids

# Function to get the first available GPU index
get_available_device() {
    for idx in "${!cuda_devices[@]}"; do
        if [ "${device_in_use[$idx]}" -eq 0 ]; then
            device_in_use[$idx]=1
            return $idx
        fi
    done
}

# Wait for any process to finish and free its device
wait_for_process_and_free_device() {
    wait -n  # Wait for any job to finish
    # Find which job finished and free its GPU
    for idx in "${!pids[@]}"; do
        if ! kill -0 ${pids[$idx]} 2>/dev/null; then  # Process is no longer running
            device_in_use[$idx]=0
            unset pids[$idx]  # Remove PID from array
            echo "GPU ${cuda_devices[$idx]} is now free"
            break
        fi
    done
}

# Process each method
for method_path in "$base_dir"/*; do
    # if [ -d "$method_path" ]; then
    if [ -d "$method_path" ] && [[ "$(basename "$method_path")" == *"ga_gdr5e-6"* ]]; then
        method=$(basename "$method_path")
        model_dirs=""
        names=""
        
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
                checkpoint_path="$method_path/$checkpoint"
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
        cuda_device=${cuda_devices[device_idx]}

        command="CUDA_VISIBLE_DEVICES=$cuda_device python eval.py --model_dirs $model_dirs --names $names --corpus books --out_file baselines/ckpt/books/$method/$output_file"
        
        echo "Starting job on GPU $cuda_device: $command"
        eval $command &

        # Track PID of the job
        pids[$device_idx]=$!

        # If there are 8 active jobs, wait for one to finish before continuing
        [[ ${#pids[@]} -eq 8 ]] && wait_for_process_and_free_device
    fi
done


# Wait for all remaining jobs to complete
for pid in "${pids[@]}"; do
    wait $pid
    echo "Job with PID $pid has completed."
done

# echo "All jobs have completed."
# CUDA_VISIBLE_DEVICES=5 python eval.py --model_dirs muse-bench/MUSE-books_retrain --names base_model --corpus books --out_file baselines/ckpt/books/tv_0.0/output.csv