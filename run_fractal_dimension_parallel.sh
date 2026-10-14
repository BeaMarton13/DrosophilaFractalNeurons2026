#!/bin/bash
# filepath: /Users/bea/Documents/work/research/drosophila/run_fractal_dimension_parallel.sh

# Configuration
TOTAL_IDS=127507  # Total number of IDs to process (adjust as needed)
NUM_PARALLEL=120  # Number of parallel jobs

# Calculate batch size based on number of parallel jobs
BATCH_SIZE=$(( (TOTAL_IDS + NUM_PARALLEL - 1) / NUM_PARALLEL ))

echo "Starting parallel fractal dimension calculation"
echo "Total IDs: $TOTAL_IDS, Batch size: $BATCH_SIZE, Parallel jobs: $NUM_PARALLEL"

# Generate exactly NUM_PARALLEL start indices and run in parallel
for i in $(seq 0 $((NUM_PARALLEL - 1))); do
    START=$((i * BATCH_SIZE))
    echo "Starting job $((i + 1)) with start index $START"
    python calculate_fractal_dimension_main.py "$START" "$BATCH_SIZE" &
done

# Wait for all background jobs to complete
wait

echo "All jobs completed"
