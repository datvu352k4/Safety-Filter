#!/bin/bash

# Script to automate Go2 Navigation Benchmarks
# Seeds 1 to 10, with and without Safety Filter for both MPPI and DWA.

echo "Starting Navigation Benchmarks Automation..."

for map_name in map2
do
    echo "########################################"
    echo "RUNNING MAP: $map_name"
    echo "########################################"
    
    for seed in {1..30}
    do
        echo "========================================"
        echo "MAP: $map_name | SEED: $seed"
        echo "========================================"

        # --- MPPI ---
        echo "[$map_name | MPPI] Case 1: No Safety Filter..."
        python legged_gym/scripts/play_go2_mppi_terrain.py --map $map_name --seed $seed --no_safety_filter --headless
        
        echo "[$map_name | MPPI] Case 2: With Safety Filter..."
        python legged_gym/scripts/play_go2_mppi_terrain.py --map $map_name --seed $seed --safety_filter --headless

        # --- DWA ---
        # echo "[$map_name | DWA] Case 3: No Safety Filter..."
        # python legged_gym/scripts/play_go2_dwa_terrain.py --map $map_name --seed $seed --no_safety_filter --headless

        # echo "[$map_name | DWA] Case 4: With Safety Filter..."
        # python legged_gym/scripts/play_go2_dwa_terrain.py --map $map_name --seed $seed --safety_filter --headless
    done
done

echo "Benchmarks completed. Results saved in /home/datvu/LeggedGym-Ex/test_results/"
